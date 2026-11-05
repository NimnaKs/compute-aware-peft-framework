"""Train/evaluate one controlled LoRA or QLoRA experiment.

Examples:
  python -m peft_bench.run --config configs/pilot.yaml --task sst2 --method lora --rank 16 --seed 42
  python -m peft_bench.run --config configs/pilot.yaml --task sst2 --method qlora --rank 16 --seed 42
"""
import argparse
import copy
import math
import sys
import time
import traceback
from pathlib import Path

from .utils import (dependency_versions, dump_json, experiment_directory,
                    load_config, now_utc)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--task", choices=["sst2", "mrpc", "qnli"], default="sst2")
    parser.add_argument("--method", choices=["lora", "qlora"], required=True)
    parser.add_argument("--rank", type=int, choices=[4, 8, 16, 32], default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None, help="Override train sample cap")
    parser.add_argument("--max-tune", type=int, default=None, help="Override tune sample cap")
    parser.add_argument("--max-final", type=int, default=None, help="Override final eval sample cap")
    parser.add_argument("--usd-per-hour", type=float, default=None,
                        help="Actual instance hourly rate; don't guess this value")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run")
    return parser


def run_one(args):
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from .data import CausalLabelCollator, prepare_data
    from .evaluation import make_compute_metrics, make_logit_preprocessor
    from .modeling import load_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU. Use python -m peft_bench.environment to check.")

    cfg = copy.deepcopy(load_config(args.config))
    cfg["dataset"]["task"] = args.task
    for key, value in [
        ("max_train_samples", args.max_train),
        ("max_tune_samples", args.max_tune),
        ("max_final_samples", args.max_final),
    ]:
        if value is not None:
            cfg["dataset"][key] = value
    for split_name in ("max_train_samples", "max_tune_samples", "max_final_samples"):
        val = cfg["dataset"].get(split_name)
        if val is not None and val < 1:
            raise ValueError(f"{split_name} must be > 0 or null")

    outdir = experiment_directory(cfg, args.method, args.rank, args.seed)
    if (outdir / "metrics.json").exists() and not args.force:
        raise FileExistsError(f"{outdir} already exists. Use --force only if overwriting is intended.")
    outdir.mkdir(parents=True, exist_ok=True)
    start_whole = time.perf_counter()
    record = {
        "status": "running", "start_utc": now_utc(),
        "model": cfg["model"]["id"], "task": args.task,
        "method": args.method, "rank": args.rank, "seed": args.seed,
        "alpha": args.rank * cfg["training"]["alpha_multiplier"],
        "usd_per_hour": args.usd_per_hour,
        "versions": dependency_versions(),
    }
    dump_json(outdir / "config_resolved.json", cfg)
    dump_json(outdir / "metrics.json", record)

    try:
        set_seed(args.seed)
        gpu_name = torch.cuda.get_device_name(0)
        use_bf16 = bool(torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        record.update({"gpu": gpu_name, "gpu_vram_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
            "compute_precision": "bf16" if use_bf16 else "fp16"})
        print(f"[START] {args.method} rank={args.rank} on {gpu_name} / {cfg['model']['id']}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["id"], revision=cfg["model"].get("revision", "main"),
            use_fast=True,
        )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("This implementation requires an instruct model with a chat template.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        data = prepare_data(cfg, tokenizer)
        record["data"] = data.metadata
        print("Dataset split counts:", {name: x["n"] for name, x in data.metadata["splits"].items()}, flush=True)

        model, adapter_cfg, trainable = load_peft_model(cfg, args.method, args.rank, dtype)
        record["model_revision_resolved"] = getattr(model.config, "_commit_hash", None)
        record["trainable_parameters"] = int(trainable)
        print("Trainable parameters:", f"{trainable:,}", flush=True)
        model.print_trainable_parameters()
        print("LoRA adapter configuration:", adapter_cfg, flush=True)

        train_args = TrainingArguments(
            output_dir=str(outdir / "checkpoints"),
            num_train_epochs=float(cfg["training"]["epochs"]),
            learning_rate=float(cfg["training"]["learning_rate"]),
            per_device_train_batch_size=int(cfg["training"]["micro_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
            gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            max_grad_norm=1.0,
            bf16=use_bf16,
            fp16=not use_bf16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=int(cfg["training"].get("save_total_limit", 1)),
            logging_strategy="steps",
            logging_steps=int(cfg["training"].get("logging_steps", 25)),
            eval_accumulation_steps=8,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            save_safetensors=True,
            seed=args.seed,
            data_seed=args.seed,
            disable_tqdm=False,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            processing_class=tokenizer,
            train_dataset=data.train,
            eval_dataset=data.tune,
            data_collator=CausalLabelCollator(tokenizer.pad_token_id),
            preprocess_logits_for_metrics=make_logit_preprocessor(data.token_ids),
            compute_metrics=make_compute_metrics(data.token_ids, args.task),
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_result = trainer.train()
        torch.cuda.synchronize()
        training_wall_time_s = time.perf_counter() - t0
        peak_alloc_gib = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3

        # Final labeled official GLUE validation evaluated AFTER checkpoint selection.
        # For publication, this is our LOCAL test set, not the hidden official GLUE test.
        final_metrics = trainer.evaluate(eval_dataset=data.final, metric_key_prefix="final")
        adapter_dir = outdir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_bytes = sum(x.stat().st_size for x in adapter_dir.rglob("*") if x.is_file())

        # Estimate tokens processed from dataset length and completed epoch fraction.
        # This is an approximate end-to-end throughput that includes eval/save time.
        completed_epochs = float(trainer.state.epoch or cfg["training"]["epochs"])
        approximate_tokens = data.metadata["splits"]["train"]["tokens_per_one_pass"] * completed_epochs
        record.update({
            "status": "completed",
            "training_wall_time_s": round(training_wall_time_s, 3),
            "end_to_end_run_s": round(time.perf_counter() - start_whole, 3),
            "training_runtime_reported_s": train_result.metrics.get("train_runtime"),
            "estimated_input_tokens_processed": int(approximate_tokens),
            "estimated_end_to_end_tokens_per_sec": round(approximate_tokens / training_wall_time_s, 3),
            "completed_epochs": completed_epochs,
            "peak_cuda_allocated_gib": round(peak_alloc_gib, 4),
            "peak_cuda_reserved_gib": round(peak_reserved_gib, 4),
            "adapter_checkpoint_mib": round(adapter_bytes / 1024**2, 3),
            "tune_best_eval_loss": float(trainer.state.best_metric),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "final_accuracy": final_metrics.get("final_accuracy"),
            "final_macro_f1": final_metrics.get("final_macro_f1"),
            "final_mrpc_f1": final_metrics.get("final_f1"),
            "final_loss": final_metrics.get("final_loss"),
            "gpu_training_cost_usd": (round(training_wall_time_s * args.usd_per_hour / 3600, 4)
                                      if args.usd_per_hour is not None else None),
            "end_utc": now_utc(),
            "notes": "Peak CUDA metrics are process allocation/reservation, not whole-device NVML peak. "
                     "Throughput is estimated from tokenized samples and includes eval/save time. "
                     "Cost is compute-only runtime x provided hourly rate (no startup/storage).",
        })
        dump_json(outdir / "trainer_log_history.json", trainer.state.log_history)
        dump_json(outdir / "metrics.json", record)
        print(f"[DONE] {outdir}", flush=True)
        print(f"  Accuracy: {record['final_accuracy']:.4f} | Macro-F1: {record['final_macro_f1']:.4f}")
        print(f"  Peak alloc: {peak_alloc_gib:.2f} GiB | train duration: {training_wall_time_s/60:.2f} min")
        return 0

    except Exception as exc:
        record.update({
            "status": "oom" if "out of memory" in str(exc).lower() else "failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "end_utc": now_utc(),
            "end_to_end_run_s": round(time.perf_counter() - start_whole, 3),
        })
        dump_json(outdir / "metrics.json", record)
        (outdir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FAILED] {outdir}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main():
    args = create_parser().parse_args()
    if args.usd_per_hour is not None and args.usd_per_hour < 0:
        raise SystemExit("--usd-per-hour must be >= 0")
    if args.rank <= 0:
        raise SystemExit("Rank must be positive")
    sys.exit(run_one(args))


if __name__ == "__main__":
    main()
