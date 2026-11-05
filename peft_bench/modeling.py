"""Build exactly matched LoRA / QLoRA adapters (other than quantization)."""


def load_peft_model(cfg: dict, method: str, rank: int, compute_dtype):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if method not in ("lora", "qlora"):
        raise ValueError("method must be lora or qlora")
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU required for this study. Run on EC2 G4/G5 or similar.")

    model_cfg = cfg["model"]
    kwargs = {
        "revision": model_cfg.get("revision", "main"),
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if method == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        # Single-GPU placement: ensure both methods use the same device.
        kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], **kwargs)
    model.config.use_cache = False

    if method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    adapter_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=int(cfg["training"]["alpha_multiplier"]) * rank,
        lora_dropout=float(cfg["training"]["dropout"]),
        target_modules=cfg["training"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, adapter_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Quantized parameter .numel() may reflect packed storage rather than logical
    # pretrained parameter count; do NOT use it as a comparable percentage denominator.
    return model, adapter_cfg, trainable
