"""Leakage-safe GLUE loading and single-label-token supervised training data."""
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .tasks import TASKS, label_letter, make_prompt_text


@dataclass
class PreparedData:
    train: Any
    tune: Any
    final: Any
    metadata: dict
    token_ids: tuple[int, int]


def get_label_token_ids(tokenizer) -> tuple[int, int]:
    """Fail rather than silently use multi-token labels in this experiment."""
    token_ids = []
    for letter in ("A", "B"):
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(
                f"Label {letter!r} tokenized into {tokens}; choose a different "
                "single-token label scheme for this model and rerun all methods."
            )
        token_ids.append(tokens[0])
    if token_ids[0] == token_ids[1]:
        raise ValueError("A and B must map to distinct token IDs")
    return tuple(token_ids)


def prompt_ids(tokenizer, text: str) -> list[int]:
    """Return a plain list of token IDs."""

    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )

    # Handle newer Transformers versions returning BatchEncoding.
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]

    # Handle a possible batch dimension.
    if hasattr(result, "tolist"):
        result = result.tolist()

    if result and isinstance(result[0], list):
        result = result[0]

    return result

def encode_row(tokenizer, task: str, row: dict, token_ids, max_length: int):
    prompt = prompt_ids(tokenizer, make_prompt_text(task, row))
    was_truncated = len(prompt) + 1 > max_length
    if was_truncated:
        # Preserve the initial task instructions AND the final assistant header.
        # Middle truncation is deterministic; count/report how often it happens.
        keep = max_length - 1
        head = min(48, keep // 3)
        prompt = prompt[:head] + prompt[-(keep - head):]
    target = token_ids[int(row["label"])]
    # Only the final answer token has a training loss. This is a next-token
    # classification objective (not an entire free-text generation objective).
    ids = prompt + [target]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt) + [target],
        "truncated": was_truncated,
    }


def _cap(ds, maximum: int | None, seed: int):
    if maximum is None or maximum >= len(ds):
        return ds
    # Fixed subset independent of model, adapter or training seed.
    return ds.shuffle(seed=seed).select(range(maximum))


def prepare_data(cfg: dict, tokenizer) -> PreparedData:
    from datasets import Dataset, load_dataset

    task = cfg["dataset"]["task"]
    if task not in TASKS:
        raise ValueError(f"Unsupported classification task {task!r}; expected {list(TASKS)}")

    ds_cfg = cfg["dataset"]
    split_seed = int(ds_cfg["split_seed"])
    raw = load_dataset(ds_cfg["id"], task, revision=ds_cfg.get("revision", "main"))
    # Official GLUE test labels are hidden; official labeled validation is our final local test.
    # Split original training set to obtain a separate checkpoint-selection set.
    parts = raw["train"].train_test_split(
        test_size=float(ds_cfg["tune_fraction"]), seed=split_seed,
        stratify_by_column="label",
    )
    raw_splits = {
        "train": _cap(parts["train"], ds_cfg.get("max_train_samples"), split_seed),
        "tune": _cap(parts["test"], ds_cfg.get("max_tune_samples"), split_seed),
        "final": _cap(raw["validation"], ds_cfg.get("max_final_samples"), split_seed),
    }
    ids = get_label_token_ids(tokenizer)
    metadata = {
        "dataset_id": ds_cfg["id"],
        "dataset_revision_requested": ds_cfg.get("revision", "main"),
        "task": task,
        "split_seed": split_seed,
        "split_names": {"train": "internal train subset of GLUE train",
                        "tune": "internal holdout subset of GLUE train",
                        "final": "official labeled GLUE validation"},
        "raw_train_fingerprint": raw["train"]._fingerprint,
        "raw_validation_fingerprint": raw["validation"]._fingerprint,
        "label_map": {"A": {"glue_label": 0, "meaning": TASKS[task].labels[0]},
                      "B": {"glue_label": 1, "meaning": TASKS[task].labels[1]}},
        "label_token_ids": ids,
        "splits": {},
    }
    encoded = {}
    for name, ds in raw_splits.items():
        rows = [encode_row(tokenizer, task, item, ids, int(cfg["training"]["max_length"]))
                for item in ds]
        truncated_count = sum(row.pop("truncated") for row in rows)
        if not rows:
            raise ValueError(f"Empty dataset split {name}")
        metadata["splits"][name] = {
            "n": len(rows),
            "label_counts": dict(Counter(int(item["label"]) for item in ds)),
            "truncated_count": truncated_count,
            "tokens_per_one_pass": sum(len(r["input_ids"]) for r in rows),
        }
        encoded[name] = Dataset.from_list(rows)
    return PreparedData(encoded["train"], encoded["tune"], encoded["final"], metadata, ids)


class CausalLabelCollator:
    """Right-padding collator: all non-target labels use -100 to mask loss."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        n = max(len(ex["input_ids"]) for ex in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in examples:
            pad = n - len(ex["input_ids"])
            batch["input_ids"].append(ex["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(ex["attention_mask"] + [0] * pad)
            batch["labels"].append(ex["labels"] + [-100] * pad)
        return {name: torch.tensor(rows, dtype=torch.long) for name, rows in batch.items()}
