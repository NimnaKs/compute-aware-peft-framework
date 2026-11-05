"""Configurations, reproducibility metadata, safe experiment file management."""
import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for field in ("model", "dataset", "training", "results_dir"):
        if field not in config:
            raise ValueError(f"Config missing: {field}")
    return config


def slug(value: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def experiment_directory(cfg: dict, method: str, rank: int, seed: int):
    name = f"{slug(cfg['model']['id'].split('/')[-1])}_{cfg['dataset']['task']}_{method}_r{rank}_s{seed}"
    return Path(cfg["results_dir"]) / name


def dump_json(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def dependency_versions():
    names = ["torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes", "scikit-learn"]
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def now_utc():
    return datetime.now(timezone.utc).isoformat()
