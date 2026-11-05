"""Run full benchmark sequentially: one process/GPU experiment at a time.

By default this is a DRY RUN. Pass --execute to actually train models.
"""
import argparse
import itertools
import subprocess
import sys

from .utils import experiment_directory, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/main.yaml")
    p.add_argument("--tasks", nargs="+", choices=["sst2", "mrpc", "qnli"], default=["sst2", "mrpc", "qnli"])
    p.add_argument("--methods", nargs="+", choices=["lora", "qlora"], default=["lora", "qlora"])
    p.add_argument("--ranks", nargs="+", type=int, choices=[4, 8, 16, 32], default=[4, 8, 16, 32])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--usd-per-hour", type=float, default=None)
    p.add_argument("--execute", action="store_true", help="Without this, only print the commands")
    p.add_argument("--retry-failed", action="store_true", help="Retry runs that were started but not completed")
    args = p.parse_args()
    cfg = load_config(args.config)
    jobs = list(itertools.product(args.tasks, args.methods, args.ranks, args.seeds))
    print(f"Experiment matrix: {len(jobs)} independent training jobs")
    errors = 0
    skipped = 0
    for i, (task, method, rank, seed) in enumerate(jobs, start=1):
        cfg["dataset"]["task"] = task
        folder = experiment_directory(cfg, method, rank, seed)
        metric_path = folder / "metrics.json"
        previous = {}
        if metric_path.exists():
            import json
            previous = json.loads(metric_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed":
                skipped += 1
                print(f"[{i}/{len(jobs)}] SKIP completed {folder}")
                continue
            if not args.retry_failed:
                print(f"[{i}/{len(jobs)}] SKIP existing non-completed {folder} (use --retry-failed)")
                skipped += 1
                continue
        cmd = [sys.executable, "-m", "peft_bench.run", "--config", args.config,
               "--task", task, "--method", method, "--rank", str(rank),
               "--seed", str(seed)]
        if previous:
            cmd.append("--force")
        if args.usd_per_hour is not None:
            cmd.extend(["--usd-per-hour", str(args.usd_per_hour)])
        print(f"[{i}/{len(jobs)}] {' '.join(cmd)}", flush=True)
        if args.execute:
            result = subprocess.run(cmd, check=False)
            if result.returncode:
                errors += 1
    if not args.execute:
        print("DRY RUN only. Add --execute to start training jobs sequentially.")
    print(f"Jobs={len(jobs)}, skipped={skipped}, errors={errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
