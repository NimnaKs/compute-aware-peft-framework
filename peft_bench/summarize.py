"""Convert run JSON results into experiment CSV tables and summary aggregation."""
import argparse
import json
from pathlib import Path


def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    files = sorted(Path(args.runs).glob("*/metrics.json"))
    if not files:
        raise SystemExit(f"No run metrics found in {args.runs}/*/metrics.json")
    rows = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        data.pop("versions", None)
        data.pop("data", None)
        data["run_dir"] = str(f.parent)
        rows.append(data)
    frame = pd.DataFrame(rows)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "all_runs.csv", index=False)
    completed = frame.loc[frame.status == "completed"].copy()
    if completed.empty:
        print(f"No completed runs yet. Wrote {dest/'all_runs.csv'}")
        return
    measures = [name for name in (
        "final_accuracy", "final_macro_f1", "final_mrpc_f1", "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib", "training_wall_time_s",
        "estimated_end_to_end_tokens_per_sec", "trainable_parameters",
        "adapter_checkpoint_mib", "gpu_training_cost_usd",
    ) if name in completed.columns]
    group = completed.groupby(["model", "task", "method", "rank"], dropna=False)
    report = group[measures].agg(["mean", "std", "count"]).reset_index()
    report.columns = ["_".join(filter(None, col)).rstrip("_") if isinstance(col, tuple) else col
                      for col in report.columns]
    report.to_csv(dest / "summary_by_configuration.csv", index=False)
    print(f"Wrote {dest/'all_runs.csv'} and {dest/'summary_by_configuration.csv'}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
