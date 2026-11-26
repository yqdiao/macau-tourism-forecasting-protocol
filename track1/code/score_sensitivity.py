"""Score the frozen A-group run and the 180/365 history runs. Does not retune."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "track1/results/sensitivity"
TARGETS = REPO / "shared/data/targets.parquet"
OUTPUT = REPO / "track1/evaluation/sensitivity"
GPU_HOUR = 0.53
CPU_HOUR = 0.04
THREADS = {"random_forest": 2, "parrot_random_forest_800": 2, "lightgbm": 2, "transformer": 2}
SLICES = [("pandemic_stress", "all"), ("outer_backtest", "180"), ("outer_backtest", "365")]


def _macro(frame: pd.DataFrame) -> float:
    macau = frame.loc[frame["destination"] == "Macau"]
    if macau.empty:
        return float("nan")
    return float(macau.groupby("market")["MASE7"].mean().mean())


def _score_frame(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(targets, on=["task_id", "horizon_day"], suffixes=("", "_target"))
    merged["abs_error"] = (merged["prediction"] - merged["actual"]).abs()
    rows = []
    for key, group in merged.groupby(["model", "seed", "task_id", "destination", "market", "fold_id"], sort=False):
        scale = float(group["mase7_scale"].iloc[0])
        mae = float(group["abs_error"].mean())
        rows.append({"model": key[0], "seed": key[1], "task_id": key[2], "destination": key[3],
                     "market": key[4], "fold_id": key[5], "MAE": mae,
                     "MASE7": mae / scale if scale > 0 else np.nan})
    per_seed = pd.DataFrame(rows)
    return (per_seed.drop(columns="seed")
            .groupby(["model", "task_id", "destination", "market", "fold_id"], as_index=False)
            .mean(numeric_only=True))


def _foundation_cost(split: str, history: str, model: str) -> float | None:
    files = list((ROOT / split / history / model).glob("seed*/A*_*/timings.jsonl"))
    files += list((ROOT / split / history / model).glob("seed*/O*_*/timings.jsonl"))
    if not files:
        return None
    seconds = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            continue
        load = float(rows[0].get("load_seconds_shared") or 0)
        infer = sum(float(row.get("inference_seconds") or 0) for row in rows)
        seconds.append((load + infer) / len(rows))
    return float(np.mean(seconds)) if seconds else None


def _baseline_cost(split: str, history: str, model: str) -> float | None:
    files = list((ROOT / split / history / "baselines" / "predictions" / model).glob("*/*/*.parquet"))
    if not files:
        return None
    totals = []
    for path in files:
        frame = pd.read_parquet(path, columns=["total_seconds"])
        totals.append(float(frame["total_seconds"].iloc[0]))
    threads = THREADS.get(model, 1)
    return float(np.mean(totals) * threads)


def main() -> None:
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split, history in SLICES:
        part = targets[(targets["split_name"] == split) & (targets["history_length"] == history)]
        frames = []
        base = ROOT / split / history
        for path in base.glob("*/seed*/*_*/predictions.parquet"):
            frames.append(pd.read_parquet(path, columns=["task_id", "destination", "market", "fold_id",
                                                         "model", "seed", "horizon_day", "prediction", "mase7_scale"]))
        pred_root = base / "baselines" / "predictions"
        if pred_root.is_dir():
            for path in pred_root.glob("*/*/*/*.parquet"):
                frames.append(pd.read_parquet(path, columns=["task_id", "destination", "market", "fold_id",
                                                             "model", "seed", "horizon_day", "prediction", "mase7_scale"]))
        if not frames:
            summaries.append({"split": split, "history": history, "model": None, "status": "no_predictions"})
            continue
        scored = _score_frame(pd.concat(frames, ignore_index=True), part)
        for model, group in scored.groupby("model"):
            kind = "gpu" if (base / model).is_dir() else "cpu"
            seconds = _foundation_cost(split, history, model) if kind == "gpu" else _baseline_cost(split, history, model)
            price = GPU_HOUR if kind == "gpu" else CPU_HOUR
            summaries.append({
                "split": split, "history": history, "model": model, "status": "scored",
                "macau_tasks": int((group["destination"] == "Macau").sum()),
                "Macau_MASE7": _macro(group),
                "Macau_MAE": float(group.loc[group["destination"] == "Macau"].groupby("market")["MAE"].mean().mean()),
                "seconds_per_task": seconds,
                "usd_per_task": None if seconds is None else seconds * price / 3600,
            })
        inapplicable = list(base.glob("*/seed*/inapplicable.json")) + list(base.glob("*/*/seed*/inapplicable.json"))
        # inapplicable.json sits beside the fold directory's parent seed path: model/seed/fold/inapplicable.json
        for path in base.glob("*/*/*/inapplicable.json"):
            body = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({"split": split, "history": history, "model": body["model"],
                              "status": "inapplicable", "reason": body["reason"],
                              "macau_tasks": 0, "Macau_MASE7": None, "Macau_MAE": None,
                              "seconds_per_task": None, "usd_per_task": None})
    table = pd.DataFrame(summaries).drop_duplicates(["split", "history", "model", "status"])
    table.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "status.json").write_text(json.dumps({
        "slices": [{"split": s, "history": h} for s, h in SLICES],
        "prices": {"gpu_usd_per_hour": GPU_HOUR, "cpu_usd_per_vcpu_hour": CPU_HOUR, "price_date": "2026-10-05"},
        "note": "Same frozen candidates as the main path. PatchTST and Informer are inapplicable at history 180.",
    }, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
