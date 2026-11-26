"""Score the added light-model slices: LL01 on A01-A13, and NL01/GL03/LL01 on 180/365.

Does not retune. Seed means are taken before the market-equal MASE(7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "track1/results/light_models"
TARGETS = REPO / "shared/data/targets.parquet"
OUTPUT = REPO / "track1/evaluation/light_models_slices"

SLICES = [
    ("pandemic_stress", "all", "local_lightgbm", "LL01", 13, 1),
    ("outer_backtest", "180", "global_linear", "NL01", 33, 3),
    ("outer_backtest", "365", "global_linear", "NL01", 33, 3),
    ("outer_backtest", "180", "global_lightgbm", "GL03", 33, 1),
    ("outer_backtest", "365", "global_lightgbm", "GL03", 33, 1),
    ("outer_backtest", "180", "local_lightgbm", "LL01", 33, 1),
    ("outer_backtest", "365", "local_lightgbm", "LL01", 33, 1),
]


def _macro(frame: pd.DataFrame) -> float:
    return float(frame.groupby("market")["MASE7"].mean().mean())


def _paths(split: str, history: str, model: str, candidate: str) -> list[Path]:
    base = ROOT / split
    if history != "all":
        base = base / history
    return list((base / model / candidate).glob("seed*/*/predictions.parquet"))


def _score(paths: list[Path], targets: pd.DataFrame) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=["task_id", "market", "fold_id", "seed", "horizon_day",
                                             "prediction", "mase7_scale", "history_length"])
              for path in paths]
    pred = pd.concat(frames, ignore_index=True)
    merged = pred.merge(targets, on=["task_id", "horizon_day", "history_length"])
    merged["abs_error"] = (merged["prediction"] - merged["actual"]).abs()
    rows = []
    for key, group in merged.groupby(["seed", "task_id", "market", "fold_id"], sort=False):
        scale = float(group["mase7_scale"].iloc[0])
        mae = float(group["abs_error"].mean())
        rows.append({"seed": key[0], "task_id": key[1], "market": key[2], "fold_id": key[3],
                     "MAE": mae, "MASE7": mae / scale if scale > 0 else np.nan})
    per_seed = pd.DataFrame(rows)
    return (per_seed.drop(columns="seed")
            .groupby(["task_id", "market", "fold_id"], as_index=False)
            .mean(numeric_only=True))


def main() -> None:
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name",
                                                "history_length", "destination"])
    targets = targets.loc[targets["destination"] == "Macau"]
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, history, model, candidate, folds, seeds in SLICES:
        paths = _paths(split, history, model, candidate)
        fold_count = len({path.parent.name for path in paths})
        seed_count = len({path.parents[1].name for path in paths})
        if fold_count != folds or seed_count != seeds:
            rows.append({"split": split, "history": history, "model": model, "candidate_id": candidate,
                         "status": f"incomplete folds {fold_count}/{folds} seeds {seed_count}/{seeds}"})
            continue
        part = targets[(targets["split_name"] == split) & (targets["history_length"] == history)]
        scored = _score(paths, part)
        rows.append({"split": split, "history": history, "model": model, "candidate_id": candidate,
                     "status": "scored", "tasks": int(scored["task_id"].nunique()),
                     "Macau_MASE7": _macro(scored),
                     "Macau_MAE": float(scored.groupby("market")["MAE"].mean().mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
