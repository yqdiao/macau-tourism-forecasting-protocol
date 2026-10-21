"""Score frozen lightweight runs on outer and A, plus the two GL ablations.

Reads targets only after the prediction manifests exist. O01-O33 remains exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "track1/results/light_models"
TARGETS = REPO / "shared/data/targets.parquet"
MAIN = REPO / "track1/evaluation/outer_main_table/task_metrics.parquet"
OUTPUT = REPO / "track1/evaluation/light_models_extra"
MODELS = [
    ("global_lightgbm", "GL03"),
    ("global_linear", "NL01"),
    ("global_lightgbm_nocal", "GL03"),
    ("global_lightgbm_local", "GL03"),
]
SPLITS = {"outer_backtest": 33, "pandemic_stress": 13, "recovery_tuning": 12}
REFERENCES = ["chronos_2", "granite_patchtst_fm", "ets", "seasonal_naive"]


def _macro(frame: pd.DataFrame) -> float:
    return float(frame.groupby("market")["MASE7"].mean().mean())


def _score_split(predictions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(targets, on=["task_id", "horizon_day"])
    merged["abs_error"] = (merged["prediction"] - merged["actual"]).abs()
    rows = []
    for key, group in merged.groupby(["model", "candidate_id", "seed", "task_id", "market", "fold_id"], sort=False):
        scale = float(group["mase7_scale"].iloc[0])
        mae = float(group["abs_error"].mean())
        rows.append({"model": key[0], "candidate_id": key[1], "seed": key[2], "task_id": key[3],
                     "market": key[4], "fold_id": key[5], "MAE": mae,
                     "MASE7": mae / scale if scale > 0 else np.nan})
    per_seed = pd.DataFrame(rows)
    return (per_seed.drop(columns="seed")
            .groupby(["model", "candidate_id", "task_id", "market", "fold_id"], as_index=False)
            .mean(numeric_only=True))


def main() -> None:
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length"])
    targets = targets.loc[targets["history_length"] == "all"]
    summaries = []
    task_frames = []
    for split, folds in SPLITS.items():
        frames = []
        for model, candidate in MODELS:
            paths = list((ROOT / split / model / candidate).glob("seed*/*/predictions.parquet"))
            fold_count = len({path.parent.name for path in paths})
            if not paths or fold_count != folds:
                summaries.append({"split": split, "model": model, "candidate_id": candidate,
                                  "status": f"incomplete {len(paths)}/{folds}"})
                continue
            frames.extend(pd.read_parquet(path, columns=["task_id", "market", "fold_id", "model",
                                                         "candidate_id", "seed", "horizon_day", "prediction",
                                                         "mase7_scale"]) for path in paths)
        if not frames:
            continue
        scored = _score_split(pd.concat(frames, ignore_index=True), targets.loc[targets["split_name"] == split])
        scored["split"] = split
        task_frames.append(scored)
        for (model, candidate), group in scored.groupby(["model", "candidate_id"]):
            summaries.append({"split": split, "model": model, "candidate_id": candidate, "status": "scored",
                              "tasks": int(group["task_id"].nunique()), "Macau_MASE7": _macro(group),
                              "Macau_MAE": float(group.groupby("market")["MAE"].mean().mean())})
    table = pd.DataFrame(summaries)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if task_frames:
        tasks = pd.concat(task_frames, ignore_index=True)
        outer = tasks.loc[tasks["split"] == "outer_backtest"]
        published = pd.read_parquet(MAIN, columns=["model", "task_id", "market", "MASE7"])
        published = published.loc[published["model"].isin(REFERENCES)]
        rows = []
        champion = outer.loc[outer["model"] == "global_linear", ["task_id", "market", "MASE7"]]
        for model, part in published.groupby("model"):
            paired = champion.merge(part[["task_id", "MASE7"]], on="task_id", suffixes=("_light", "_ref"))
            rows.append({"reference": model, "tasks": int(len(paired)),
                         "light_minus_reference_MASE7": _macro(paired.rename(columns={"MASE7_light": "MASE7"}))
                         - float(paired.groupby("market")["MASE7_ref"].mean().mean())})
        pd.DataFrame(rows).to_csv(OUTPUT / "champion_vs_registered.csv", index=False)
        tasks.to_parquet(OUTPUT / "task_metrics.parquet", index=False)
    table.to_csv(OUTPUT / "summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
