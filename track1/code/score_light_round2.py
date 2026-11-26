"""Score round-2 per-market LightGBM on B and outer. B selects; outer is exploratory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "track1/results/light_models"
TARGETS = REPO / "shared/data/targets.parquet"
MAIN = REPO / "track1/evaluation/outer_main_table/task_metrics.parquet"
OUTPUT = REPO / "track1/evaluation/light_models_round2"


def _macro(frame: pd.DataFrame, column: str = "MASE7") -> float:
    return float(frame.groupby("market")[column].mean().mean())


def main() -> None:
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length"])
    targets = targets.loc[targets["history_length"] == "all"]
    rows = []
    scored = []
    for split, folds in (("recovery_tuning", 12), ("outer_backtest", 33)):
        for candidate in ("LL01", "LL02", "LL03"):
            paths = list((ROOT / split / "local_lightgbm" / candidate).glob("seed42/*/predictions.parquet"))
            if len(paths) != folds:
                rows.append({"split": split, "candidate_id": candidate, "status": f"incomplete {len(paths)}/{folds}"})
                continue
            frames = [pd.read_parquet(p, columns=["task_id", "market", "horizon_day", "prediction", "mase7_scale"]) for p in paths]
            pred = pd.concat(frames, ignore_index=True)
            merged = pred.merge(targets.loc[targets["split_name"] == split], on=["task_id", "horizon_day"])
            merged["abs_error"] = (merged["prediction"] - merged["actual"]).abs()
            recs = []
            for (task_id, market), g in merged.groupby(["task_id", "market"]):
                mae = float(g["abs_error"].mean())
                scale = float(g["mase7_scale"].iloc[0])
                recs.append({"task_id": task_id, "market": market, "MAE": mae,
                             "MASE7": mae / scale if scale > 0 else np.nan, "candidate_id": candidate, "split": split})
            part = pd.DataFrame(recs)
            scored.append(part)
            rows.append({"split": split, "candidate_id": candidate, "status": "scored",
                         "tasks": int(part["task_id"].nunique()), "Macau_MASE7": _macro(part)})
    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "summary.csv", index=False)
    b = table.loc[(table["split"] == "recovery_tuning") & (table["status"] == "scored")]
    if len(b) == 3:
        winner = b.sort_values(["Macau_MASE7", "candidate_id"]).iloc[0]
        (OUTPUT / "b_winner.json").write_text(json.dumps({"candidate_id": winner["candidate_id"],
                                                          "Macau_MASE7": float(winner["Macau_MASE7"])}, indent=2) + "\n")
    if scored:
        outer = pd.concat(scored)
        outer = outer.loc[outer["split"] == "outer_backtest"]
        published = pd.read_parquet(MAIN, columns=["model", "task_id", "market", "MASE7"])
        ref = published.loc[published["model"] == "chronos_2", ["task_id", "MASE7"]]
        gaps = []
        for candidate, part in outer.groupby("candidate_id"):
            paired = part.merge(ref, on="task_id", suffixes=("_light", "_ref"))
            gaps.append({"candidate_id": candidate,
                         "minus_chronos2": _macro(paired.rename(columns={"MASE7_light": "MASE7"}))
                         - float(paired.groupby("market")["MASE7_ref"].mean().mean())})
        pd.DataFrame(gaps).to_csv(OUTPUT / "gap_vs_chronos2.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
