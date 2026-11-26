"""T2-A0 and T2-A1. Numeric rules match the frozen fallback policy."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CONFIG, HORIZON, MEMBERS, RESULTS, TRACK1, history_scales, load_predictions, write_json

FALLBACK = json.loads((CONFIG / "track2_fallback_policy.yaml").read_text(encoding="utf-8"))
FROZEN = json.loads((CONFIG / "track2_ensemble_config.yaml").read_text(encoding="utf-8"))
WEIGHTS = np.array([FROZEN["weights"][model] for model in MEMBERS], dtype=float)


def _member_frame(destination: str) -> pd.DataFrame:
    frame = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS, destination=destination)
    scales = history_scales("outer_backtest", "all", destination)
    return frame.merge(scales, on="task_id", how="left")


def _seasonal(destination: str) -> pd.DataFrame:
    root = TRACK1 / "outer_server_all" / "predictions" / "seasonal_naive" / "SN01" / "42"
    frame = pd.concat((pd.read_parquet(path) for path in root.glob("*.parquet")), ignore_index=True)
    return frame[frame["destination"] == destination]


def _ok(part: pd.DataFrame) -> tuple[bool, str]:
    part = part.sort_values("horizon_day")
    if len(part) != HORIZON or not np.isfinite(part["prediction"].to_numpy(float)).all():
        return False, "structure"
    if part["horizon_day"].tolist() != list(range(1, HORIZON + 1)):
        return False, "structure"
    recent = float(part["recent28_mean"].iloc[0])
    if recent < FALLBACK["recent28_skip_below"]:
        return True, "level_skipped"
    ratio = float(part["prediction"].mean() / recent)
    if not (FALLBACK["r_low"] <= ratio <= FALLBACK["r_high"]):
        return False, "level"
    return True, "ok"


def run_policy(members: pd.DataFrame, seasonal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    logs = []
    for task_id, group in members.groupby("task_id", sort=False):
        usable = []
        reasons = {}
        for model in MEMBERS:
            part = group[group["model"] == model]
            passed, reason = _ok(part)
            reasons[model] = reason
            if passed:
                usable.append(model)
        path = "frozen_ensemble"
        if usable == MEMBERS:
            acc = np.zeros(HORIZON)
            total = 0.0
            for model in MEMBERS:
                acc += WEIGHTS[MEMBERS.index(model)] * group[group["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float)
                total += WEIGHTS[MEMBERS.index(model)]
            prediction = np.maximum(0.0, acc)
            if total <= 0:
                path = "equal_remaining"
                prediction = np.maximum(0.0, np.mean([
                    group[group["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float) for model in usable
                ], axis=0))
        elif usable:
            prior = np.array([WEIGHTS[MEMBERS.index(model)] for model in usable])
            if prior.sum() <= 0:
                path = "equal_remaining"
                weights = np.full(len(usable), 1.0 / len(usable))
            else:
                path = "renormalize_remaining"
                weights = prior / prior.sum()
            acc = np.zeros(HORIZON)
            for weight, model in zip(weights, usable):
                acc += weight * group[group["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float)
            prediction = np.maximum(0.0, acc)
        else:
            path = "seasonal_naive"
            naive = seasonal[seasonal["task_id"] == task_id].sort_values("horizon_day")
            prediction = naive["prediction"].to_numpy(float)
        template = group[group["model"] == MEMBERS[0]].sort_values("horizon_day").copy()
        template["prediction_raw"] = prediction
        template["prediction"] = prediction
        template["model"] = "T2-A0"
        template["candidate_id"] = "T2-A0"
        outputs.append(template)
        logs.append({"task_id": task_id, "path": path, "reasons": reasons})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(logs)


def main() -> None:
    out = RESULTS / "agents"
    out.mkdir(parents=True, exist_ok=True)
    frame = _member_frame("Macau")
    seasonal = _seasonal("Macau")
    predictions, logs = run_policy(frame, seasonal)
    for column in ("origin_date", "target_date"):
        predictions[column] = pd.to_datetime(predictions[column])
    a0 = out / "T2-A0"
    a1 = out / "T2-A1"
    for directory, label in ((a0, "T2-A0"), (a1, "T2-A1")):
        directory.mkdir(parents=True, exist_ok=True)
        saved = predictions.copy()
        saved["model"] = label
        saved["candidate_id"] = label
        saved.to_parquet(directory / "predictions.parquet", index=False)
    logs.to_json(out / "fallbacks.jsonl", orient="records", lines=True)
    reference = pd.read_parquet(RESULTS / "outer_macau" / "T2-05" / "predictions.parquet")
    merged = predictions.merge(reference[["task_id", "horizon_day", "prediction"]], on=["task_id", "horizon_day"], suffixes=("_a0", "_ref"))
    merged = merged.merge(logs[["task_id", "path"]], on="task_id", how="left")
    clean = merged[merged["path"] == "frozen_ensemble"]
    gap = float(np.max(np.abs(clean["prediction_a0"] - clean["prediction_ref"])))
    write_json(out / "numeric_identity.json", {
        "max_abs_gap_A0_vs_T2-05_on_clean_tasks": gap,
        "clean_tasks": int(clean["task_id"].nunique()),
        "A1_equals_A0": True,
        "fallback_counts": logs["path"].value_counts().to_dict(),
        "actuals_visible_to_coordinator": False,
        "note": "Quality-check failures are supposed to leave the frozen point forecast. Only tasks that stay on frozen_ensemble are required to match T2-05.",
    })
    print(json.dumps({"clean_gap": gap, "paths": logs["path"].value_counts().to_dict()}, ensure_ascii=False))
    if gap != 0:
        raise SystemExit("A0 does not match the frozen ensemble on clean outer tasks")


if __name__ == "__main__":
    main()
