"""Fill registered slices that were scored in Track 1 but not copied into Track 2.

Does not refit weights. Reuses sealed baseline forecasts and the sealed fault schedule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import MEMBERS, RESULTS, TRACK1, history_scales, load_targets, write_json
from stage_apply import _write, _write_c2
from stage_crossfit import _interval_score, _pinball
from stage_faults import _deliver, _inject, _patterns

BASELINE_MEMBERS = (("ets", "ETS03"), ("ridge", "RID02"), ("elasticnet", "EN03"))


def _read_member(folder: Path, destination: str) -> pd.DataFrame:
    frame = pd.concat((pd.read_parquet(path) for path in folder.glob("*.parquet")), ignore_index=True)
    return frame[frame["destination"] == destination]


def _baseline_equal(root: Path, destination: str) -> pd.DataFrame:
    frames = []
    for model, candidate in BASELINE_MEMBERS:
        part = _read_member(root / model / candidate / "42", destination)
        frames.append(part[["task_id", "horizon_day", "prediction"]].rename(columns={"prediction": model}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["task_id", "horizon_day"], how="inner")
    merged["prediction"] = merged[[model for model, _ in BASELINE_MEMBERS]].mean(axis=1).clip(lower=0)
    merged["prediction_raw"] = merged["prediction"]
    meta = _read_member(root / "ets" / "ETS03" / "42", destination)
    keep = ["task_id", "fold_id", "destination", "market", "origin_date", "target_date", "horizon_day", "history_length", "mase7_scale", "split_name"]
    return meta[keep].merge(merged[["task_id", "horizon_day", "prediction", "prediction_raw"]], on=["task_id", "horizon_day"])


def _seasonal(root: Path, destination: str) -> pd.DataFrame:
    return _read_member(root / "seasonal_naive" / "SN01" / "42", destination)


def _c2_from_m1(m1_path: Path, seasonal: pd.DataFrame, recent_map: dict, directory: Path) -> None:
    m1 = pd.read_parquet(m1_path)
    _write_c2(m1, seasonal, directory, recent_map)


def build_missing() -> None:
    outer_root = TRACK1 / "outer_server_all" / "predictions"
    hk = _baseline_equal(outer_root, "HongKong")
    _write(hk, "T2-C1", RESULTS / "extension_hong_kong" / "T2-C1")
    slices = {
        "180": (TRACK1 / "sensitivity" / "outer_backtest" / "180" / "baselines" / "predictions", RESULTS / "sensitivity_180", "outer_backtest", "180"),
        "365": (TRACK1 / "sensitivity" / "outer_backtest" / "365" / "baselines" / "predictions", RESULTS / "sensitivity_365", "outer_backtest", "365"),
        "pandemic": (TRACK1 / "sensitivity" / "pandemic_stress" / "all" / "baselines" / "predictions", RESULTS / "pandemic_stress", "pandemic_stress", "all"),
    }
    for label, (root, out_root, split, history) in slices.items():
        c1 = _baseline_equal(root, "Macau")
        _write(c1, "T2-C1", out_root / "T2-C1")
        recent = history_scales(split, history, "Macau")
        recent_map = dict(zip(recent["task_id"], recent["recent28_mean"]))
        _c2_from_m1(out_root / "T2-01" / "predictions.parquet", _seasonal(root, "Macau"), recent_map, out_root / "T2-C2")
        print(label, "C1", c1["task_id"].nunique(), "written")


def outer_probability() -> dict:
    pred = pd.read_parquet(RESULTS / "outer_macau" / "T2-06" / "predictions.parquet")
    targets = load_targets("outer_backtest", "all", "Macau")
    merged = pred.merge(targets[["task_id", "horizon_day", "actual"]], on=["task_id", "horizon_day"])
    actual = merged["actual"].to_numpy(float)
    return {
        "pinball_q10": _pinball(actual, merged["q0.1"].to_numpy(float), 0.1),
        "pinball_q50": _pinball(actual, merged["q0.5"].to_numpy(float), 0.5),
        "pinball_q90": _pinball(actual, merged["q0.9"].to_numpy(float), 0.9),
        "wis80": _interval_score(actual, merged["q0.1"].to_numpy(float), merged["q0.9"].to_numpy(float)),
    }


def fault_by_type() -> pd.DataFrame:
    schedule = json.loads((RESULTS / "fault_injection" / "schedule.json").read_text(encoding="utf-8"))
    from common import load_predictions
    frame = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS, destination="Macau")
    seasonal = _seasonal(TRACK1 / "outer_server_all" / "predictions", "Macau")
    recent_map = history_scales("outer_backtest", "all", "Macau").set_index("task_id")["recent28_mean"].to_dict()
    actuals = load_targets("outer_backtest", "all", "Macau")
    scale_map = frame.groupby("task_id")["mase7_scale"].first().to_dict()
    patterns = _patterns()
    by_task = {task_id: part for task_id, part in frame.groupby("task_id")}
    naive = {task_id: part.sort_values("horizon_day")["prediction"].to_numpy(float) for task_id, part in seasonal.groupby("task_id")}
    actual_by = {task_id: part.sort_values("horizon_day")["actual"].to_numpy(float) for task_id, part in actuals.groupby("task_id")}
    faulted = {(row["rate"], row["task_id"]): row for row in schedule}
    rows = []
    for rate in (0.01, 0.05, 0.10):
        for system in ("T2-A0p", "T2-C2", "T2-A1"):
            bucket: dict[str, list[float]] = {}
            undelivered: dict[str, int] = {}
            for task_id, parts in by_task.items():
                event = faulted.get((rate, task_id))
                if event is None:
                    continue
                vectors = {model: parts[parts["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float) for model in MEMBERS}
                member = event["member"] if system == "T2-A1" else MEMBERS[0]
                vectors, _ = _inject(vectors, event["fault"], event["market"], patterns, member)
                prediction, _detected, ok = _deliver(vectors, float(recent_map[task_id]), naive[task_id], system)
                fault = event["fault"]
                if not ok or prediction is None:
                    undelivered[fault] = undelivered.get(fault, 0) + 1
                    continue
                scale = float(scale_map[task_id])
                mase = float(np.mean(np.abs(prediction - actual_by[task_id])) / scale)
                bucket.setdefault(fault, []).append(mase)
            faults = sorted(set(bucket) | set(undelivered))
            for fault in faults:
                values = bucket.get(fault, [])
                rows.append({"rate": rate, "system": system, "fault": fault, "n": len(values), "undelivered": undelivered.get(fault, 0), "task_mase": float(np.mean(values)) if values else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "fault_injection" / "fault_by_type.csv", index=False)
    return table


def main() -> None:
    build_missing()
    probability = outer_probability()
    write_json(RESULTS / "evaluation" / "outer_t2_06_probability.json", probability)
    print(json.dumps(probability, indent=2))
    print(fault_by_type().to_string(index=False))


if __name__ == "__main__":
    main()
