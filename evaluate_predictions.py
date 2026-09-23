"""Merge completed forecasts with protected targets and compute task-level metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

HORIZON = 28


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _write_json_atomic(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def _task_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "task_id",
        "split_name",
        "fold_id",
        "destination",
        "market",
        "origin_date",
        "history_length",
        "model",
        "seed",
    ]
    rows: list[dict] = []
    for identity, group in scored.groupby(keys, sort=True, dropna=False):
        group = group.sort_values("horizon_day")
        if len(group) != HORIZON or group["horizon_day"].tolist() != list(range(1, HORIZON + 1)):
            raise ValueError(f"{identity}: expected one complete 28-day path")
        error = group["prediction"].to_numpy(float) - group["actual"].to_numpy(float)
        absolute = np.abs(error)
        squared = error**2
        mase_scale = group["mase7_scale"].iloc[0]
        rmsse_scale = group["rmsse7_scale"].iloc[0]
        high = group["actual"].to_numpy(float) > float(group["high_demand_threshold"].iloc[0])
        high_abs = absolute[high]
        high_under = np.maximum(0.0, -error[high])
        row = dict(zip(keys, identity))
        row.update(
            {
                "MAE": float(absolute.mean()),
                "RMSE": float(np.sqrt(squared.mean())),
                "WAPE": _safe_ratio(float(absolute.sum()), float(group["actual"].sum())),
                "MASE7": _safe_ratio(float(absolute.mean()), mase_scale),
                "RMSSE7": _safe_ratio(float(np.sqrt(squared.mean())), rmsse_scale),
                "bias": float(error.mean()),
                "high_demand_n": int(high.sum()),
                "high_demand_MASE7": (
                    _safe_ratio(float(high_abs.mean()), mase_scale) if high_abs.size else None
                ),
                "high_demand_mean_underforecast": (
                    float(high_under.mean()) if high_under.size else None
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(task_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE",
        "RMSE",
        "WAPE",
        "MASE7",
        "RMSSE7",
        "bias",
        "high_demand_MASE7",
        "high_demand_mean_underforecast",
    ]
    rows: list[dict] = []
    for market_mode in ("individual", "ALL"):
        group_columns = ["split_name", "destination", "model", "seed", "history_length"]
        source = task_metrics.copy()
        if market_mode == "individual":
            group_columns.insert(1, "market")
        else:
            source["market"] = "ALL"
            group_columns.insert(1, "market")
        for identity, group in source.groupby(group_columns, sort=True, dropna=False):
            row = dict(zip(group_columns, identity))
            row["task_count"] = len(group)
            row["market_count"] = (
                1
                if market_mode == "individual"
                else int(task_metrics.loc[group.index, "market"].nunique())
            )
            for metric in metric_columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean()) if values.notna().any() else None
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(predictions_path: Path, targets_path: Path, output: Path) -> dict:
    predictions_path = Path(predictions_path).resolve()
    targets_path = Path(targets_path).resolve()
    if predictions_path.name != "predictions.parquet":
        raise ValueError("predictions input must be named predictions.parquet")
    if targets_path.name != "targets.parquet":
        raise ValueError("targets input must be named targets.parquet")
    predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    targets = pd.read_parquet(targets_path, engine="pyarrow")
    for frame, label in ((predictions, "predictions"), (targets, "targets")):
        if not {"split_name", "fold_id"}.issubset(frame.columns):
            raise ValueError(f"{label} must identify its protocol split and fold")
    prediction_key = ["task_id", "model", "seed", "target_date", "horizon_day"]
    target_key = ["task_id", "target_date", "horizon_day"]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Duplicate prediction rows")
    if targets.duplicated(target_key).any():
        raise ValueError("Duplicate target rows")
    unknown = set(predictions["task_id"]) - set(targets["task_id"])
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} unknown task_id values")
    target_subset = targets[targets["task_id"].isin(predictions["task_id"].unique())]
    task_tags = predictions[["task_id", "split_name", "fold_id"]].drop_duplicates()
    target_tags = target_subset[["task_id", "split_name", "fold_id"]].drop_duplicates()
    if len(task_tags) != predictions["task_id"].nunique() or len(target_tags) != target_subset["task_id"].nunique():
        raise ValueError("A task has inconsistent split or fold labels")
    if set(map(tuple, task_tags.itertuples(index=False, name=None))) != set(map(tuple, target_tags.itertuples(index=False, name=None))):
        raise ValueError("Prediction and target split/fold labels disagree")
    scored = predictions.merge(
        target_subset[["task_id", "target_date", "horizon_day", "actual"]],
        on=["task_id", "target_date", "horizon_day"],
        how="left",
        validate="many_to_one",
    )
    if scored["actual"].isna().any():
        raise ValueError("At least one forecast could not be matched to a hidden target")
    scored["error"] = scored["prediction"] - scored["actual"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["high_demand"] = scored["actual"] > scored["high_demand_threshold"]
    tasks = _task_metrics(scored)
    aggregate = _aggregate(tasks)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "scored_predictions.parquet"
    tasks_path = output / "task_metrics.parquet"
    aggregate_path = output / "aggregate_metrics.parquet"
    _write_parquet_atomic(scored, scored_path)
    _write_parquet_atomic(tasks, tasks_path)
    _write_parquet_atomic(aggregate, aggregate_path)
    manifest = {
        "schema_version": 1,
        "prediction_rows": len(predictions),
        "task_model_paths": len(tasks),
        "predictions_sha256": _file_sha256(predictions_path),
        "targets_sha256": _file_sha256(targets_path),
        "aggregation": "28-day task metrics followed by split-specific equal-weight macro mean",
        "outputs": {
            scored_path.name: _file_sha256(scored_path),
            tasks_path.name: _file_sha256(tasks_path),
            aggregate_path.name: _file_sha256(aggregate_path),
        },
    }
    _write_json_atomic(manifest, output / "evaluation_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecasts after prediction completion")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.predictions, args.targets, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
