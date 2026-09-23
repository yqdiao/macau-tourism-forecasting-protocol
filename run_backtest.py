"""Generate forecasts from task_inputs.parquet without access to targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from data_pipeline import ForecastInputTask, HORIZON, read_task_inputs
from models import MODEL_NAMES, forecast


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scales(history: np.ndarray, period: int = 7) -> tuple[float | None, float | None]:
    difference = np.asarray(history[period:] - history[:-period], dtype=float)
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    return (mae or None, rmse or None)


def run_one(
    task: ForecastInputTask, model: str, seed: int = 42
) -> tuple[list[dict], dict]:
    """Forecast one task. No observed target values are accepted or returned."""
    start = time.perf_counter()
    result = forecast(model, task.history, HORIZON, seed)
    total = time.perf_counter() - start
    if len(result.prediction) != HORIZON or len(result.raw_prediction) != HORIZON:
        raise ValueError(f"{task.task_id}: {model} did not return 28 values")
    mase_scale, rmsse_scale = _scales(task.history)
    high_demand_threshold = float(np.quantile(task.history, 0.9))
    effective_context = int(result.detail.get("effective_context_days", len(task.history)))
    rows = []
    for horizon_day, (target_day, raw, clipped) in enumerate(
        zip(task.target_dates, result.raw_prediction, result.prediction), 1
    ):
        rows.append(
            {
                "task_id": task.task_id,
                "split_name": task.split_name,
                "fold_id": task.fold_id,
                "destination": task.destination,
                "market": task.market,
                "origin_date": task.origin,
                "history_length": task.history_length_label,
                "history_start": task.history_start,
                "training_history_days": len(task.history),
                "effective_context_days": effective_context,
                "target_date": target_day,
                "horizon_day": horizon_day,
                "model": model,
                "seed": seed,
                "prediction_raw": float(raw),
                "prediction": float(clipped),
                "was_clipped": bool(raw < 0),
                "mase7_scale": mase_scale,
                "rmsse7_scale": rmsse_scale,
                "high_demand_threshold": high_demand_threshold,
            }
        )
    timing = {
        "task_id": task.task_id,
        "split_name": task.split_name,
        "fold_id": task.fold_id,
        "model": model,
        "seed": seed,
        "fit_seconds": result.fit_seconds,
        "inference_seconds": result.inference_seconds,
        "total_seconds": total,
        "other_seconds": max(
            0.0, total - result.fit_seconds - result.inference_seconds
        ),
        "model_detail": result.detail,
    }
    return rows, timing


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _write_json_atomic(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(
    *,
    inputs: Path,
    model_names: tuple[str, ...],
    output: Path,
    destinations: tuple[str, ...] = (),
    markets: tuple[str, ...] = (),
    history_lengths: tuple[str, ...] = (),
    splits: tuple[str, ...] = (),
    origin_date: date | None = None,
    max_tasks: int | None = None,
    seed: int = 42,
) -> dict:
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"models must be selected from: {', '.join(MODEL_NAMES)}")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    inputs = Path(inputs).resolve()
    if inputs.name != "task_inputs.parquet" or not inputs.is_file():
        raise ValueError("--inputs must point to an existing task_inputs.parquet")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task in read_task_inputs(inputs):
        if destinations and task.destination not in destinations:
            continue
        if markets and task.market not in markets and f"{task.destination}:{task.market}" not in markets:
            continue
        if history_lengths and task.history_length_label not in history_lengths:
            continue
        if splits and task.split_name not in splits:
            continue
        if origin_date and task.origin != origin_date:
            continue
        tasks.append(task)
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    if not tasks:
        raise ValueError("No task_inputs rows matched the requested filters")

    prediction_rows: list[dict] = []
    timings_path = output / "timings.jsonl"
    timings_temporary = output / ".timings.jsonl.tmp"
    with timings_temporary.open("w", encoding="utf-8") as timings:
        for task_number, task in enumerate(tasks, 1):
            for name in model_names:
                rows, timing = run_one(task, name, seed)
                prediction_rows.extend(rows)
                timings.write(json.dumps(timing, ensure_ascii=False) + "\n")
            if task_number % 10 == 0:
                print(f"Completed {task_number} input tasks", flush=True)
    os.replace(timings_temporary, timings_path)

    predictions = pd.DataFrame(prediction_rows)
    expected = len(tasks) * len(model_names) * HORIZON
    if len(predictions) != expected:
        raise ValueError(f"Expected {expected} prediction rows, got {len(predictions)}")
    counts = predictions.groupby(["task_id", "model", "seed"]).size()
    if not counts.eq(HORIZON).all():
        raise ValueError("Every task/model/seed must contain exactly 28 forecasts")
    predictions_path = output / "predictions.parquet"
    _write_parquet_atomic(predictions, predictions_path)
    manifest = {
        "schema_version": 1,
        "models": list(model_names),
        "destinations": list(destinations),
        "markets": list(markets),
        "history_lengths": list(history_lengths),
        "splits": list(splits),
        "origin_date": origin_date.isoformat() if origin_date else None,
        "seed": seed,
        "forecast_tasks": len(tasks),
        "prediction_rows": len(predictions),
        "max_tasks": max_tasks,
        "input_artifact": {
            "name": inputs.name,
            "sha256": _file_sha256(inputs),
        },
        "target_artifact_accessed": False,
        "outputs": {
            "predictions.parquet": _file_sha256(predictions_path),
            "timings.jsonl": _file_sha256(timings_path),
        },
    }
    _write_json_atomic(manifest, output / "run_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 28-day forecasts from model-only task inputs"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "task_inputs.parquet",
    )
    parser.add_argument(
        "--models",
        default="seasonal_naive,ets,sarima,random_forest,lightgbm,transformer",
    )
    parser.add_argument("--destinations", default="")
    parser.add_argument("--markets", default="")
    parser.add_argument("--history", default="", help="Comma-separated 180,365,all filter")
    parser.add_argument("--splits", default="", help="Comma-separated pandemic_stress,recovery_tuning,outer_backtest")
    parser.add_argument("--origin-date", type=date.fromisoformat)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "run",
    )
    args = parser.parse_args()
    manifest = run(
        inputs=args.inputs,
        model_names=tuple(x.strip() for x in args.models.split(",") if x.strip()),
        destinations=tuple(x.strip() for x in args.destinations.split(",") if x.strip()),
        markets=tuple(x.strip() for x in args.markets.split(",") if x.strip()),
        history_lengths=tuple(x.strip() for x in args.history.split(",") if x.strip()),
        splits=tuple(x.strip() for x in args.splits.split(",") if x.strip()),
        output=args.output,
        origin_date=args.origin_date,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
