"""Run registered B01-B12 candidate forecasts without mounting target values.

Each 28-day path is written atomically. Re-running skips complete paths, so a
long CPU run can resume after interruption. This process never opens targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from baseline_config import MODEL_NAMES, SEARCH_PATH, validate
from data_pipeline import read_task_inputs
from models import forecast
from protocol_config import DEFAULT_PROTOCOL

RUN_ORDER = (
    "seasonal_naive", "ets", "sarima", "ridge", "linear_regression",
    "svr", "lasso", "elasticnet", "lightgbm", "transformer",
    "random_forest", "gradient_boosting", "parrot_random_forest_800",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run_job(job: tuple) -> dict:
    output, model, candidate_id, params, seed, task, retry = job
    output = Path(output)
    relative = Path(model) / candidate_id / str(seed) / f"{task.task_id}.parquet"
    prediction_path = output / "predictions" / relative
    failure_path = output / "failures" / relative.with_suffix(".json")
    if prediction_path.is_file():
        frame = pd.read_parquet(prediction_path, columns=["horizon_day"])
        if frame["horizon_day"].tolist() == list(range(1, 29)):
            return {"status": "skipped", "model": model, "candidate_id": candidate_id}
        raise ValueError(f"Incomplete existing prediction shard: {prediction_path}")
    previous_attempts = 0
    if failure_path.is_file():
        previous_attempts = json.loads(failure_path.read_text(encoding="utf-8"))["attempts"]
        if not retry or previous_attempts >= 2:
            return {"status": "failed_existing", "model": model, "candidate_id": candidate_id}
    start = time.perf_counter()
    try:
        result = forecast(model, task.history, 28, seed, config=params)
        if len(result.prediction) != 28 or len(result.raw_prediction) != 28:
            raise ValueError("Model did not return exactly 28 values")
        if not np.isfinite(result.raw_prediction).all():
            raise ValueError("Model returned a non-finite raw forecast")
        scale = float(np.mean(np.abs(task.history[7:] - task.history[:-7])))
        if scale == 0:
            scale = float("nan")
        elapsed = time.perf_counter() - start
        rows = pd.DataFrame({
            "task_id": task.task_id,
            "split_name": task.split_name,
            "fold_id": task.fold_id,
            "destination": task.destination,
            "market": task.market,
            "origin_date": task.origin,
            "history_length": task.history_length_label,
            "history_sha256": hashlib.sha256(np.asarray(task.history, dtype="<i8").tobytes()).hexdigest(),
            "training_history_days": len(task.history),
            "effective_context_days": int(result.detail.get("effective_context_days", len(task.history))),
            "model": model,
            "candidate_id": candidate_id,
            "seed": seed,
            "target_date": list(task.target_dates),
            "horizon_day": list(range(1, 29)),
            "prediction_raw": result.raw_prediction,
            "prediction": result.prediction,
            "was_clipped": result.raw_prediction < 0,
            "mase7_scale": scale,
            "fit_seconds": result.fit_seconds,
            "inference_seconds": result.inference_seconds,
            "total_seconds": elapsed,
            "model_detail_json": json.dumps(result.detail, ensure_ascii=False),
        })
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_name(f".{prediction_path.name}.tmp")
        rows.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, prediction_path)
        return {"status": "completed", "model": model, "candidate_id": candidate_id,
                "seconds": elapsed}
    except Exception as exc:
        _atomic_json({
            "task_id": task.task_id, "fold_id": task.fold_id, "market": task.market,
            "model": model, "candidate_id": candidate_id, "seed": seed,
            "attempts": previous_attempts + 1, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, failure_path)
        return {"status": "failed", "model": model, "candidate_id": candidate_id,
                "error": str(exc)}


def run(*, inputs: Path, task_manifest_path: Path, output: Path, search_path: Path, models: tuple[str, ...],
        max_tasks: int | None, workers: int, retry_failures: bool,
        allow_dirty_code: bool) -> dict:
    search_path, inputs, output = Path(search_path).resolve(), Path(inputs).resolve(), Path(output).resolve()
    task_manifest = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
    expected_input = task_manifest["artifacts"]["splits/recovery_tuning/task_inputs.parquet"]["sha256"]
    if inputs.name != "task_inputs.parquet" or _sha(inputs) != expected_input:
        raise ValueError("B validation requires the audited, B-only task_inputs.parquet")
    validate(search_path=search_path, task_manifest_path=task_manifest_path)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if any(model not in MODEL_NAMES for model in models):
        raise ValueError("Unknown baseline in --models")
    if workers < 1 or (max_tasks is not None and max_tasks < 1):
        raise ValueError("workers and max_tasks must be positive")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty and not allow_dirty_code:
        raise ValueError("Commit the model and search code before an auditable B run")
    tasks = [task for task in read_task_inputs(inputs)
             if task.split_name == "recovery_tuning" and task.history_length_label == "all"]
    if len(tasks) != 324 or len({task.fold_id for task in tasks}) != 12:
        raise ValueError("Expected exactly 27 Macau markets × 12 B folds")
    if any(task.destination != "Macau" for task in tasks):
        raise ValueError("Hong Kong must not participate in B tuning")
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    manifest = {
        "schema_version": 1,
        "kind": "B_candidate_predictions",
        "input_sha256": _sha(inputs),
        "search_sha256": _sha(search_path),
        "protocol_sha256": _sha(DEFAULT_PROTOCOL),
        "code_commit": commit,
        "code_dirty": dirty,
        "target_artifact_accessed": False,
        "expected_B_main_tasks": 324,
        "task_limit": max_tasks,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("input_sha256", "search_sha256", "protocol_sha256", "code_commit", "code_dirty", "task_limit"):
            if old[key] != manifest[key]:
                raise ValueError(f"Cannot resume with a changed {key}")
    else:
        _atomic_json(manifest, manifest_path)
    totals = {"completed": 0, "skipped": 0, "failed": 0, "failed_existing": 0}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for model in models:
            candidates = search["models"][model]["candidates"]
            seeds = search["seeds"] if model in search["multi_seed_models"] else [search["single_seed"]]
            for candidate in candidates:
                jobs = ((str(output), model, candidate["id"], candidate["params"], seed,
                         task, retry_failures) for seed in seeds for task in tasks)
                count = 0
                for result in pool.map(_run_job, jobs, chunksize=1):
                    totals[result["status"]] += 1
                    count += 1
                    if count % 27 == 0 or count == len(tasks) * len(seeds):
                        print(json.dumps({"model": model, "candidate": candidate["id"],
                                          "finished_jobs": count, "total_jobs": len(tasks) * len(seeds),
                                          "counts": totals, "elapsed_seconds": round(time.perf_counter() - start, 1)}), flush=True)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated, resumable B-fold candidate forecasts")
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "data" / "splits" / "recovery_tuning" / "task_inputs.parquet")
    parser.add_argument("--task-manifest", type=Path, default=Path(__file__).resolve().parent / "data" / "task_manifest.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "b_validation")
    parser.add_argument("--search", type=Path, default=SEARCH_PATH)
    parser.add_argument("--models", default=",".join(RUN_ORDER))
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--allow-dirty-code", action="store_true", help="Pilot only; cannot support freezing")
    args = parser.parse_args()
    print(json.dumps(run(inputs=args.inputs, task_manifest_path=args.task_manifest,
                         output=args.output, search_path=args.search,
                         models=tuple(x.strip() for x in args.models.split(",") if x.strip()),
                         max_tasks=args.max_tasks, workers=args.workers,
                         retry_failures=args.retry_failures,
                         allow_dirty_code=args.allow_dirty_code)), flush=True)


if __name__ == "__main__":
    main()
