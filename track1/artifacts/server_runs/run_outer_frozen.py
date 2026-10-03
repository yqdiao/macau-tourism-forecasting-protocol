"""Frozen-config outer backtest, one atomic shard per task.

Uses the already frozen B-group candidate for each baseline. History is `all`.
Seed is 42, including transformer. This does not retune and does not read targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
sys.path.insert(0, str(REPO / "track1" / "code"))

from baseline_config import SEARCH_PATH, validate as validate_baseline_config
from data_pipeline import read_task_inputs
from models import forecast

JOURNAL = Path("/root/DYQ/run-logs/nodes.jsonl")
FREEZE = REPO / "track1/configs/frozen_baseline_configs.yaml"
MANIFEST = REPO / "shared/data/task_manifest.json"
INPUTS = REPO / "shared/data/task_inputs.parquet"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal(event: dict) -> None:
    import fcntl
    event.setdefault("at_utc", _now())
    line = json.dumps(event, ensure_ascii=False)
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(line + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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
    import numpy as np
    import pandas as pd

    output, model, candidate_id, params, seed, task = job
    output = Path(output)
    relative = Path(model) / candidate_id / str(seed) / f"{task.fold_id}_{task.task_id}.parquet"
    prediction_path = output / "predictions" / relative
    failure_path = output / "failures" / relative.with_suffix(".json")
    if prediction_path.is_file():
        frame = pd.read_parquet(prediction_path, columns=["horizon_day"])
        if frame["horizon_day"].tolist() == list(range(1, 29)):
            return {"status": "skipped", "model": model, "fold_id": task.fold_id,
                    "market": task.market, "seconds": 0.0}
        raise ValueError(f"Incomplete existing prediction shard: {prediction_path}")
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
        return {"status": "completed", "model": model, "fold_id": task.fold_id,
                "origin_date": task.origin.isoformat(), "market": task.market,
                "destination": task.destination, "candidate_id": candidate_id,
                "seconds": round(elapsed, 3)}
    except Exception as exc:
        _atomic_json({
            "task_id": task.task_id, "fold_id": task.fold_id, "market": task.market,
            "model": model, "candidate_id": candidate_id, "seed": seed,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "failed_at_utc": _now(),
        }, failure_path)
        return {"status": "failed", "model": model, "fold_id": task.fold_id,
                "market": task.market, "error": str(exc), "seconds": round(time.perf_counter() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    audit = validate_baseline_config(search_path=SEARCH_PATH, freeze_path=FREEZE, task_manifest_path=MANIFEST)
    if audit["freeze_status"] != "frozen_after_B_group_tuning":
        raise SystemExit("frozen config is not ready")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    selected = {}
    for name in models:
        chosen = freeze["models"][name]["candidate_id"]
        match = [c for c in search["models"][name]["candidates"] if c["id"] == chosen]
        if len(match) != 1:
            raise SystemExit(f"{name}: frozen candidate missing")
        selected[name] = (chosen, match[0]["params"])
    tasks = [task for task in read_task_inputs(INPUTS)
             if task.split_name == "outer_backtest" and task.history_length_label == "all"]
    folds = sorted({task.fold_id for task in tasks})
    if len(folds) != 33 or len(tasks) != 33 * 29:
        raise SystemExit(f"expected 33 origins x 29 series, got {len(folds)} origins and {len(tasks)} tasks")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    header = {
        "stage": "baseline_outer", "event": "group_start", "models": list(models),
        "workers": args.workers, "seed": 42, "history": "all", "folds": folds,
        "tasks": len(tasks), "freeze_sha256": audit["freeze_sha256"],
        "input_sha256": _sha(INPUTS), "target_artifact_accessed": False,
    }
    _journal(header)
    print(json.dumps(header), flush=True)
    jobs = []
    for name, (candidate_id, params) in selected.items():
        for task in tasks:
            jobs.append((str(output), name, candidate_id, params, 42, task))
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    origin_done: dict[tuple[str, str], int] = {}
    origin_seconds: dict[tuple[str, str], float] = {}
    expected_per_origin = 29
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_run_job, jobs, chunksize=1):
            counts[result["status"]] += 1
            key = (result["model"], result["fold_id"])
            origin_done[key] = origin_done.get(key, 0) + 1
            origin_seconds[key] = origin_seconds.get(key, 0.0) + float(result.get("seconds") or 0)
            if result["status"] == "failed" or origin_done[key] == expected_per_origin:
                event = {
                    "stage": "baseline_outer",
                    "event": "origin_done" if origin_done[key] == expected_per_origin else "task_failed",
                    "model": result["model"], "fold_id": result["fold_id"],
                    "finished_tasks": origin_done[key], "expected_tasks": expected_per_origin,
                    "origin_fit_seconds": round(origin_seconds[key], 3),
                    "status": result["status"], "market": result.get("market"),
                    "error": result.get("error"),
                    "group_counts": counts,
                    "elapsed_seconds": round(time.perf_counter() - start, 1),
                }
                _journal(event)
                print(json.dumps(event, ensure_ascii=False), flush=True)
    footer = {"stage": "baseline_outer", "event": "group_done", "models": list(models), "counts": counts,
              "elapsed_seconds": round(time.perf_counter() - start, 1)}
    _journal(footer)
    print(json.dumps(footer), flush=True)


if __name__ == "__main__":
    main()
