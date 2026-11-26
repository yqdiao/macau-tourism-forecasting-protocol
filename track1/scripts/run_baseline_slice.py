"""Frozen-baseline forecasts for one split and one history length. Does not retune."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "track1" / "code"))

from baseline_config import SEARCH_PATH, validate as validate_baseline_config
from data_pipeline import read_task_inputs
from models import forecast

FREEZE = REPO / "track1/configs/frozen_baseline_configs.yaml"
MANIFEST = REPO / "shared/data/task_manifest.json"
INPUTS = REPO / "shared/data/task_inputs.parquet"
JOURNAL = Path("/root/DYQ/run-logs/sensitivity_nodes.jsonl")


def _journal(event: dict) -> None:
    import fcntl
    event.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print(json.dumps(event, ensure_ascii=False), flush=True)


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
            return {"status": "skipped", "model": model, "fold_id": task.fold_id, "seconds": 0.0}
    start = time.perf_counter()
    try:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = forecast(model, task.history, 28, seed, config=params)
        if len(result.prediction) != 28 or not np.isfinite(result.raw_prediction).all():
            raise ValueError("forecast is not 28 finite values")
        scale = float(np.mean(np.abs(task.history[7:] - task.history[:-7]))) or float("nan")
        elapsed = time.perf_counter() - start
        rows = pd.DataFrame({
            "task_id": task.task_id, "split_name": task.split_name, "fold_id": task.fold_id,
            "destination": task.destination, "market": task.market, "origin_date": task.origin,
            "history_length": task.history_length_label, "training_history_days": len(task.history),
            "model": model, "candidate_id": candidate_id, "seed": seed,
            "target_date": list(task.target_dates), "horizon_day": list(range(1, 29)),
            "prediction_raw": result.raw_prediction, "prediction": result.prediction,
            "mase7_scale": scale, "fit_seconds": result.fit_seconds,
            "inference_seconds": result.inference_seconds, "total_seconds": elapsed,
        })
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_name(f".{prediction_path.name}.tmp")
        rows.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, prediction_path)
        return {"status": "completed", "model": model, "fold_id": task.fold_id, "seconds": round(elapsed, 3)}
    except Exception as exc:
        _atomic_json({"task_id": task.task_id, "model": model, "error": str(exc),
                      "traceback": traceback.format_exc()}, failure_path)
        return {"status": "failed", "model": model, "fold_id": task.fold_id,
                "error": str(exc), "seconds": round(time.perf_counter() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--split", required=True, choices=["pandemic_stress", "outer_backtest"])
    parser.add_argument("--history", required=True, choices=["all", "180", "365"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = validate_baseline_config(search_path=SEARCH_PATH, freeze_path=FREEZE, task_manifest_path=MANIFEST)
    if audit["freeze_status"] != "frozen_after_B_group_tuning":
        raise SystemExit("baseline config is not frozen")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    selected = {}
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        chosen = freeze["models"][name]["candidate_id"]
        match = [c for c in search["models"][name]["candidates"] if c["id"] == chosen]
        if len(match) != 1:
            raise SystemExit(f"{name}: frozen candidate missing")
        selected[name] = (chosen, match[0]["params"])
    tasks = [task for task in read_task_inputs(INPUTS)
             if task.split_name == args.split and task.history_length_label == args.history]
    _journal({"stage": "baseline_slice", "event": "group_start", "split": args.split,
              "history": args.history, "models": list(selected), "tasks": len(tasks)})
    jobs = [(str(args.output), name, candidate, params, 42, task)
            for name, (candidate, params) in selected.items() for task in tasks]
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_run_job, jobs, chunksize=1):
            counts[result["status"]] += 1
            if result["status"] == "failed":
                _journal({"stage": "baseline_slice", "event": "task_failed", "split": args.split,
                          "history": args.history, **result})
    _journal({"stage": "baseline_slice", "event": "group_done", "split": args.split,
              "history": args.history, "models": list(selected), "counts": counts})
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} baseline tasks failed")


if __name__ == "__main__":
    main()
