"""Run one foundation model across B01-B12 with the accepted smoke runtime.

This is the registered default runtime, not a candidate search and not a freeze.
Each origin shard is one node. Targets are not mounted.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
SHARDS = Path("/root/DYQ/origin_shards")
REGISTRY = Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml")
JOURNAL = Path("/root/DYQ/run-logs/nodes.jsonl")
INDEX = json.loads((SHARDS / "origin_shards_manifest.json").read_text(encoding="utf-8"))


def _journal(event: dict) -> None:
    import fcntl
    event.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "track1/results/foundation_b_default")
    args = parser.parse_args()
    folds = [shard for shard in INDEX["shards"] if shard["split_name"] == "recovery_tuning"]
    folds.sort(key=lambda shard: shard["fold_id"])
    if len(folds) != 12:
        raise SystemExit(f"expected 12 B shards, found {len(folds)}")
    for shard in folds:
        fold = shard["fold_id"]
        origin = shard["origin_date"]
        shard_input = SHARDS / shard["relative_path"]
        output = args.output_root / f"{fold}_{origin}" / args.model
        manifest = output / "run_manifest.json"
        if manifest.is_file():
            event = {"stage": "foundation_b_default", "event": "origin_skip", "model": args.model,
                     "fold_id": fold, "origin_date": origin, "gpu": args.gpu}
            _journal(event)
            print(json.dumps(event), flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        start_event = {"stage": "foundation_b_default", "event": "origin_start", "model": args.model,
                       "fold_id": fold, "origin_date": origin, "gpu": args.gpu,
                       "history": "all", "runtime": "accepted_smoke_default", "batch_size": args.batch_size}
        _journal(start_event)
        print(json.dumps(start_event), flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["HF_HOME"] = f"/root/DYQ/foundation-cache/{args.model}"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command = [
            str(args.python), str(REPO / "track1/code/run_foundation_backtest.py"),
            "--model", args.model, "--inputs", str(shard_input), "--output", str(output),
            "--registry", str(REGISTRY), "--batch-size", str(args.batch_size), "--history", "all",
        ]
        started = time.perf_counter()
        log_path = output / "node.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        elapsed = round(time.perf_counter() - started, 1)
        done = {"stage": "foundation_b_default", "event": "origin_done", "model": args.model,
                "fold_id": fold, "origin_date": origin, "gpu": args.gpu,
                "returncode": completed.returncode, "elapsed_seconds": elapsed,
                "manifest": manifest.is_file()}
        if manifest.is_file():
            body = json.loads(manifest.read_text(encoding="utf-8"))
            done.update({
                "forecast_tasks": body.get("forecast_tasks"),
                "prediction_rows": body.get("prediction_rows"),
                "peak_gpu_memory_mb": body.get("peak_gpu_memory_mb"),
                "target_artifact_accessed": body.get("target_artifact_accessed"),
            })
        _journal(done)
        print(json.dumps(done), flush=True)
        if completed.returncode != 0:
            raise SystemExit(f"{args.model} {fold} failed, see {log_path}")
    _journal({"stage": "foundation_b_default", "event": "model_done", "model": args.model, "gpu": args.gpu})
    print(json.dumps({"event": "model_done", "model": args.model}), flush=True)


if __name__ == "__main__":
    main()
