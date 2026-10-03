"""Run one foundation model across outer folds O01-O33.

Uses the same accepted smoke runtime and batch size as the B01-B12 run.
Each origin shard is isolated. Targets are not mounted.
Multiple folds of the same model can run concurrently to occupy a GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
SHARDS = Path("/root/DYQ/origin_shards")
REGISTRY = Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml")
JOURNAL = Path("/root/DYQ/run-logs/nodes.jsonl")
INDEX = json.loads((SHARDS / "origin_shards_manifest.json").read_text(encoding="utf-8"))
EXPECTED_TASKS = 29  # 27 Macau markets + 2 Hong Kong series, history=all


def _journal(event: dict) -> None:
    import fcntl
    event.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim(output: Path) -> bool:
    claim = output / ".running"
    output.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        pid_path = claim / "pid"
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = -1
            if _pid_alive(pid):
                return False
        # Stale claim left by a dead worker.
        for child in claim.iterdir():
            child.unlink()
        claim.rmdir()
        try:
            claim.mkdir()
        except FileExistsError:
            return False
    (claim / "pid").write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release(output: Path) -> None:
    claim = output / ".running"
    if claim.is_dir():
        for child in claim.iterdir():
            child.unlink()
        claim.rmdir()


def _run_fold(shard: dict, args) -> int:
    fold = shard["fold_id"]
    origin = shard["origin_date"]
    shard_input = SHARDS / shard["relative_path"]
    output = args.output_root / f"{fold}_{origin}" / args.model
    manifest = output / "run_manifest.json"
    if manifest.is_file():
        event = {"stage": "foundation_outer_default", "event": "origin_skip", "model": args.model,
                 "fold_id": fold, "origin_date": origin, "gpu": args.gpu}
        _journal(event)
        print(json.dumps(event), flush=True)
        return 0
    if not _claim(output):
        event = {"stage": "foundation_outer_default", "event": "origin_busy", "model": args.model,
                 "fold_id": fold, "origin_date": origin, "gpu": args.gpu}
        _journal(event)
        print(json.dumps(event), flush=True)
        return 0
    start_event = {"stage": "foundation_outer_default", "event": "origin_start", "model": args.model,
                   "fold_id": fold, "origin_date": origin, "gpu": args.gpu,
                   "history": "all", "runtime": "accepted_smoke_default", "batch_size": args.batch_size}
    _journal(start_event)
    print(json.dumps(start_event), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HOME"] = f"/root/DYQ/foundation-cache/{args.model}"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    command = [
        str(args.python), str(REPO / "track1/code/run_foundation_backtest.py"),
        "--model", args.model, "--inputs", str(shard_input), "--output", str(output),
        "--registry", str(REGISTRY), "--batch-size", str(args.batch_size), "--history", "all",
    ]
    started = time.perf_counter()
    log_path = output / "node.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        elapsed = round(time.perf_counter() - started, 1)
        done = {"stage": "foundation_outer_default", "event": "origin_done", "model": args.model,
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
            if body.get("forecast_tasks") != EXPECTED_TASKS or body.get("target_artifact_accessed") is not False:
                done["returncode"] = 2
                done["error"] = f"expected {EXPECTED_TASKS} tasks and no target access"
        _journal(done)
        print(json.dumps(done), flush=True)
        code = done["returncode"]
        return code if code == 0 else code or 1
    finally:
        _release(output)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fold-mod", type=int, default=1)
    parser.add_argument("--fold-rem", type=int, default=0)
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "track1/results/foundation_outer_default")
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    folds = [shard for shard in INDEX["shards"] if shard["split_name"] == "outer_backtest"]
    folds.sort(key=lambda shard: shard["fold_id"])
    if len(folds) != 33:
        raise SystemExit(f"expected 33 outer shards, found {len(folds)}")
    if args.fold_mod > 1:
        folds = [shard for shard in folds if int(shard["fold_id"][1:]) % args.fold_mod == args.fold_rem]
    print(json.dumps({
        "event": "model_start", "model": args.model, "gpu": args.gpu, "jobs": args.jobs,
        "folds": [shard["fold_id"] for shard in folds], "batch_size": args.batch_size,
        "expected_tasks_per_fold": EXPECTED_TASKS,
    }), flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run_fold, shard, args) for shard in folds]
        for future in as_completed(futures):
            if future.result() != 0:
                failures += 1
    event = {"stage": "foundation_outer_default", "event": "model_done", "model": args.model,
             "gpu": args.gpu, "failures": failures, "folds": len(folds)}
    _journal(event)
    print(json.dumps(event), flush=True)
    if failures:
        raise SystemExit(f"{args.model} gpu {args.gpu} failed folds: {failures}")


if __name__ == "__main__":
    main()
