"""Run the registered PatchTST/Informer candidate search on B01-B12.

Every (candidate, seed, fold) job writes to its own directory:
    <output-root>/<model>/<candidate>/seed<seed>/<fold>_<origin>/
A job is complete only when its run_manifest.json matches the requested
candidate, seed and the registered search-space SHA256. Targets are never
mounted; each job sees one origin shard.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEARCH_SPACE = REPO / "track1/configs/foundation_search_spaces.yaml"
EXPECTED_TASKS = 27  # B01-B12 contain the 27 Macau markets only


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path, event: dict) -> None:
    event.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return pid > 0


def _claim(output: Path) -> bool:
    claim = output / ".running"
    output.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        pid_path = claim / "pid"
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if _pid_alive(pid):
            return False
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


def _complete(manifest: Path, candidate: str, seed: int, space_sha: str) -> bool:
    if not manifest.is_file():
        return False
    body = json.loads(manifest.read_text(encoding="utf-8"))
    return (body.get("candidate_id") == candidate and body.get("seed") == seed
            and body.get("search_space_sha256") == space_sha
            and body.get("forecast_tasks") == EXPECTED_TASKS
            and body.get("target_artifact_accessed") is False)


def _run_job(job: dict, args, space_sha: str) -> int:
    base = {"stage": "foundation_b_search", "model": args.model, "candidate_id": job["candidate"],
            "seed": job["seed"], "fold_id": job["fold"], "gpu": args.gpu}
    output = (args.output_root / args.model / job["candidate"] / f"seed{job['seed']}"
              / f"{job['fold']}_{job['origin']}")
    manifest = output / "run_manifest.json"
    if _complete(manifest, job["candidate"], job["seed"], space_sha):
        _journal(args.journal, {**base, "event": "job_skip"})
        return 0
    if not _claim(output):
        _journal(args.journal, {**base, "event": "job_busy"})
        return 0
    try:
        for attempt in range(1, args.max_attempts + 1):
            manifest.unlink(missing_ok=True)
            _journal(args.journal, {**base, "event": "job_start", "attempt": attempt})
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "HF_HOME": f"/root/DYQ/foundation-cache/{args.model}",
                        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
            command = [str(args.python), str(REPO / "track1/code/run_foundation_backtest.py"),
                       "--model", args.model, "--inputs", job["input"], "--output", str(output),
                       "--registry", str(args.registry), "--batch-size", str(args.batch_size),
                       "--history", "all", "--seed", str(job["seed"]),
                       "--search-space", str(SEARCH_SPACE), "--candidate-id", job["candidate"]]
            started = time.perf_counter()
            with (output / f"node_attempt{attempt}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            ok = completed.returncode == 0 and _complete(manifest, job["candidate"], job["seed"], space_sha)
            _journal(args.journal, {**base, "event": "job_done" if ok else "job_failed", "attempt": attempt,
                                    "returncode": completed.returncode,
                                    "elapsed_seconds": round(time.perf_counter() - started, 1)})
            if ok:
                return 0
        return 1
    finally:
        _release(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["patchtst", "informer"])
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--job-mod", type=int, default=1)
    parser.add_argument("--job-rem", type=int, default=0)
    parser.add_argument("--candidates", help="Comma-separated subset, for smoke runs")
    parser.add_argument("--seeds", help="Comma-separated subset, for smoke runs")
    parser.add_argument("--folds", help="Comma-separated subset, for smoke runs")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="1 run plus at most 2 identical retries, per the failure policy")
    parser.add_argument("--shards", type=Path, default=Path("/root/DYQ/origin_shards"))
    parser.add_argument("--registry", type=Path,
                        default=Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml"))
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/foundation_b_search")
    parser.add_argument("--journal", type=Path, default=Path("/root/DYQ/run-logs/b_search_nodes.jsonl"))
    args = parser.parse_args()

    space = json.loads(SEARCH_SPACE.read_text(encoding="utf-8"))
    if space.get("status") != "registered_before_tuning":
        raise SystemExit("search space is not registered")
    space_sha = _sha256(SEARCH_SPACE)
    candidates = [c["id"] for c in space["models"][args.model]["candidates"]]
    seeds = list(space["seeds"])
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    folds = sorted((s for s in index["shards"] if s["split_name"] == "recovery_tuning"),
                   key=lambda s: s["fold_id"])
    if [s["fold_id"] for s in folds] != space["fold_ids"]:
        raise SystemExit("origin shards do not match the registered B folds")
    if args.candidates:
        wanted = args.candidates.split(",")
        if not set(wanted) <= set(candidates):
            raise SystemExit(f"unknown candidates: {set(wanted) - set(candidates)}")
        candidates = wanted
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.folds:
        folds = [s for s in folds if s["fold_id"] in args.folds.split(",")]

    jobs = [{"candidate": c, "seed": seed, "fold": s["fold_id"], "origin": s["origin_date"],
             "input": str(args.shards / s["relative_path"])}
            for seed in seeds for c in candidates for s in folds]
    jobs = [job for i, job in enumerate(jobs) if i % args.job_mod == args.job_rem]
    _journal(args.journal, {"stage": "foundation_b_search", "event": "runner_start", "model": args.model,
                            "gpu": args.gpu, "jobs": len(jobs), "parallel": args.jobs,
                            "search_space_sha256": space_sha})
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(_run_job, job, args, space_sha) for job in jobs]):
            failures += future.result() != 0
    _journal(args.journal, {"stage": "foundation_b_search", "event": "runner_done", "model": args.model,
                            "gpu": args.gpu, "jobs": len(jobs), "failures": failures})
    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
