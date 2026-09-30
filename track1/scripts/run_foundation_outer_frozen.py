"""Run a frozen PatchTST/Informer candidate on O01-O33 with every registered seed.

Output: <output-root>/<model>/<candidate>/seed<seed>/<fold>_<origin>/
The candidate comes from frozen_foundation_configs.yaml, never from the command line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_foundation_b_search import REPO, SEARCH_SPACE, _claim, _journal, _release, _sha256  # noqa: E402

FREEZE_PATH = REPO / "track1/configs/frozen_foundation_configs.yaml"
EXPECTED_TASKS = 29  # 27 Macau markets + 2 Hong Kong series, history=all


def _complete(manifest: Path, candidate: str, seed: int, space_sha: str) -> bool:
    if not manifest.is_file():
        return False
    body = json.loads(manifest.read_text(encoding="utf-8"))
    return (body.get("candidate_id") == candidate and body.get("seed") == seed
            and body.get("search_space_sha256") == space_sha
            and body.get("forecast_tasks") == EXPECTED_TASKS
            and body.get("target_artifact_accessed") is False)


def _run_job(job: dict, args, space_sha: str) -> int:
    base = {"stage": "foundation_outer_frozen", "model": args.model, "candidate_id": job["candidate"],
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
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--shards", type=Path, default=Path("/root/DYQ/origin_shards"))
    parser.add_argument("--registry", type=Path,
                        default=Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml"))
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/foundation_outer_frozen")
    parser.add_argument("--journal", type=Path, default=Path("/root/DYQ/run-logs/outer_frozen_nodes.jsonl"))
    args = parser.parse_args()

    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    space_sha = _sha256(SEARCH_SPACE)
    if frozen.get("status") != "frozen_after_B_group_tuning" or frozen["source_search_space_sha256"] != space_sha:
        raise SystemExit("frozen config is missing or does not match the registered search space")
    candidate = frozen["models"][args.model]["candidate_id"]
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    folds = sorted((s for s in index["shards"] if s["split_name"] == "outer_backtest"),
                   key=lambda s: s["fold_id"])
    if len(folds) != 33:
        raise SystemExit(f"expected 33 outer shards, found {len(folds)}")
    jobs = [{"candidate": candidate, "seed": seed, "fold": s["fold_id"], "origin": s["origin_date"],
             "input": str(args.shards / s["relative_path"])}
            for seed in frozen["seeds"] for s in folds]
    jobs = [job for i, job in enumerate(jobs) if i % args.job_mod == args.job_rem]
    _journal(args.journal, {"stage": "foundation_outer_frozen", "event": "runner_start", "model": args.model,
                            "candidate_id": candidate, "gpu": args.gpu, "jobs": len(jobs),
                            "parallel": args.jobs, "frozen_config_sha256": _sha256(FREEZE_PATH)})
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(_run_job, job, args, space_sha) for job in jobs]):
            failures += future.result() != 0
    _journal(args.journal, {"stage": "foundation_outer_frozen", "event": "runner_done", "model": args.model,
                            "gpu": args.gpu, "jobs": len(jobs), "failures": failures})
    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
