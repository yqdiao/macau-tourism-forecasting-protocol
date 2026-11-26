"""Run frozen foundation models on one registered split and one history length.

Does not retune. PatchTST and Informer use the candidate already frozen on B.
A 180-day history cannot train those two models: their frozen context is 180
days and training needs context plus the 28-day horizon. Those jobs are
recorded as inapplicable instead of being launched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_foundation_b_search import REPO, SEARCH_SPACE, _claim, _journal, _release, _sha256

FREEZE_PATH = REPO / "track1/configs/frozen_foundation_configs.yaml"
TRAINED = {"patchtst", "informer"}
EXPECTED = {"pandemic_stress": 27, "outer_backtest": 29}


def _complete(manifest: Path, candidate: str | None, seed: int, history: str, expected: int,
              space_sha: str | None) -> bool:
    if not manifest.is_file():
        return False
    body = json.loads(manifest.read_text(encoding="utf-8"))
    history_ok = body.get("history_lengths") == [history]
    candidate_ok = body.get("candidate_id") == candidate
    space_ok = True if space_sha is None else body.get("search_space_sha256") == space_sha
    return (candidate_ok and space_ok and history_ok and body.get("seed") == seed
            and body.get("forecast_tasks") == expected
            and body.get("target_artifact_accessed") is False)


def _run_job(job: dict, args, space_sha: str | None, expected: int) -> int:
    base = {"stage": "foundation_slice", "model": args.model, "split": args.split,
            "history": args.history, "seed": job["seed"], "fold_id": job["fold"], "gpu": args.gpu}
    output = (args.output_root / args.split / args.history / args.model / f"seed{job['seed']}"
              / f"{job['fold']}_{job['origin']}")
    if job["inapplicable"]:
        note = output / "inapplicable.json"
        if not note.is_file():
            output.mkdir(parents=True, exist_ok=True)
            note.write_text(json.dumps({
                "model": args.model, "candidate_id": job["candidate"], "history": args.history,
                "reason": "frozen context is 180 days and training requires context + 28 forecast days",
            }, indent=2) + "\n", encoding="utf-8")
        _journal(args.journal, {**base, "event": "job_inapplicable"})
        return 0
    manifest = output / "run_manifest.json"
    if _complete(manifest, job["candidate"], job["seed"], args.history, expected, space_sha):
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
                       "--history", args.history, "--seed", str(job["seed"])]
            if job["candidate"]:
                command += ["--search-space", str(SEARCH_SPACE), "--candidate-id", job["candidate"]]
            started = time.perf_counter()
            with (output / f"node_attempt{attempt}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            ok = completed.returncode == 0 and _complete(
                manifest, job["candidate"], job["seed"], args.history, expected, space_sha)
            _journal(args.journal, {**base, "event": "job_done" if ok else "job_failed",
                                    "attempt": attempt, "returncode": completed.returncode,
                                    "elapsed_seconds": round(time.perf_counter() - started, 1)})
            if ok:
                return 0
        return 1
    finally:
        _release(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--split", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--history", required=True, choices=["all", "180", "365"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fold-mod", type=int, default=1)
    parser.add_argument("--fold-rem", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--shards", type=Path, default=Path("/root/DYQ/origin_shards"))
    parser.add_argument("--registry", type=Path,
                        default=Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml"))
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/sensitivity")
    parser.add_argument("--journal", type=Path, default=Path("/root/DYQ/run-logs/sensitivity_nodes.jsonl"))
    args = parser.parse_args()
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_after_B_group_tuning":
        raise SystemExit("foundation config is not frozen")
    entry = frozen["models"][args.model]
    candidate = entry["candidate_id"] if args.model in TRAINED else None
    space_sha = _sha256(SEARCH_SPACE) if candidate else None
    seeds = frozen["seeds"] if args.model in TRAINED else [42]
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    folds = sorted((s for s in index["shards"] if s["split_name"] == args.split), key=lambda s: s["fold_id"])
    folds = [s for s in folds if int(s["fold_id"][1:]) % args.fold_mod == args.fold_rem]
    inapplicable = args.model in TRAINED and args.history == "180"
    jobs = [{"candidate": candidate, "seed": seed, "fold": s["fold_id"], "origin": s["origin_date"],
             "input": str(args.shards / s["relative_path"]), "inapplicable": inapplicable}
            for seed in seeds for s in folds]
    expected = EXPECTED[args.split]
    _journal(args.journal, {"stage": "foundation_slice", "event": "runner_start", "model": args.model,
                            "split": args.split, "history": args.history, "gpu": args.gpu,
                            "jobs": len(jobs), "inapplicable": inapplicable})
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(_run_job, job, args, space_sha, expected) for job in jobs]):
            failures += future.result() != 0
    _journal(args.journal, {"stage": "foundation_slice", "event": "runner_done", "model": args.model,
                            "split": args.split, "history": args.history, "gpu": args.gpu,
                            "jobs": len(jobs), "failures": failures})
    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
