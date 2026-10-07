"""Run the registered lightweight global models on one split.

B01-B12 runs every registered candidate. Other splits and the two ablations run
only the frozen candidates in track1/configs/frozen_light_configs.yaml.
Output: <output-root>/<split>/<model>/<candidate>/seed<seed>/<fold>_<origin>/
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "track1/configs/light_model_plan.yaml"
FREEZE = REPO / "track1/configs/frozen_light_configs.yaml"
CALENDAR = REPO / "shared/data/holiday_calendar.parquet"
CODE = REPO / "track1/code/light_models.py"
MACAU_MARKETS = 27
FAMILY_MODEL = {"global_lightgbm": "global_lightgbm", "global_linear": "global_linear"}
ABLATIONS = {"GL-noCal": "global_lightgbm_nocal", "GL-local": "global_lightgbm_local"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _journal(path: Path, event: dict) -> None:
    event.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _atomic_write(path: Path, write) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write(temporary)
    os.replace(temporary, path)


def _limit_threads(threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _complete(manifest: Path, job: dict, plan_sha: str, code_sha: str) -> bool:
    if not manifest.is_file():
        return False
    body = json.loads(manifest.read_text(encoding="utf-8"))
    return (body.get("model") == job["model"] and body.get("candidate_id") == job["candidate"]
            and body.get("seed") == job["seed"] and body.get("plan_sha256") == plan_sha
            and body.get("code_sha256") == code_sha and body.get("forecast_tasks") == MACAU_MARKETS
            and body.get("target_artifact_accessed") is False)


def _run_job(job: dict) -> dict:
    _limit_threads(job["threads"])
    sys.path.insert(0, str(REPO / "track1/code"))
    import numpy as np
    import pandas as pd
    from data_pipeline import read_task_inputs
    import light_models

    output = Path(job["output"])
    manifest_path = output / "run_manifest.json"
    if _complete(manifest_path, job, job["plan_sha"], job["code_sha"]):
        return {**job["base"], "event": "job_skip"}
    output.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(1, job["max_attempts"] + 1):
        started = time.perf_counter()
        try:
            tasks = [task for task in read_task_inputs(Path(job["input"]))
                     if task.history_length_label == "all" and task.destination == "Macau"]
            if len(tasks) != MACAU_MARKETS:
                raise ValueError(f"expected {MACAU_MARKETS} Macau tasks, got {len(tasks)}")
            histories = [light_models.MarketHistory(t.task_id, t.market, t.history_start, t.history)
                         for t in tasks]
            params = job["params"]
            if job["family"] == "global_lightgbm":
                calendar = None if job["ablation"] == "GL-noCal" else light_models.Calendar(CALENDAR)
                result = light_models.global_lightgbm(histories, params, calendar,
                                                      pooled=job["ablation"] != "GL-local",
                                                      threads=job["threads"], seed=job["seed"])
            else:
                result = light_models.global_linear(histories, params, seed=job["seed"],
                                                    threads=job["threads"])
            rows = []
            for task in tasks:
                raw = np.asarray(result.predictions[task.market], dtype=float)
                if raw.shape != (28,) or not np.isfinite(raw).all():
                    raise ValueError(f"{task.market}: forecast is not 28 finite values")
                scale_values = task.history[7:] - task.history[:-7]
                scale = float(np.mean(np.abs(scale_values))) or None
                for h, (target_date, value) in enumerate(zip(task.target_dates, raw), 1):
                    rows.append({"task_id": task.task_id, "split_name": task.split_name,
                                 "fold_id": task.fold_id, "destination": task.destination,
                                 "market": task.market, "origin_date": task.origin,
                                 "history_length": task.history_length_label,
                                 "target_date": target_date, "horizon_day": h,
                                 "model": job["model"], "candidate_id": job["candidate"],
                                 "seed": job["seed"], "prediction_raw": float(value),
                                 "prediction": max(0.0, float(value)), "mase7_scale": scale})
            frame = pd.DataFrame(rows)
            prediction_path = output / "predictions.parquet"
            _atomic_write(prediction_path, lambda p: frame.to_parquet(p, index=False))
            per_task = (result.fit_seconds + result.inference_seconds) / len(tasks)
            timing_path = output / "timings.jsonl"
            lines = "".join(json.dumps({"task_id": t.task_id, "inference_seconds": per_task}) + "\n"
                            for t in tasks)
            _atomic_write(timing_path, lambda p: p.write_text(lines, encoding="utf-8"))
            wall = time.perf_counter() - started
            manifest = {
                "model": job["model"], "family": job["family"], "candidate_id": job["candidate"],
                "ablation": job["ablation"], "seed": job["seed"], "params": job["params"],
                "split_name": job["split"], "fold_id": job["fold"], "origin_date": job["origin"],
                "forecast_tasks": len(tasks), "prediction_rows": len(frame), "threads": job["threads"],
                "fit_seconds": result.fit_seconds, "inference_seconds": result.inference_seconds,
                "wall_seconds": wall, "detail": result.detail,
                "plan_sha256": job["plan_sha"], "code_sha256": job["code_sha"],
                "calendar_sha256": job["calendar_sha"], "input_sha256": _sha(Path(job["input"])),
                "target_artifact_accessed": False,
                "outputs": {"predictions.parquet": _sha(prediction_path), "timings.jsonl": _sha(timing_path)},
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
            }
            _atomic_write(manifest_path, lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n",
                                                                encoding="utf-8"))
            return {**job["base"], "event": "job_done", "attempt": attempt, "wall_seconds": round(wall, 1),
                    "fit_seconds": round(result.fit_seconds, 1)}
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            error = f"{type(exc).__name__}: {exc}"
            (output / f"failure_attempt{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return {**job["base"], "event": "job_failed", "error": error}


def _jobs(args, plan: dict) -> list[dict]:
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    shards = sorted((s for s in index["shards"] if s["split_name"] == args.split), key=lambda s: s["fold_id"])
    if args.folds:
        shards = [s for s in shards if s["fold_id"] in args.folds]
    plan_sha, code_sha, calendar_sha = _sha(PLAN), _sha(CODE), _sha(CALENDAR)
    if calendar_sha != plan["calendar"]["sha256"]:
        raise SystemExit("holiday calendar differs from the registered SHA256")
    entries = []
    if args.ablation:
        frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
        gl = frozen["families"]["global_lightgbm"]
        entries.append(("global_lightgbm", ABLATIONS[args.ablation], gl["candidate_id"], gl["params"], [42]))
    elif args.split == "recovery_tuning":
        for family in args.families:
            spec = plan["families"][family]
            seeds = spec["fixed"].get("seeds", [spec["fixed"].get("seed", 42)])
            for candidate in spec["candidates"]:
                if args.candidates and candidate["id"] not in args.candidates:
                    continue
                entries.append((family, FAMILY_MODEL[family], candidate["id"], candidate["params"], seeds))
    else:
        if not FREEZE.is_file():
            raise SystemExit("splits other than B require frozen_light_configs.yaml")
        frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
        for family in args.families:
            entry = frozen["families"][family]
            seeds = plan["families"][family]["fixed"].get("seeds", [42])
            entries.append((family, FAMILY_MODEL[family], entry["candidate_id"], entry["params"], seeds))
    jobs = []
    for family, model, candidate, params, seeds in entries:
        threads = int(plan["families"][family]["fixed"]["threads"])
        for seed in seeds:
            if args.seeds and seed not in args.seeds:
                continue
            for shard in shards:
                output = (args.output_root / args.split / model / candidate / f"seed{seed}"
                          / f"{shard['fold_id']}_{shard['origin_date']}")
                jobs.append({"family": family, "model": model, "candidate": candidate, "params": params,
                             "seed": seed, "fold": shard["fold_id"], "origin": shard["origin_date"],
                             "split": args.split, "input": str(args.shards / shard["relative_path"]),
                             "output": str(output), "threads": threads, "ablation": args.ablation,
                             "plan_sha": plan_sha, "code_sha": code_sha, "calendar_sha": calendar_sha,
                             "max_attempts": args.max_attempts,
                             "base": {"stage": "light_models", "model": model, "candidate_id": candidate,
                                      "seed": seed, "fold_id": shard["fold_id"], "split": args.split}})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["recovery_tuning", "outer_backtest", "pandemic_stress"])
    parser.add_argument("--families", default="global_lightgbm,global_linear")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--folds", default="")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--shards", type=Path, default=Path("/root/DYQ/origin_shards"))
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/light_models")
    parser.add_argument("--journal", type=Path, default=Path("/root/DYQ/run-logs/light_models_nodes.jsonl"))
    args = parser.parse_args()
    args.families = [x for x in args.families.split(",") if x]
    args.candidates = {x for x in args.candidates.split(",") if x}
    args.seeds = {int(x) for x in args.seeds.split(",") if x}
    args.folds = {x for x in args.folds.split(",") if x}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "registered_before_tuning":
        raise SystemExit("light model plan is not registered")
    jobs = _jobs(args, plan)
    _journal(args.journal, {"stage": "light_models", "event": "runner_start", "split": args.split,
                            "jobs": len(jobs), "parallel": args.jobs, "ablation": args.ablation,
                            "families": args.families})
    failures = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        for future in as_completed([pool.submit(_run_job, job) for job in jobs]):
            event = future.result()
            failures += event["event"] == "job_failed"
            _journal(args.journal, event)
    _journal(args.journal, {"stage": "light_models", "event": "runner_done", "split": args.split,
                            "jobs": len(jobs), "failures": failures})
    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
