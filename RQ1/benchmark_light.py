"""Isolated CPU timing for GL03, LL01 and NL01 under the registered benchmark protocol.

Schedule from track1/configs/volume_breakeven_plan.yaml: warm-up O02 (discarded),
then O01, O17, O33 three times each = 9 measured runs per model. One process
forecasts the 27 Macau markets of an origin with the registered thread count;
allocated vCPU seconds = wall seconds x threads. Targets are never read.
Appends to RQ1/raw/light_benchmark.jsonl.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
PLAN = REPO / "track1/configs/volume_breakeven_plan.yaml"
CALENDAR = REPO / "shared/data/holiday_calendar.parquet"
SHARDS = Path("/root/DYQ/origin_shards")
LOG = REPO / "RQ1/raw/light_benchmark.jsonl"
MODELS = {
    # config id: (family, params, threads, seed)
    "GL03": ("global_lightgbm", {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 1000, "anchor_window": "all"}, 4, 42),
    "LL01": ("local_lightgbm", {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 1000, "anchor_window": "all"}, 4, 42),
    "NL01": ("global_linear", {"model": "nlinear", "lookback": 56}, 2, 42),
}


def schedule() -> list[tuple[str, int, bool]]:
    bench = json.loads(PLAN.read_text())["benchmark"]
    runs = [(bench["warmup_origin"], 0, True)]
    runs += [(fold, rep, False) for rep in range(1, bench["repeats"] + 1) for fold in bench["origins"]]
    return runs


def run_once(config: str, fold: str) -> dict:
    family, params, threads, seed = MODELS[config]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(REPO / "track1/code"))
    from data_pipeline import read_task_inputs
    import light_models

    manifest = json.loads((SHARDS / "origin_shards_manifest.json").read_text())
    shard = next(s for s in manifest["shards"] if s["split_name"] == "outer_backtest" and s["fold_id"] == fold)
    load_started = time.perf_counter()
    tasks = [t for t in read_task_inputs(SHARDS / shard["relative_path"])
             if t.history_length_label == "all" and t.destination == "Macau"]
    histories = [light_models.MarketHistory(t.task_id, t.market, t.history_start, t.history) for t in tasks]
    calendar = light_models.Calendar(CALENDAR) if family != "global_linear" else None
    load_seconds = time.perf_counter() - load_started
    cpu_before = time.process_time()
    started = time.perf_counter()
    if family == "global_linear":
        result = light_models.global_linear(histories, params, seed=seed, threads=threads)
    else:
        result = light_models.global_lightgbm(histories, params, calendar, pooled=family == "global_lightgbm",
                                              threads=threads, seed=seed)
    wall = time.perf_counter() - started
    return {"model": config, "family": family, "fold_id": fold, "tasks": len(tasks), "threads": threads,
            "load_seconds": load_seconds, "wall_seconds": wall, "fit_seconds": result.fit_seconds,
            "inference_seconds": result.inference_seconds, "cpu_seconds": time.process_time() - cpu_before,
            "allocated_vcpu_seconds": wall * threads,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if LOG.is_file():
        done = {(r["model"], r["fold_id"], r["repeat"]) for r in map(json.loads, LOG.read_text().splitlines())}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for config in MODELS:
        for fold, rep, warmup in schedule():
            if (config, fold, rep) in done:
                continue
            # A fresh process per run keeps imports, caches and thread pools isolated.
            with ProcessPoolExecutor(max_workers=1) as pool:
                row = pool.submit(run_once, config, fold).result()
            row.update({"repeat": rep, "warmup": warmup, "at_utc": datetime.now(timezone.utc).isoformat()})
            with LOG.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("model", "fold_id", "repeat", "warmup", "wall_seconds",
                                                    "allocated_vcpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()
