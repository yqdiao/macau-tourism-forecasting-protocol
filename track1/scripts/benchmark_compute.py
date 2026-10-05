"""Isolated compute benchmark for the volume-breakeven analysis.

GPU models run one at a time on one GPU; each origin is one process holding all
29 tasks. CPU baselines run one model at a time with the 29 tasks of an origin
as concurrent processes. Targets are never read. Every measured run appends
one JSON line to <output-root>/benchmark.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "track1/configs/volume_breakeven_plan.yaml"
SEARCH_SPACE = REPO / "track1/configs/foundation_search_spaces.yaml"
FROZEN_FOUNDATION = REPO / "track1/configs/frozen_foundation_configs.yaml"
FROZEN_BASELINE = REPO / "track1/configs/frozen_baseline_configs.yaml"
BASELINE_SEARCH = REPO / "track1/configs/baseline_search_spaces.yaml"
INPUTS = REPO / "shared/data/task_inputs.parquet"
TASKS_PER_ORIGIN = 29

GPU_MODELS = {
    "timesfm_3": ("/root/DYQ/timesfm-cu128/bin/python", 8),
    "chronos_2": ("/root/DYQ/chronos-cu128/bin/python", 8),
    "moirai": ("/root/DYQ/moirai-cu128/bin/python", 1),
    "granite_patchtst_fm": ("/root/DYQ/granite-cu128/bin/python", 1),
    "lag_llama": ("/root/DYQ/cu121-py311/bin/python", 1),
    "patchtst": ("/root/DYQ/timesfm-cu128/bin/python", 32),
    "informer": ("/root/DYQ/cu121-py311/bin/python", 16),
}
TRAINED = {"patchtst", "informer"}
BASELINE_THREADS = {"random_forest": 2, "parrot_random_forest_800": 2, "lightgbm": 2, "transformer": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _schedule(plan: dict) -> list[tuple[str, int, bool]]:
    bench = plan["benchmark"]
    runs = [(bench["warmup_origin"], 0, True)]
    runs += [(fold, rep, False) for rep in range(1, bench["repeats"] + 1) for fold in bench["origins"]]
    return runs


class _PowerSampler:
    """Samples one GPU with nvidia-smi every 500 ms."""

    def __init__(self, gpu: str):
        self.gpu, self.samples, self._proc = gpu, [], None

    def __enter__(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            try:
                power, util, memory = (float(x) for x in line.strip().split(","))
                self.samples.append((power, util, memory))
            except ValueError:
                continue

    def __exit__(self, *exc):
        self._proc.terminate()
        self._proc.wait()

    def summary(self, wall: float) -> dict:
        if not self.samples:
            return {"power_samples": 0}
        powers = [s[0] for s in self.samples]
        mean_power = sum(powers) / len(powers)
        return {"power_samples": len(self.samples), "mean_power_w": mean_power, "max_power_w": max(powers),
                "energy_wh": mean_power * wall / 3600,
                "mean_gpu_util_pct": sum(s[1] for s in self.samples) / len(self.samples),
                "max_device_memory_mb": max(s[2] for s in self.samples)}


def run_gpu(model: str, args, plan: dict) -> None:
    python, batch_size = GPU_MODELS[model]
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    shards = {s["fold_id"]: s for s in index["shards"] if s["split_name"] == "outer_backtest"}
    candidate = None
    if model in TRAINED:
        candidate = json.loads(FROZEN_FOUNDATION.read_text(encoding="utf-8"))["models"][model]["candidate_id"]
    for fold, rep, warmup in _schedule(plan):
        shard = shards[fold]
        output = args.output_root / "gpu" / model / f"{fold}_rep{rep}"
        if (output / "run_manifest.json").is_file():
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [python, str(REPO / "track1/code/run_foundation_backtest.py"), "--model", model,
                   "--inputs", str(args.shards / shard["relative_path"]), "--output", str(output),
                   "--registry", str(args.registry), "--batch-size", str(batch_size),
                   "--history", "all", "--seed", "42"]
        if candidate:
            command += ["--search-space", str(SEARCH_SPACE), "--candidate-id", candidate]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "HF_HOME": f"/root/DYQ/foundation-cache/{model}",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with _PowerSampler(args.gpu) as sampler, (output / "node.log").open("w") as log:
            started = time.perf_counter()
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            wall = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        row = {"kind": "gpu", "model": model, "candidate_id": candidate, "fold_id": fold, "repeat": rep,
               "warmup": warmup, "returncode": completed.returncode, "wall_seconds": wall,
               "child_cpu_seconds": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
               "batch_size": batch_size, "gpu": args.gpu, "at_utc": _now(), **sampler.summary(wall)}
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            timings = [json.loads(line) for line in (output / "timings.jsonl").read_text().splitlines()]
            row.update({"tasks": manifest["forecast_tasks"], "load_seconds": manifest["load_seconds"],
                        "peak_gpu_allocated_mb": manifest["peak_gpu_memory_mb"],
                        "peak_rss_mb": manifest["peak_process_rss_mb"],
                        "task_seconds": [t["inference_seconds"] for t in timings],
                        "target_artifact_accessed": manifest["target_artifact_accessed"]})
        _append(args.output_root / "benchmark.jsonl", row)
        if completed.returncode != 0:
            raise SystemExit(f"{model} {fold} rep{rep} failed, see {output / 'node.log'}")


def _baseline_task(job: tuple) -> dict:
    model, params, threads, task = job
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(REPO / "track1/code"))
    from models import forecast
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    result = forecast(model, task.history, 28, 42, config=params)
    wall = time.perf_counter() - wall_start
    return {"task_id": task.task_id, "market": task.market, "wall_seconds": wall,
            "cpu_seconds": time.process_time() - cpu_start, "allocated_vcpu_seconds": wall * threads,
            "fit_seconds": result.fit_seconds, "inference_seconds": result.inference_seconds,
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def run_cpu(model: str, args, plan: dict) -> None:
    sys.path.insert(0, str(REPO / "track1/code"))
    from data_pipeline import read_task_inputs
    frozen = json.loads(FROZEN_BASELINE.read_text(encoding="utf-8"))["models"][model]["candidate_id"]
    search = json.loads(BASELINE_SEARCH.read_text(encoding="utf-8"))
    params = next(c["params"] for c in search["models"][model]["candidates"] if c["id"] == frozen)
    threads = BASELINE_THREADS.get(model, 1)
    tasks = [t for t in read_task_inputs(INPUTS)
             if t.split_name == "outer_backtest" and t.history_length_label == "all"]
    done = set()
    log = args.output_root / "benchmark.jsonl"
    if log.is_file():
        done = {(r["model"], r["fold_id"], r["repeat"]) for r in map(json.loads, log.read_text().splitlines())
                if r["kind"] == "cpu"}
    for fold, rep, warmup in _schedule(plan):
        if (model, fold, rep) in done:
            continue
        jobs = [(model, params, threads, t) for t in tasks if t.fold_id == fold]
        if len(jobs) != TASKS_PER_ORIGIN:
            raise SystemExit(f"{fold}: expected {TASKS_PER_ORIGIN} tasks, got {len(jobs)}")
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=TASKS_PER_ORIGIN) as pool:
            results = list(pool.map(_baseline_task, jobs))
        _append(log, {"kind": "cpu", "model": model, "candidate_id": frozen, "fold_id": fold, "repeat": rep,
                      "warmup": warmup, "threads_per_task": threads, "concurrent_tasks": TASKS_PER_ORIGIN,
                      "origin_wall_seconds": time.perf_counter() - started, "tasks": len(results),
                      "task_results": results, "at_utc": _now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["gpu", "cpu"], required=True)
    parser.add_argument("--models", required=True, help="Comma-separated, run in order")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--shards", type=Path, default=Path("/root/DYQ/origin_shards"))
    parser.add_argument("--registry", type=Path, default=Path("/root/DYQ/run-logs/upstream_revisions_runtime.yaml"))
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/compute_benchmark")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan["status"] != "registered_before_analysis":
        raise SystemExit("volume_breakeven_plan.yaml is not registered")
    for model in (m.strip() for m in args.models.split(",") if m.strip()):
        (run_gpu if args.kind == "gpu" else run_cpu)(model, args, plan)


if __name__ == "__main__":
    main()
