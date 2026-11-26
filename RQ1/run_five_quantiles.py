"""Re-run the five foundation models with quantiles 0.1, 0.25, 0.5, 0.75, 0.9.

Same frozen weights, interpreters, shards, cache and seed as the original
outer and pandemic runs; only the requested quantile levels differ.
Output: RQ1/raw/five_quantile_runs/<split>/<model>/<fold>/predictions.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
SHARDS = Path("/root/DYQ/origin_shards")
REGISTRY = REPO / "RQ1/raw/registry_five_quantiles.yaml"
OUT = REPO / "RQ1/raw/five_quantile_runs"
MODELS = {
    "granite_patchtst_fm": ("/root/DYQ/granite-cu128/bin/python", 1),
    "chronos_2": ("/root/DYQ/chronos-cu128/bin/python", 8),
    "timesfm_3": ("/root/DYQ/timesfm-cu128/bin/python", 8),
    "moirai": ("/root/DYQ/moirai-cu128/bin/python", 1),
    "lag_llama": ("/root/DYQ/cu121-py311/bin/python", 1),
}
# All five foundation models now run on pandemic_stress as well as outer_backtest.
STRESS_MODELS = {"granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--splits", default="outer_backtest,pandemic_stress",
                        help="Comma-separated split_name values to run")
    args = parser.parse_args()
    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    manifest = json.loads((SHARDS / "origin_shards_manifest.json").read_text())
    jobs = []
    for model in [m for m in args.models.split(",") if m]:
        for shard in manifest["shards"]:
            split = shard["split_name"]
            if split not in wanted_splits:
                continue
            if split == "outer_backtest" or (split == "pandemic_stress" and model in STRESS_MODELS):
                jobs.append((model, shard))
    for done, (model, shard) in enumerate(jobs, 1):
        python, batch = MODELS[model]
        output = OUT / shard["split_name"] / model / shard["fold_id"]
        if (output / "run_manifest.json").is_file():
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [python, str(REPO / "track1/code/run_foundation_backtest.py"), "--model", model,
                   "--inputs", str(SHARDS / shard["relative_path"]), "--output", str(output),
                   "--registry", str(REGISTRY), "--batch-size", str(batch), "--history", "all", "--seed", "42"]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "HF_HOME": f"/root/DYQ/foundation-cache/{model}", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
        started = time.perf_counter()
        with (output / "node.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        print(json.dumps({"job": f"{done}/{len(jobs)}", "model": model, "split": shard["split_name"],
                          "fold": shard["fold_id"], "returncode": result.returncode,
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        if result.returncode != 0:
            raise SystemExit(f"{model} {shard['fold_id']} failed; see {output / 'node.log'}")


if __name__ == "__main__":
    main()
