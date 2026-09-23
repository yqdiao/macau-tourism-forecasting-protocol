"""Run one model container once per immutable origin shard with resumable outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

MODELS = ("timesfm_3", "chronos_2", "lag_llama", "moirai",
          "granite_patchtst_fm", "patchtst", "informer")
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--shards", type=Path, default=REPO_ROOT / "shared" / "data" / "origin_shards")
    parser.add_argument("--splits", default="recovery_tuning")
    parser.add_argument("--history", default="all")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / "track1" / ".env.pinned")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "track1" / "results" / "foundation")
    args = parser.parse_args()
    index = json.loads((args.shards / "origin_shards_manifest.json").read_text(encoding="utf-8"))
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    for shard in index["shards"]:
        if shard["split_name"] not in selected_splits:
            continue
        shard_dir = (args.shards / shard["relative_path"]).parent.resolve()
        subdir = f"{shard['split_name']}/{shard['fold_id']}_{shard['origin_date']}"
        done = args.output_root / subdir / args.model / "run_manifest.json"
        if done.is_file():
            print(f"skip completed {subdir}/{args.model}", flush=True)
            continue
        environment = os.environ.copy()
        environment.update({
            "ORIGIN_SHARD_DIR": str(shard_dir),
            "RUN_OUTPUT_SUBDIR": subdir,
            "HISTORY_LENGTHS": args.history,
            "FOUNDATION_OUTPUT_ROOT": str(args.output_root.resolve()),
        })
        subprocess.run([
            "docker", "compose", "--env-file", str(args.env_file),
            "-f", str(REPO_ROOT / "track1" / "compose" / "foundation-gpu.yaml"),
            "run", "--rm", args.model,
        ], check=True, env=environment, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
