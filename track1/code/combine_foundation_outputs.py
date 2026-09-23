"""Combine atomic per-origin predictions after verifying each run manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from foundation_models.origin_io import file_sha256, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "foundation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames, sources = [], []
    for manifest_path in sorted(args.root.glob(f"**/{args.model}/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = manifest_path.parent / "predictions.parquet"
        expected = manifest["outputs"]["predictions.parquet"]
        if file_sha256(prediction_path) != expected:
            raise ValueError(f"prediction SHA mismatch: {prediction_path}")
        if manifest["model"] != args.model or manifest["target_artifact_accessed"] is not False:
            raise ValueError(f"invalid run manifest: {manifest_path}")
        frames.append(pd.read_parquet(prediction_path, engine="pyarrow"))
        sources.append({"manifest": manifest_path.as_posix(), "prediction_sha256": expected})
    if not frames:
        raise ValueError("no completed model runs found")
    combined = pd.concat(frames, ignore_index=True)
    key = ["task_id", "model", "seed", "target_date", "horizon_day"]
    if combined.duplicated(key).any():
        raise ValueError("combined outputs contain duplicate forecast rows")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "predictions.parquet"
    temporary = args.output / ".predictions.parquet.tmp"
    combined.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)
    write_json_atomic({
        "schema_version": 1, "model": args.model,
        "prediction_rows": len(combined), "source_run_count": len(sources),
        "sources": sources, "predictions_sha256": file_sha256(path),
        "target_artifact_accessed": False,
    }, args.output / "combined_manifest.json")


if __name__ == "__main__":
    main()
