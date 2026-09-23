"""Split task inputs into immutable per-origin artifacts; targets are never read."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from foundation_models.origin_io import FORBIDDEN_COLUMNS, file_sha256, write_json_atomic


def build(inputs: Path, output: Path) -> dict:
    inputs = Path(inputs).resolve()
    if inputs.name != "task_inputs.parquet" or not inputs.is_file():
        raise ValueError("--inputs must be an existing task_inputs.parquet")
    frame = pd.read_parquet(inputs, engine="pyarrow")
    forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"source input exposes target columns: {sorted(forbidden)}")
    required = {"task_id", "split_name", "fold_id", "origin_date", "history_end", "history"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source input is missing columns: {sorted(missing)}")
    if frame["task_id"].duplicated().any():
        raise ValueError("source input contains duplicate task IDs")

    source_sha = file_sha256(inputs)
    records = []
    group_columns = ["split_name", "fold_id", "origin_date"]
    for (split_name, fold_id, origin_value), group in frame.groupby(group_columns, sort=True):
        origin = pd.Timestamp(origin_value).date().isoformat()
        if not (pd.to_datetime(group["history_end"]).dt.date <= pd.Timestamp(origin).date()).all():
            raise ValueError(f"{split_name}/{fold_id}: history exceeds forecast origin")
        folder = output / str(split_name) / f"{fold_id}_{origin}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "task_inputs.parquet"
        temporary = folder / ".task_inputs.parquet.tmp"
        group.reset_index(drop=True).to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
        manifest = {
            "schema_version": 1,
            "kind": "single_origin_model_input",
            "split_name": str(split_name),
            "fold_id": str(fold_id),
            "origin_date": origin,
            "task_count": len(group),
            "contains_targets": False,
            "source_task_inputs_sha256": source_sha,
            "task_inputs_sha256": file_sha256(path),
        }
        write_json_atomic(manifest, folder / "origin_shard_manifest.json")
        records.append({"relative_path": path.relative_to(output).as_posix(), **manifest})

    index = {
        "schema_version": 1,
        "source_task_inputs_sha256": source_sha,
        "shard_count": len(records),
        "task_count": sum(item["task_count"] for item in records),
        "contains_targets": False,
        "shards": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(index, output / "origin_shards_manifest.json")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/origin_shards"))
    args = parser.parse_args()
    print(json.dumps(build(args.inputs, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
