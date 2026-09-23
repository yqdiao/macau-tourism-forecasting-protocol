"""Create and verify physically isolated, single-origin model inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

FORBIDDEN_COLUMNS = {"actual", "target", "target_value", "y_true"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def validate_single_origin_input(path: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path).resolve()
    manifest_path = path.parent / "origin_shard_manifest.json"
    if path.name != "task_inputs.parquet" or not manifest_path.is_file():
        raise ValueError("predictor requires a task_inputs.parquet beside origin_shard_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contains_targets") is not False:
        raise ValueError("origin shard manifest must declare contains_targets=false")
    if manifest.get("task_inputs_sha256") != file_sha256(path):
        raise ValueError("origin shard SHA256 does not match its manifest")
    frame = pd.read_parquet(path, engine="pyarrow")
    forbidden = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"model input exposes target columns: {sorted(forbidden)}")
    required = {"task_id", "origin_date", "history_end", "history", "history_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"origin shard is missing columns: {sorted(missing)}")
    origins = pd.to_datetime(frame["origin_date"]).dt.normalize().unique()
    if len(origins) != 1:
        raise ValueError("one predictor process may read exactly one forecast origin")
    if not (pd.to_datetime(frame["history_end"]) <= origins[0]).all():
        raise ValueError("origin shard contains history later than its forecast origin")
    if manifest.get("origin_date") != pd.Timestamp(origins[0]).date().isoformat():
        raise ValueError("origin date differs between shard and manifest")
    if int(manifest.get("task_count", -1)) != len(frame):
        raise ValueError("origin shard task count differs from its manifest")
    return frame, manifest
