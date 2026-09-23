"""Audit protocol dates, leakage boundaries, and the generated task artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from data_pipeline import read_task_inputs
from protocol_config import DEFAULT_PROTOCOL, load_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder: Path, protocol_path: Path = DEFAULT_PROTOCOL) -> dict:
    folder = Path(folder)
    protocol = load_protocol(protocol_path)
    manifest = json.loads((folder / "task_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("Task manifest was generated with another protocol")
    for name in ("task_index.parquet", "task_inputs.parquet", "targets.parquet"):
        if _sha(folder / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"{name}: SHA256 mismatch")
    inputs = pd.read_parquet(folder / "task_inputs.parquet")
    targets = pd.read_parquet(folder / "targets.parquet")
    index = pd.read_parquet(folder / "task_index.parquet")
    if len(index) != len(inputs) or index["task_id"].tolist() != inputs["task_id"].tolist():
        raise ValueError("Task index is out of sync with model inputs")
    if "history" in index or "actual" in index:
        raise ValueError("Task index contains restricted values")
    if "actual" in inputs or "history" in targets:
        raise ValueError("Inputs and targets are not physically isolated")
    if len(inputs) != manifest["task_count"] or len(targets) != manifest["target_row_count"]:
        raise ValueError("Artifact row counts differ from manifest")
    if inputs["task_id"].duplicated().any():
        raise ValueError("Duplicate task_id")
    if set(inputs["task_id"]) != set(targets["task_id"]):
        raise ValueError("Model inputs and protected targets have different task IDs")
    expected_folds = {(f.split_name, f.fold_id): f for f in protocol.folds}
    split_counts: Counter[str] = Counter()
    index_lines = []
    b_lines = []
    for row in inputs.itertuples(index=False):
        fold = expected_folds.get((row.split_name, row.fold_id))
        origin = pd.Timestamp(row.origin_date).date()
        start = pd.Timestamp(row.history_start).date()
        if fold is None or origin != fold.origin or row.destination not in fold.destinations:
            raise ValueError(f"{row.task_id}: split/fold/origin/destination mismatch")
        if pd.Timestamp(row.history_end).date() != origin or start > origin:
            raise ValueError(f"{row.task_id}: invalid historical boundary")
        if (pd.Timestamp(row.target_start_date).date(), pd.Timestamp(row.target_end_date).date()) != (
            origin + timedelta(days=1), origin + timedelta(days=28)
        ):
            raise ValueError(f"{row.task_id}: invalid 28-day target dates")
        if len(row.history) != row.training_history_days or len(row.history) != (origin - start).days + 1:
            raise ValueError(f"{row.task_id}: historical array length mismatch")
        if row.history_length != "all" and len(row.history) != int(row.history_length):
            raise ValueError(f"{row.task_id}: sensitivity history length mismatch")
        split_counts[row.split_name] += 1
        line = f"{row.split_name}\t{row.fold_id}\t{row.task_id}"
        index_lines.append(line)
        if row.split_name == "recovery_tuning" and row.history_length == "all":
            b_lines.append(line)
    if dict(split_counts) != manifest["split_task_counts"]:
        raise ValueError("Split task counts differ from manifest")
    for split_name, expected_count in split_counts.items():
        relative = f"splits/{split_name}/task_inputs.parquet"
        split_path = folder / relative
        if _sha(split_path) != manifest["artifacts"][relative]["sha256"]:
            raise ValueError(f"{split_name}: split input SHA256 mismatch")
        split_inputs = pd.read_parquet(split_path)
        if len(split_inputs) != expected_count or set(split_inputs["split_name"]) != {split_name}:
            raise ValueError(f"{split_name}: split input membership mismatch")
        if split_inputs["task_id"].tolist() != inputs.loc[inputs["split_name"] == split_name, "task_id"].tolist():
            raise ValueError(f"{split_name}: split input differs from canonical inputs")
    if hashlib.sha256(("\n".join(index_lines) + "\n").encode()).hexdigest() != manifest["task_list_sha256"]:
        raise ValueError("Task list SHA256 mismatch")
    if hashlib.sha256(("\n".join(b_lines) + "\n").encode()).hexdigest() != manifest["B_main_task_list_sha256"]:
        raise ValueError("B main task list SHA256 mismatch")
    if len(list(read_task_inputs(folder / "task_inputs.parquet"))) != len(inputs):
        raise ValueError("Model-side loader did not accept every input task")
    target_counts = targets.groupby("task_id").size()
    if not target_counts.eq(28).all():
        raise ValueError("A target path is not 28 days")
    if not (pd.to_datetime(targets["target_date"]) ==
            pd.to_datetime(targets["origin_date"]) + pd.to_timedelta(targets["horizon_day"], unit="D")).all():
        raise ValueError("Target dates do not follow the horizon index")
    if not np.isfinite(targets["actual"].to_numpy(float)).all():
        raise ValueError("Non-finite actuals")
    main_counts = inputs.loc[inputs["history_length"] == "all"].groupby("split_name").size().to_dict()
    expected = {"pandemic_stress": 27 * 13, "recovery_tuning": 27 * 12,
                "outer_backtest": 29 * 33}
    if main_counts != expected:
        raise ValueError(f"Main-history sample differs from protocol: {main_counts}")
    return {
        "protocol_sha256": protocol.sha256,
        "task_list_sha256": manifest["task_list_sha256"],
        "input_sha256": manifest["artifacts"]["task_inputs.parquet"]["sha256"],
        "index_sha256": manifest["artifacts"]["task_index.parquet"]["sha256"],
        "target_sha256": manifest["artifacts"]["targets.parquet"]["sha256"],
        "tasks": len(inputs), "target_rows": len(targets),
        "main_history_tasks_by_split": main_counts,
        "checks": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit protocol task artifacts")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
