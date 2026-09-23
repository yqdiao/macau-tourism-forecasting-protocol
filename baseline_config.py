"""Validate the registered baseline search and the later B-selected freeze file.

Configuration files use JSON syntax, which is a subset of YAML 1.2. This keeps
the audit files readable without adding a parser dependency to the pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from protocol_config import DEFAULT_PROTOCOL, load_protocol

HERE = Path(__file__).resolve().parent
SEARCH_PATH = HERE / "configs" / "baseline_search_spaces.yaml"
FREEZE_PATH = HERE / "configs" / "frozen_baseline_configs.yaml"
MODEL_NAMES = (
    "seasonal_naive", "ets", "sarima", "ridge", "linear_regression",
    "random_forest", "gradient_boosting", "svr", "lasso", "elasticnet",
    "parrot_random_forest_800", "lightgbm", "transformer",
)


def _read(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate(search_path: Path = SEARCH_PATH, freeze_path: Path = FREEZE_PATH,
             protocol_path: Path = DEFAULT_PROTOCOL, task_manifest_path: Path | None = None) -> dict:
    protocol = load_protocol(protocol_path)
    search, search_hash = _read(search_path)
    frozen, freeze_hash = _read(freeze_path)
    if search.get("schema_version") != 1 or frozen.get("schema_version") != 1:
        raise ValueError("baseline config schema_version must be 1")
    if search.get("tuning_split") != "recovery_tuning" or search.get("outer_split") != "outer_backtest":
        raise ValueError("Baseline selection must use B, not outer targets")
    if search.get("history_setting") != "all" or search.get("selection_metric") != protocol_metric():
        raise ValueError("Baseline search metric or history does not match the protocol")
    expected_b_ids = [f"B{i:02d}" for i in range(1, 13)]
    if search.get("fold_ids") != expected_b_ids:
        raise ValueError("Baseline search must use exactly B01-B12")
    if not search.get("seeds") or len(set(search["seeds"])) != len(search["seeds"]):
        raise ValueError("Baseline seeds must be fixed and unique")
    if set(search["models"]) != set(MODEL_NAMES) or set(frozen["models"]) != set(MODEL_NAMES):
        raise ValueError("Both baseline configs must list all 13 models")
    candidate_lookup = {}
    for model, spec in search["models"].items():
        candidates = spec.get("candidates", [])
        if not candidates or spec.get("max_trials") != len(candidates):
            raise ValueError(f"{model}: every finite candidate must fit the trial budget")
        ids = [candidate["id"] for candidate in candidates]
        if len(set(ids)) != len(ids) or any(not isinstance(c["params"], dict) for c in candidates):
            raise ValueError(f"{model}: duplicate id or invalid parameters")
        candidate_lookup[model] = set(ids)
    if frozen.get("selection_metric") != search["selection_metric"]:
        raise ValueError("Freeze metric differs from the registered search metric")
    if task_manifest_path is not None:
        task_manifest = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
        if task_manifest.get("protocol_sha256") != protocol.sha256:
            raise ValueError("Task manifest protocol SHA differs from the selected protocol")
    if frozen.get("status") == "pending_B_group_tuning":
        if frozen.get("selected_after_B_only") is not False:
            raise ValueError("Pending freeze cannot claim completed selection")
        for model, entry in frozen["models"].items():
            expected_id = "SN01" if model == "seasonal_naive" else None
            expected_status = "fixed_without_tuning" if model == "seasonal_naive" else "pending"
            if entry.get("candidate_id") != expected_id or entry.get("status") != expected_status:
                raise ValueError(f"{model}: pending freeze file claims an unvalidated winner")
    elif frozen.get("status") == "frozen_after_B_group_tuning":
        if task_manifest_path is None:
            raise ValueError("A completed freeze requires the audited task manifest")
        required = {
            "source_search_space_sha256": search_hash,
            "source_protocol_sha256": protocol.sha256,
            "B_task_list_sha256": task_manifest["B_main_task_list_sha256"],
        }
        for key, expected in required.items():
            if frozen.get(key) != expected:
                raise ValueError(f"Frozen {key} differs from its source")
        if frozen.get("selected_after_B_only") is not True or not frozen.get("frozen_at_utc") or not frozen.get("model_code_commit"):
            raise ValueError("Completed freeze is missing its audit provenance")
        for model, entry in frozen["models"].items():
            if entry.get("candidate_id") not in candidate_lookup[model]:
                raise ValueError(f"{model}: selected candidate is outside the registered search")
            if model != "seasonal_naive":
                score = entry.get("validation_score")
                if entry.get("status") != "selected_on_B" or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"{model}: missing B validation evidence")
    else:
        raise ValueError("Unknown baseline freeze status")
    return {
        "search_sha256": search_hash,
        "freeze_sha256": freeze_hash,
        "protocol_sha256": protocol.sha256,
        "freeze_status": frozen["status"],
        "model_count": len(MODEL_NAMES),
        "candidate_count": sum(len(v["candidates"]) for v in search["models"].values()),
    }


def protocol_metric() -> str:
    return "task_MASE7_market_fold_macro"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline search and freeze records")
    parser.add_argument("--search", type=Path, default=SEARCH_PATH)
    parser.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task-manifest", type=Path, default=HERE / "data" / "task_manifest.json")
    args = parser.parse_args()
    print(json.dumps(validate(args.search, args.freeze, args.protocol, args.task_manifest), indent=2))


if __name__ == "__main__":
    main()
