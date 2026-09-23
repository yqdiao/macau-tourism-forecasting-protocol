"""Score sealed B predictions with protected targets; freeze only after full coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK1_ROOT = REPO_ROOT / "track1"
LOCAL_DEPS = REPO_ROOT / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from baseline_config import FREEZE_PATH, SEARCH_PATH, validate
from protocol_config import DEFAULT_PROTOCOL


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _terminal_failure(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("attempts", 0) >= 2


def score(*, predictions_dir: Path, targets_path: Path, task_manifest_path: Path,
          output: Path, search_path: Path = SEARCH_PATH, freeze_path: Path = FREEZE_PATH,
          freeze: bool = False) -> dict:
    predictions_dir, targets_path, output = map(Path, (predictions_dir, targets_path, output))
    search = json.loads(Path(search_path).read_text(encoding="utf-8"))
    task_manifest = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
    run_manifest = json.loads((predictions_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest["input_sha256"] != task_manifest["artifacts"]["splits/recovery_tuning/task_inputs.parquet"]["sha256"]:
        raise ValueError("Predictor inputs differ from the audited task artifact")
    if run_manifest["search_sha256"] != _sha(search_path) or run_manifest["protocol_sha256"] != _sha(DEFAULT_PROTOCOL):
        raise ValueError("Predictor used another registered search or protocol")
    if _sha(targets_path) != task_manifest["artifacts"]["targets.parquet"]["sha256"]:
        raise ValueError("Protected target SHA256 differs from the task manifest")
    targets = pd.read_parquet(targets_path)
    targets = targets.loc[(targets["split_name"] == "recovery_tuning") &
                          (targets["history_length"] == "all")]
    if targets["task_id"].nunique() != 324 or len(targets) != 324 * 28:
        raise ValueError("Protected B target set is incomplete")
    target_by_task = {task_id: group.set_index("horizon_day")
                      for task_id, group in targets.groupby("task_id")}
    scores: list[dict] = []
    candidate_rows: list[dict] = []
    for model, specification in search["models"].items():
        seeds = search["seeds"] if model in search["multi_seed_models"] else [search["single_seed"]]
        for candidate in specification["candidates"]:
            candidate_id = candidate["id"]
            seen: set[tuple[str, int]] = set()
            for seed in seeds:
                directory = predictions_dir / "predictions" / model / candidate_id / str(seed)
                for shard in sorted(directory.glob("*.parquet")):
                    prediction = pd.read_parquet(shard)
                    task_id = str(prediction["task_id"].iloc[0])
                    identity = (task_id, seed)
                    if task_id not in target_by_task or identity in seen:
                        raise ValueError(f"Unexpected or duplicate B prediction shard: {shard}")
                    seen.add(identity)
                    if len(prediction) != 28 or prediction["horizon_day"].tolist() != list(range(1, 29)):
                        raise ValueError(f"Incomplete 28-day path: {shard}")
                    if (prediction["model"].nunique() != 1 or prediction["model"].iloc[0] != model or
                            prediction["candidate_id"].nunique() != 1 or prediction["candidate_id"].iloc[0] != candidate_id or
                            prediction["seed"].nunique() != 1 or int(prediction["seed"].iloc[0]) != seed or
                            prediction["split_name"].nunique() != 1 or prediction["split_name"].iloc[0] != "recovery_tuning"):
                        raise ValueError(f"Prediction metadata mismatch: {shard}")
                    actual = target_by_task[task_id].loc[list(range(1, 29))]
                    if not np.array_equal(pd.to_datetime(prediction["target_date"]),
                                          pd.to_datetime(actual["target_date"])):
                        raise ValueError(f"Prediction target dates differ from protected targets: {shard}")
                    if (prediction["fold_id"].iloc[0] != actual["fold_id"].iloc[0] or
                            prediction["market"].iloc[0] != actual["market"].iloc[0]):
                        raise ValueError(f"Prediction fold/market differs from protected targets: {shard}")
                    values = prediction["prediction"].to_numpy(float)
                    raw = prediction["prediction_raw"].to_numpy(float)
                    if not np.isfinite(values).all() or not np.isfinite(raw).all():
                        raise ValueError(f"Non-finite forecast: {shard}")
                    mae = float(np.mean(np.abs(values - actual["actual"].to_numpy(float))))
                    scale = float(prediction["mase7_scale"].iloc[0])
                    scores.append({
                        "model": model, "candidate_id": candidate_id, "seed": seed,
                        "task_id": task_id, "fold_id": actual["fold_id"].iloc[0],
                        "market": actual["market"].iloc[0], "origin_date": actual["origin_date"].iloc[0],
                        "MAE": mae, "MASE7": mae / scale if math.isfinite(scale) and scale > 0 else None,
                        "undefined_scale": not math.isfinite(scale) or scale <= 0,
                        "fit_seconds": float(prediction["fit_seconds"].iloc[0]),
                        "inference_seconds": float(prediction["inference_seconds"].iloc[0]),
                        "total_seconds": float(prediction["total_seconds"].iloc[0]),
                        "prediction_sha256": _sha(shard),
                    })
            candidate_rows.append({
                "model": model, "candidate_id": candidate_id,
                "expected_paths": 324 * len(seeds), "completed_paths": len(seen),
                "missing_paths": 324 * len(seeds) - len(seen),
                "terminal_failure_paths": sum(
                    1 for seed in seeds for task_id in target_by_task
                    if (task_id, seed) not in seen
                    and _terminal_failure(predictions_dir / "failures" / model / candidate_id / str(seed) / f"{task_id}.json")
                ),
            })
    task_scores = pd.DataFrame(scores)
    summary = pd.DataFrame(candidate_rows)
    for index, candidate in summary.iterrows():
        subset = task_scores.loc[(task_scores["model"] == candidate["model"]) &
                                 (task_scores["candidate_id"] == candidate["candidate_id"])] if not task_scores.empty else pd.DataFrame()
        if subset.empty:
            summary.loc[index, "mean_MASE7"] = float("nan")
            summary.loc[index, "mean_fit_seconds"] = float("nan")
            summary.loc[index, "undefined_scale_paths"] = 0
            continue
        # First average seeds within task, then average folds within market and markets equally.
        per_task = subset.groupby(["market", "fold_id", "task_id"], as_index=False)["MASE7"].mean()
        per_market = per_task.groupby("market")["MASE7"].mean()
        summary.loc[index, "mean_MASE7"] = float(per_market.mean())
        summary.loc[index, "mean_fit_seconds"] = float(subset["fit_seconds"].mean())
        summary.loc[index, "undefined_scale_paths"] = int(subset["undefined_scale"].sum())
    summary["unresolved_paths"] = summary["missing_paths"] - summary["terminal_failure_paths"]
    summary["eligible"] = (summary["missing_paths"] == 0) & (summary["undefined_scale_paths"] == 0)
    output.mkdir(parents=True, exist_ok=True)
    task_path, summary_path = output / "task_metrics.parquet", output / "candidate_summary.parquet"
    for frame, path in ((task_scores, task_path), (summary, summary_path)):
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    all_jobs_resolved = bool(summary["unresolved_paths"].eq(0).all())
    every_model_has_candidate = all(
        bool(summary.loc[summary["model"] == model, "eligible"].any())
        for model in search["models"]
    )
    complete = all_jobs_resolved and every_model_has_candidate
    status = {
        "complete": complete, "candidate_count": len(summary),
        "all_jobs_resolved": all_jobs_resolved,
        "every_model_has_eligible_candidate": every_model_has_candidate,
        "completed_paths": int(summary["completed_paths"].sum()),
        "expected_paths": int(summary["expected_paths"].sum()),
        "terminal_failure_paths": int(summary["terminal_failure_paths"].sum()),
        "unresolved_paths": int(summary["unresolved_paths"].sum()),
        "incomplete_candidates": summary.loc[~summary["eligible"],
            ["model", "candidate_id", "missing_paths", "terminal_failure_paths", "unresolved_paths"]].to_dict("records"),
        "run_code_commit": run_manifest["code_commit"],
        "run_code_dirty": run_manifest["code_dirty"],
        "task_metrics_sha256": _sha(task_path),
        "candidate_summary_sha256": _sha(summary_path),
        "protected_targets_sha256": _sha(targets_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(status, output / "validation_status.json")
    if freeze:
        if not complete or run_manifest["code_dirty"] or run_manifest["task_limit"] is not None:
            raise ValueError("Cannot freeze: all candidates, all 324 B tasks, clean committed code required")
        existing = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
        if existing["status"] != "pending_B_group_tuning":
            raise ValueError("Refusing to overwrite an existing frozen decision")
        selected = {}
        for model in search["models"]:
            group = summary.loc[(summary["model"] == model) & summary["eligible"]].sort_values(
                ["mean_MASE7", "mean_fit_seconds", "candidate_id"], kind="stable")
            winner = group.iloc[0]
            selected[model] = {
                "candidate_id": winner["candidate_id"],
                "status": "fixed_without_tuning" if model == "seasonal_naive" else "selected_on_B",
                "validation_score": None if model == "seasonal_naive" else float(winner["mean_MASE7"]),
                "mean_fit_seconds": float(winner["mean_fit_seconds"]),
            }
        frozen = {
            "schema_version": 1, "status": "frozen_after_B_group_tuning",
            "source_search_space_sha256": _sha(search_path),
            "source_protocol_sha256": _sha(DEFAULT_PROTOCOL),
            "B_task_list_sha256": task_manifest["B_main_task_list_sha256"],
            "selection_metric": search["selection_metric"],
            "selected_after_B_only": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_code_commit": run_manifest["code_commit"],
            "validation_summary_sha256": _sha(summary_path),
            "models": selected,
        }
        _atomic_json(frozen, freeze_path)
        validate(search_path=search_path, freeze_path=freeze_path,
                 task_manifest_path=task_manifest_path)
        status["frozen_config_sha256"] = _sha(freeze_path)
        _atomic_json(status, output / "validation_status.json")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Score B candidates with protected targets and optionally freeze")
    parser.add_argument("--predictions", type=Path, default=TRACK1_ROOT / "results" / "b_validation")
    parser.add_argument("--targets", type=Path, default=REPO_ROOT / "shared" / "data" / "targets.parquet")
    parser.add_argument("--task-manifest", type=Path, default=REPO_ROOT / "shared" / "data" / "task_manifest.json")
    parser.add_argument("--output", type=Path, default=TRACK1_ROOT / "evaluation" / "b_validation")
    parser.add_argument("--search", type=Path, default=SEARCH_PATH)
    parser.add_argument("--freeze-path", type=Path, default=FREEZE_PATH)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    status = score(predictions_dir=args.predictions, targets_path=args.targets,
                   task_manifest_path=args.task_manifest, output=args.output,
                   search_path=args.search, freeze_path=args.freeze_path, freeze=args.freeze)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
