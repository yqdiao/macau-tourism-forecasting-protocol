"""Score the PatchTST/Informer B01-B12 candidate search and optionally freeze.

Only recovery_tuning targets are read. Aggregation matches evaluate_b_validation.py:
average seeds within task, folds within market, then markets equally. The
selection rule is lowest mean MASE7, then lower mean fit time, then candidate ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_PATH = REPO_ROOT / "track1/configs/foundation_search_spaces.yaml"
FREEZE_PATH = REPO_ROOT / "track1/configs/frozen_foundation_configs.yaml"
B_TASKS = 324  # 27 Macau markets x 12 folds, history=all
RUN_CODE_PATHS = ["track1/code/foundation_models", "track1/code/run_foundation_backtest.py",
                  "track1/code/data_pipeline.py", "track1/scripts/run_foundation_b_search.py"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _registration_commit() -> str:
    return _git("log", "-1", "--diff-filter=A", "--format=%H", "--", str(SEARCH_PATH.relative_to(REPO_ROOT)))


def _run_code_unchanged(since: str) -> bool:
    committed = _git("diff", "--name-only", since, "HEAD", "--", *RUN_CODE_PATHS)
    working = _git("status", "--porcelain", "--", *RUN_CODE_PATHS)
    return not committed and not working


def score(*, predictions_root: Path, targets_path: Path, task_manifest_path: Path,
          output: Path, freeze: bool) -> dict:
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    if search["status"] != "registered_before_tuning":
        raise ValueError("search space is not registered")
    search_sha = _sha(SEARCH_PATH)
    task_manifest = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
    if _sha(targets_path) != task_manifest["artifacts"]["targets.parquet"]["sha256"]:
        raise ValueError("Protected target SHA256 differs from the task manifest")
    targets = pd.read_parquet(targets_path)
    targets = targets.loc[(targets["split_name"] == "recovery_tuning") &
                          (targets["history_length"] == "all")]
    if targets["task_id"].nunique() != B_TASKS or len(targets) != B_TASKS * 28:
        raise ValueError("Protected B target set is incomplete")
    target_by_task = {task_id: group.set_index("horizon_day").sort_index()
                      for task_id, group in targets.groupby("task_id")}

    scores, candidate_rows = [], []
    seeds = list(search["seeds"])
    for model, specification in search["models"].items():
        for candidate in specification["candidates"]:
            candidate_id = candidate["id"]
            seen: set[tuple[str, int]] = set()
            for seed in seeds:
                for fold_dir in sorted((predictions_root / model / candidate_id / f"seed{seed}").glob("B*_*")):
                    manifest_path = fold_dir / "run_manifest.json"
                    if not manifest_path.is_file():
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (manifest.get("candidate_id") != candidate_id or manifest.get("seed") != seed
                            or manifest.get("search_space_sha256") != search_sha
                            or manifest.get("target_artifact_accessed") is not False):
                        raise ValueError(f"Manifest mismatch: {manifest_path}")
                    prediction_path = fold_dir / "predictions.parquet"
                    if _sha(prediction_path) != manifest["outputs"]["predictions.parquet"]:
                        raise ValueError(f"Prediction file changed after sealing: {prediction_path}")
                    predictions = pd.read_parquet(prediction_path)
                    if set(predictions["split_name"]) != {"recovery_tuning"}:
                        raise ValueError(f"Non-B predictions in {prediction_path}")
                    fit_seconds = {}
                    for line in (fold_dir / "timings.jsonl").read_text(encoding="utf-8").splitlines():
                        row = json.loads(line)
                        fit_seconds[row["task_id"]] = float(row["inference_seconds"])
                    for task_id, prediction in predictions.groupby("task_id"):
                        prediction = prediction.sort_values("horizon_day")
                        if task_id not in target_by_task or (task_id, seed) in seen:
                            raise ValueError(f"Unexpected or duplicate B task {task_id} in {fold_dir}")
                        seen.add((task_id, seed))
                        if prediction["horizon_day"].tolist() != list(range(1, 29)):
                            raise ValueError(f"Incomplete 28-day path: {task_id} in {fold_dir}")
                        actual = target_by_task[task_id].loc[list(range(1, 29))]
                        if not np.array_equal(pd.to_datetime(prediction["target_date"]).to_numpy(),
                                              pd.to_datetime(actual["target_date"]).to_numpy()):
                            raise ValueError(f"Target dates differ: {task_id} in {fold_dir}")
                        values = prediction["prediction"].to_numpy(float)
                        if not np.isfinite(values).all() or not np.isfinite(prediction["prediction_raw"]).all():
                            raise ValueError(f"Non-finite forecast: {task_id} in {fold_dir}")
                        mae = float(np.mean(np.abs(values - actual["actual"].to_numpy(float))))
                        scale = float(prediction["mase7_scale"].iloc[0])
                        defined = math.isfinite(scale) and scale > 0
                        scores.append({
                            "model": model, "candidate_id": candidate_id, "seed": seed,
                            "task_id": task_id, "fold_id": actual["fold_id"].iloc[0],
                            "market": actual["market"].iloc[0], "MAE": mae,
                            "MASE7": mae / scale if defined else None, "undefined_scale": not defined,
                            "fit_seconds": fit_seconds[task_id],
                            "prediction_sha256": manifest["outputs"]["predictions.parquet"],
                        })
            candidate_rows.append({"model": model, "candidate_id": candidate_id,
                                   "expected_paths": B_TASKS * len(seeds), "completed_paths": len(seen),
                                   "missing_paths": B_TASKS * len(seeds) - len(seen)})

    task_scores = pd.DataFrame(scores)
    summary = pd.DataFrame(candidate_rows)
    for index, candidate in summary.iterrows():
        subset = task_scores.loc[(task_scores["model"] == candidate["model"]) &
                                 (task_scores["candidate_id"] == candidate["candidate_id"])] \
            if not task_scores.empty else pd.DataFrame()
        if subset.empty:
            summary.loc[index, ["mean_MASE7", "mean_fit_seconds", "seed_sd_MASE7"]] = float("nan")
            summary.loc[index, "undefined_scale_paths"] = 0
            continue
        per_task = subset.groupby(["market", "fold_id", "task_id"], as_index=False)["MASE7"].mean()
        summary.loc[index, "mean_MASE7"] = float(per_task.groupby("market")["MASE7"].mean().mean())
        per_seed = [float(s.groupby("market")["MASE7"].mean().mean()) for _, s in subset.groupby("seed")]
        for seed, value in zip(sorted(subset["seed"].unique()), per_seed):
            summary.loc[index, f"MASE7_seed{seed}"] = value
        summary.loc[index, "seed_sd_MASE7"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
        summary.loc[index, "mean_fit_seconds"] = float(subset["fit_seconds"].mean())
        summary.loc[index, "undefined_scale_paths"] = int(subset["undefined_scale"].sum())
    summary["eligible"] = (summary["missing_paths"] == 0) & (summary["undefined_scale_paths"] == 0)

    output.mkdir(parents=True, exist_ok=True)
    task_path, summary_path = output / "task_metrics.parquet", output / "candidate_summary.parquet"
    for frame, path in ((task_scores, task_path), (summary, summary_path)):
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    summary.to_csv(output / "candidate_summary.csv", index=False)

    registration = _registration_commit()
    code_unchanged = _run_code_unchanged(registration)
    complete = bool(summary["missing_paths"].eq(0).all())
    every_model = all(bool(summary.loc[summary["model"] == m, "eligible"].any()) for m in search["models"])
    status = {
        "complete": complete, "every_model_has_eligible_candidate": every_model,
        "candidate_count": len(summary),
        "completed_paths": int(summary["completed_paths"].sum()),
        "expected_paths": int(summary["expected_paths"].sum()),
        "incomplete_candidates": summary.loc[~summary["eligible"],
                                             ["model", "candidate_id", "missing_paths"]].to_dict("records"),
        "search_space_sha256": search_sha, "registration_commit": registration,
        "run_code_unchanged_since_registration": code_unchanged,
        "targets_read": "recovery_tuning, history=all only",
        "task_metrics_sha256": _sha(task_path), "candidate_summary_sha256": _sha(summary_path),
        "protected_targets_sha256": _sha(targets_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(status, output / "validation_status.json")

    if freeze:
        # The caller must only freeze after every runner exited, i.e. each job succeeded or
        # exhausted its retries; a partially covered candidate is then ineligible.
        if not (every_model and code_unchanged):
            raise ValueError("Cannot freeze: need a fully covered candidate per model and unchanged run code")
        if FREEZE_PATH.exists():
            raise ValueError(f"Refusing to overwrite existing {FREEZE_PATH}")
        selected = {}
        for model in search["models"]:
            group = summary.loc[(summary["model"] == model) & summary["eligible"]].sort_values(
                ["mean_MASE7", "mean_fit_seconds", "candidate_id"], kind="stable")
            winner = group.iloc[0]
            params = next(c["params"] for c in search["models"][model]["candidates"]
                          if c["id"] == winner["candidate_id"])
            selected[model] = {"candidate_id": winner["candidate_id"], "status": "selected_on_B",
                               "params": params, "validation_score": float(winner["mean_MASE7"]),
                               "seed_sd_MASE7": float(winner["seed_sd_MASE7"]),
                               "mean_fit_seconds": float(winner["mean_fit_seconds"])}
        for model, entry in search["zero_shot_models"]["models"].items():
            selected[model] = {"candidate_id": entry["id"], "status": "registered_default_without_tuning",
                               "params": entry["params"], "validation_score": None}
        frozen = {
            "schema_version": 1, "status": "frozen_after_B_group_tuning",
            "source_search_space_sha256": search_sha, "registration_commit": registration,
            "selection_metric": search["selection_metric"], "selection_rule": search["selection_rule"],
            "seeds": seeds, "selected_after_B_only": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_summary_sha256": _sha(summary_path), "models": selected,
        }
        _atomic_json(frozen, FREEZE_PATH)
        status["frozen_config_sha256"] = _sha(FREEZE_PATH)
        _atomic_json(status, output / "validation_status.json")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "track1/results/foundation_b_search")
    parser.add_argument("--targets", type=Path, default=REPO_ROOT / "shared/data/targets.parquet")
    parser.add_argument("--task-manifest", type=Path, default=REPO_ROOT / "shared/data/task_manifest.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "track1/evaluation/foundation_b_search")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    status = score(predictions_root=args.predictions, targets_path=args.targets,
                   task_manifest_path=args.task_manifest, output=args.output, freeze=args.freeze)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
