"""Score the five zero-shot foundation models on B01-B12 (registered default config).

Aggregation matches evaluate_foundation_b_search.py: task MASE7, mean over folds
within market, then markets equally. Only recovery_tuning, history=all targets are read.
Also writes a 7-model B ranking that adds the frozen PatchTST/Informer candidates.
This does not freeze the Track 2 top three.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
PRED_ROOT = REPO / "track1/results/foundation_b_default"
TARGETS = REPO / "shared/data/targets.parquet"
TASK_MANIFEST = REPO / "shared/data/task_manifest.json"
FROZEN = REPO / "track1/configs/frozen_foundation_configs.yaml"
SEARCH_SUMMARY = REPO / "track1/evaluation/foundation_b_search/candidate_summary.parquet"
OUTPUT = REPO / "track1/evaluation/foundation_b_zero_shot"
ZERO_SHOT = ["timesfm_3", "chronos_2", "moirai", "granite_patchtst_fm", "lag_llama"]
B_TASKS = 324


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    if _sha(TARGETS) != manifest["artifacts"]["targets.parquet"]["sha256"]:
        raise ValueError("protected target SHA256 differs from task manifest")
    targets = pd.read_parquet(TARGETS)
    targets = targets.loc[(targets["split_name"] == "recovery_tuning") & (targets["history_length"] == "all")]
    if targets["task_id"].nunique() != B_TASKS or len(targets) != B_TASKS * 28:
        raise ValueError("protected B target set is incomplete")
    target_by_task = {task_id: group.set_index("horizon_day").sort_index()
                      for task_id, group in targets.groupby("task_id")}
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    rows, coverage = [], []
    for model in ZERO_SHOT:
        if frozen["models"][model]["status"] != "registered_default_without_tuning":
            raise ValueError(f"{model} is not registered as an untuned default")
        seen: set[str] = set()
        for fold_dir in sorted(PRED_ROOT.glob(f"B*_*/{model}")):
            run_manifest = json.loads((fold_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if run_manifest["model"] != model or run_manifest["target_artifact_accessed"] is not False:
                raise ValueError(f"invalid manifest {fold_dir}")
            prediction_path = fold_dir / "predictions.parquet"
            if _sha(prediction_path) != run_manifest["outputs"]["predictions.parquet"]:
                raise ValueError(f"prediction changed after sealing: {prediction_path}")
            predictions = pd.read_parquet(prediction_path)
            if set(predictions["split_name"]) != {"recovery_tuning"} or set(predictions["history_length"]) != {"all"}:
                raise ValueError(f"unexpected split/history in {prediction_path}")
            for task_id, prediction in predictions.groupby("task_id"):
                prediction = prediction.sort_values("horizon_day")
                if task_id not in target_by_task or task_id in seen:
                    raise ValueError(f"unexpected or duplicate task {task_id}")
                seen.add(task_id)
                if prediction["horizon_day"].tolist() != list(range(1, 29)):
                    raise ValueError(f"incomplete 28-day path {task_id}")
                actual = target_by_task[task_id].loc[list(range(1, 29))]
                if not np.array_equal(pd.to_datetime(prediction["target_date"]).to_numpy(),
                                      pd.to_datetime(actual["target_date"]).to_numpy()):
                    raise ValueError(f"target dates differ {task_id}")
                values = prediction["prediction"].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite forecast {task_id}")
                mae = float(np.mean(np.abs(values - actual["actual"].to_numpy(float))))
                scale = float(prediction["mase7_scale"].iloc[0])
                defined = math.isfinite(scale) and scale > 0
                rows.append({"model": model, "candidate_id": frozen["models"][model]["candidate_id"],
                             "seed": int(prediction["seed"].iloc[0]), "task_id": task_id,
                             "fold_id": actual["fold_id"].iloc[0], "market": actual["market"].iloc[0],
                             "MAE": mae, "MASE7": mae / scale if defined else None,
                             "undefined_scale": not defined})
        coverage.append({"model": model, "completed_paths": len(seen), "expected_paths": B_TASKS})

    tasks = pd.DataFrame(rows)
    summary = []
    for item in coverage:
        subset = tasks.loc[tasks["model"] == item["model"]]
        market = subset.groupby("market")["MASE7"].mean()
        summary.append({**item, "candidate_id": subset["candidate_id"].iloc[0],
                        "status": "registered_default_without_tuning",
                        "mean_MASE7": float(market.mean()),
                        "median_task_MASE7": float(subset["MASE7"].median()),
                        "undefined_scale_paths": int(subset["undefined_scale"].sum()),
                        "eligible": item["completed_paths"] == B_TASKS and not subset["undefined_scale"].any()})
    summary = pd.DataFrame(summary)

    search = pd.read_parquet(SEARCH_SUMMARY)
    trained = []
    for model in ("patchtst", "informer"):
        entry = frozen["models"][model]
        hit = search.loc[(search["model"] == model) & (search["candidate_id"] == entry["candidate_id"])].iloc[0]
        trained.append({"model": model, "candidate_id": entry["candidate_id"], "status": "selected_on_B",
                        "completed_paths": int(hit["completed_paths"]), "expected_paths": int(hit["expected_paths"]),
                        "mean_MASE7": float(hit["mean_MASE7"]), "seed_sd_MASE7": float(hit["seed_sd_MASE7"]),
                        "eligible": bool(hit["eligible"])})
    ranking = pd.concat([summary, pd.DataFrame(trained)], ignore_index=True)
    ranking = ranking.sort_values(["mean_MASE7", "model"], kind="stable").reset_index(drop=True)
    ranking.insert(0, "B_rank", range(1, len(ranking) + 1))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(tasks, OUTPUT / "task_metrics.parquet")
    _atomic_parquet(summary, OUTPUT / "zero_shot_summary.parquet")
    _atomic_parquet(ranking, OUTPUT / "seven_model_B_ranking.parquet")
    ranking.to_csv(OUTPUT / "seven_model_B_ranking.csv", index=False)
    status = {
        "targets_read": "recovery_tuning, history=all only",
        "protected_targets_sha256": _sha(TARGETS),
        "frozen_foundation_config_sha256": _sha(FROZEN),
        "aggregation": "task MASE7 -> mean over folds within market -> equal-weight market mean",
        "complete": bool(summary["eligible"].all()),
        "track2_top3_frozen": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
