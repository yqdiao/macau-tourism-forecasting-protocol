"""Score the lightweight-model B01-B12 search and optionally freeze per the registered plan.

Reads recovery_tuning targets only. Aggregation: seed mean within task, fold mean
within market, equal-weight market mean. Rule: lowest MASE7, then lower mean fit
time, then candidate id. The light champion is the family winner with lower MASE7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "track1/configs/light_model_plan.yaml"
FREEZE = REPO / "track1/configs/frozen_light_configs.yaml"
CODE = REPO / "track1/code/light_models.py"
TARGETS = REPO / "shared/data/targets.parquet"
TASK_MANIFEST = REPO / "shared/data/task_manifest.json"
B_TASKS = 324
FAMILY_MODEL = {"global_lightgbm": "global_lightgbm", "global_linear": "global_linear"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write(temporary)
    os.replace(temporary, path)


def score(predictions_root: Path, output: Path, freeze: bool) -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan["status"] != "registered_before_tuning":
        raise ValueError("light model plan is not registered")
    plan_sha, code_sha = _sha(PLAN), _sha(CODE)
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    if _sha(TARGETS) != manifest["artifacts"]["targets.parquet"]["sha256"]:
        raise ValueError("protected target SHA256 differs from the task manifest")
    targets = pd.read_parquet(TARGETS)
    targets = targets.loc[(targets["split_name"] == "recovery_tuning") & (targets["history_length"] == "all")]
    if targets["task_id"].nunique() != B_TASKS or len(targets) != B_TASKS * 28:
        raise ValueError("protected B target set is incomplete")
    by_task = {task: group.set_index("horizon_day").sort_index() for task, group in targets.groupby("task_id")}

    scores, rows = [], []
    for family, spec in plan["families"].items():
        model = FAMILY_MODEL[family]
        seeds = spec["fixed"].get("seeds", [spec["fixed"].get("seed", 42)])
        for candidate in spec["candidates"]:
            cid = candidate["id"]
            seen = set()
            for seed in seeds:
                for fold_dir in sorted((predictions_root / model / cid / f"seed{seed}").glob("B*_*")):
                    path = fold_dir / "run_manifest.json"
                    if not path.is_file():
                        continue
                    body = json.loads(path.read_text(encoding="utf-8"))
                    if (body["candidate_id"] != cid or body["seed"] != seed or body["plan_sha256"] != plan_sha
                            or body["code_sha256"] != code_sha or body["target_artifact_accessed"] is not False):
                        raise ValueError(f"manifest mismatch: {path}")
                    prediction_path = fold_dir / "predictions.parquet"
                    if _sha(prediction_path) != body["outputs"]["predictions.parquet"]:
                        raise ValueError(f"prediction changed after sealing: {prediction_path}")
                    frame = pd.read_parquet(prediction_path)
                    fit_per_task = (body["fit_seconds"] + body["inference_seconds"]) / body["forecast_tasks"]
                    for task_id, part in frame.groupby("task_id"):
                        part = part.sort_values("horizon_day")
                        if task_id not in by_task or (task_id, seed) in seen:
                            raise ValueError(f"unexpected or duplicate B task {task_id}")
                        seen.add((task_id, seed))
                        actual = by_task[task_id].loc[list(range(1, 29))]
                        if not np.array_equal(pd.to_datetime(part["target_date"]).to_numpy(),
                                              pd.to_datetime(actual["target_date"]).to_numpy()):
                            raise ValueError(f"target dates differ: {task_id}")
                        mae = float(np.mean(np.abs(part["prediction"].to_numpy(float) - actual["actual"].to_numpy(float))))
                        scale = float(part["mase7_scale"].iloc[0]) if part["mase7_scale"].notna().all() else float("nan")
                        defined = math.isfinite(scale) and scale > 0
                        scores.append({"family": family, "model": model, "candidate_id": cid, "seed": seed,
                                       "task_id": task_id, "fold_id": actual["fold_id"].iloc[0],
                                       "market": actual["market"].iloc[0], "MAE": mae,
                                       "MASE7": mae / scale if defined else np.nan,
                                       "undefined_scale": not defined, "fit_seconds": fit_per_task})
            rows.append({"family": family, "model": model, "candidate_id": cid, "seeds": len(seeds),
                         "expected_paths": B_TASKS * len(seeds), "completed_paths": len(seen)})

    task_scores = pd.DataFrame(scores)
    summary = pd.DataFrame(rows)
    summary["missing_paths"] = summary["expected_paths"] - summary["completed_paths"]
    for index, row in summary.iterrows():
        part = task_scores.loc[task_scores["candidate_id"] == row["candidate_id"]] if len(task_scores) else task_scores
        if part.empty:
            continue
        per_task = part.groupby(["market", "fold_id", "task_id"], as_index=False)["MASE7"].mean()
        summary.loc[index, "mean_MASE7"] = float(per_task.groupby("market")["MASE7"].mean().mean())
        per_seed = [float(s.groupby("market")["MASE7"].mean().mean()) for _, s in part.groupby("seed")]
        summary.loc[index, "seed_sd_MASE7"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan
        summary.loc[index, "mean_fit_seconds"] = float(part["fit_seconds"].mean())
        summary.loc[index, "undefined_scale_paths"] = int(part["undefined_scale"].sum())
    summary["eligible"] = (summary["missing_paths"] == 0) & (summary.get("undefined_scale_paths", 0) == 0)

    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "task_metrics.parquet", lambda p: task_scores.to_parquet(p, index=False))
    summary.to_csv(output / "candidate_summary.csv", index=False)
    complete = bool(summary["missing_paths"].eq(0).all())
    status = {"complete": complete, "completed_paths": int(summary["completed_paths"].sum()),
              "expected_paths": int(summary["expected_paths"].sum()), "plan_sha256": plan_sha,
              "code_sha256": code_sha, "targets_read": "recovery_tuning, history=all only",
              "protected_targets_sha256": _sha(TARGETS), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic(output / "validation_status.json",
            lambda p: p.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8"))

    if freeze:
        if not complete:
            raise ValueError("cannot freeze before every registered candidate covers all B paths")
        if FREEZE.exists():
            raise ValueError(f"refusing to overwrite {FREEZE}")
        families = {}
        for family in plan["families"]:
            group = summary.loc[(summary["family"] == family) & summary["eligible"]].sort_values(
                ["mean_MASE7", "mean_fit_seconds", "candidate_id"], kind="stable")
            winner = group.iloc[0]
            params = next(c["params"] for c in plan["families"][family]["candidates"]
                          if c["id"] == winner["candidate_id"])
            families[family] = {"candidate_id": winner["candidate_id"], "params": params,
                                "validation_score": float(winner["mean_MASE7"]),
                                "seed_sd_MASE7": None if pd.isna(winner["seed_sd_MASE7"]) else float(winner["seed_sd_MASE7"]),
                                "mean_fit_seconds": float(winner["mean_fit_seconds"])}
        champion = min(families, key=lambda f: (families[f]["validation_score"], families[f]["mean_fit_seconds"],
                                                families[f]["candidate_id"]))
        frozen = {"schema_version": 1, "status": "frozen_after_B_group_tuning", "plan_sha256": plan_sha,
                  "code_sha256": code_sha, "calendar_sha256": plan["calendar"]["sha256"],
                  "selection_rule": plan["selection"]["rule"], "families": families,
                  "light_champion": {"family": champion, **families[champion]},
                  "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        _atomic(FREEZE, lambda p: p.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8"))
        status["frozen"] = frozen
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path,
                        default=REPO / "track1/results/light_models/recovery_tuning")
    parser.add_argument("--output", type=Path, default=REPO / "track1/evaluation/light_models_b")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    status = score(args.predictions_root, args.output, args.freeze)
    print(json.dumps({k: v for k, v in status.items() if k != "frozen"}, indent=2))
    print(pd.read_csv(args.output / "candidate_summary.csv").to_string(index=False))
    if "frozen" in status:
        print(json.dumps(status["frozen"]["light_champion"], indent=2))


if __name__ == "__main__":
    main()
