"""Independent O01-O33 scoring for all 20 Track 1 models (history=all).

Predictions are read only from sealed outputs; targets are merged here for the
first time. Scales are recomputed from model-visible history so every model is
scored with the same denominator. Task MASE7 is averaged over seeds within task
(PatchTST/Informer), then over folds within market, then markets equally.
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
RESULTS = REPO / "track1/results"
TARGETS = REPO / "shared/data/targets.parquet"
INPUTS = REPO / "shared/data/splits/outer_backtest/task_inputs.parquet"
TASK_MANIFEST = REPO / "shared/data/task_manifest.json"
FROZEN_BASELINE = REPO / "track1/configs/frozen_baseline_configs.yaml"
FROZEN_FOUNDATION = REPO / "track1/configs/frozen_foundation_configs.yaml"
OUTPUT = REPO / "track1/evaluation/outer_main_table"
OUTER_TASKS = 957
HORIZON = 28
ZERO_SHOT = ["timesfm_3", "chronos_2", "moirai", "granite_patchtst_fm", "lag_llama"]
TRAINED = ["patchtst", "informer"]
FAMILY = {"zero_shot": "Foundation (zero-shot)", "trained": "Foundation (trained)", "baseline": "Baseline"}


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


def _sealed_fold(fold_dir: Path, model: str) -> pd.DataFrame:
    manifest = json.loads((fold_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != model or manifest["target_artifact_accessed"] is not False:
        raise ValueError(f"invalid manifest {fold_dir}")
    path = fold_dir / "predictions.parquet"
    if _sha(path) != manifest["outputs"]["predictions.parquet"]:
        raise ValueError(f"prediction changed after sealing: {path}")
    return pd.read_parquet(path)


def load_predictions() -> tuple[pd.DataFrame, dict]:
    frozen_b = json.loads(FROZEN_BASELINE.read_text(encoding="utf-8"))
    frozen_f = json.loads(FROZEN_FOUNDATION.read_text(encoding="utf-8"))
    columns = ["task_id", "model", "seed", "target_date", "horizon_day", "prediction", "mase7_scale"]
    frames, sources = [], {}
    for model, entry in frozen_b["models"].items():
        root = RESULTS / "outer_server_all/predictions" / model / entry["candidate_id"]
        files = sorted(root.glob("*/O*_*.parquet"))
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if set(frame["candidate_id"]) != {entry["candidate_id"]}:
            raise ValueError(f"{model}: candidate differs from frozen config")
        frames.append(frame[columns].assign(family="baseline", candidate_id=entry["candidate_id"]))
        sources[model] = {"family": "baseline", "candidate_id": entry["candidate_id"], "files": len(files)}
    for model in ZERO_SHOT:
        entry = frozen_f["models"][model]
        folds = sorted((RESULTS / "foundation_outer_default").glob(f"O*_*/{model}"))
        frame = pd.concat([_sealed_fold(d, model) for d in folds], ignore_index=True)
        frames.append(frame[columns].assign(family="zero_shot", candidate_id=entry["candidate_id"]))
        sources[model] = {"family": "zero_shot", "candidate_id": entry["candidate_id"], "folds": len(folds)}
    for model in TRAINED:
        entry = frozen_f["models"][model]
        seed_dirs = sorted((RESULTS / "foundation_outer_frozen" / model / entry["candidate_id"]).glob("seed*"))
        parts = []
        for seed_dir in seed_dirs:
            for fold_dir in sorted(seed_dir.glob("O*_*")):
                part = _sealed_fold(fold_dir, model)
                if set(part["candidate_id"]) != {entry["candidate_id"]}:
                    raise ValueError(f"{fold_dir}: candidate differs from frozen config")
                parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        if sorted(frame["seed"].unique()) != sorted(frozen_f["seeds"]):
            raise ValueError(f"{model}: seeds differ from frozen config")
        frames.append(frame[columns].assign(family="trained", candidate_id=entry["candidate_id"]))
        sources[model] = {"family": "trained", "candidate_id": entry["candidate_id"],
                          "seeds": [int(s) for s in sorted(frame["seed"].unique())]}
    return pd.concat(frames, ignore_index=True), sources


def main() -> None:
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    if _sha(TARGETS) != manifest["artifacts"]["targets.parquet"]["sha256"]:
        raise ValueError("protected target SHA256 differs from task manifest")

    inputs = pd.read_parquet(INPUTS)
    inputs = inputs.loc[inputs["history_length"] == "all"]
    if len(inputs) != OUTER_TASKS:
        raise ValueError("outer input set is incomplete")
    info = []
    for row in inputs.itertuples(index=False):
        history = np.asarray(row.history, dtype=float)
        difference = history[7:] - history[:-7]
        info.append({"task_id": row.task_id, "destination": row.destination, "market": row.market,
                     "fold_id": row.fold_id, "mase_scale": float(np.mean(np.abs(difference))),
                     "rmsse_scale": float(np.sqrt(np.mean(difference ** 2))),
                     "high_threshold": float(np.quantile(history, 0.9))})
    info = pd.DataFrame(info).set_index("task_id")

    targets = pd.read_parquet(TARGETS)
    targets = targets.loc[(targets["split_name"] == "outer_backtest") & (targets["history_length"] == "all")]
    if targets["task_id"].nunique() != OUTER_TASKS or len(targets) != OUTER_TASKS * HORIZON:
        raise ValueError("protected outer target set is incomplete")

    predictions, sources = load_predictions()
    key = ["task_id", "model", "seed", "target_date", "horizon_day"]
    if predictions.duplicated(key).any():
        raise ValueError("duplicate forecast rows")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    targets = targets.assign(target_date=pd.to_datetime(targets["target_date"]))
    scored = predictions.merge(targets[["task_id", "target_date", "horizon_day", "actual"]],
                               on=["task_id", "target_date", "horizon_day"], how="left", validate="many_to_one")
    if scored["actual"].isna().any():
        raise ValueError("forecast rows without a matching target")
    if not np.isfinite(scored["prediction"].to_numpy(float)).all():
        raise ValueError("non-finite forecasts")
    scored = scored.join(info, on="task_id")
    stored = scored["mase7_scale"].to_numpy(float)
    if not np.allclose(stored, scored["mase_scale"].to_numpy(float), rtol=1e-9, atol=1e-9):
        raise ValueError("stored MASE scale differs from recomputed scale")

    scored["abs_error"] = (scored["prediction"] - scored["actual"]).abs()
    scored["sq_error"] = (scored["prediction"] - scored["actual"]) ** 2
    scored["error"] = scored["prediction"] - scored["actual"]
    scored["high"] = scored["actual"] > scored["high_threshold"]
    group_keys = ["family", "model", "candidate_id", "seed", "task_id", "destination", "market", "fold_id"]
    counts = scored.groupby(group_keys)["horizon_day"].agg(["count", "min", "max"])
    if not ((counts["count"] == HORIZON) & (counts["min"] == 1) & (counts["max"] == HORIZON)).all():
        raise ValueError("incomplete 28-day path")

    def task_row(group: pd.DataFrame) -> pd.Series:
        mae = group["abs_error"].mean()
        high = group.loc[group["high"], "abs_error"]
        scale = group["mase_scale"].iloc[0]
        by_h = group.set_index("horizon_day")["abs_error"]
        return pd.Series({
            "MAE": mae, "RMSE": math.sqrt(group["sq_error"].mean()),
            "MASE7": mae / scale if scale > 0 else np.nan,
            "RMSSE7": math.sqrt(group["sq_error"].mean()) / group["rmsse_scale"].iloc[0],
            "WAPE": group["abs_error"].sum() / group["actual"].sum() if group["actual"].sum() > 0 else np.nan,
            "bias_pct": group["error"].sum() / group["actual"].sum() if group["actual"].sum() > 0 else np.nan,
            "high_n": int(group["high"].sum()),
            "high_MASE7": high.mean() / scale if len(high) and scale > 0 else np.nan,
            **{f"MASE7_h{h}": by_h.loc[h] / scale for h in (1, 7, 14, 28)},
        })

    tasks = scored.groupby(group_keys, sort=False).apply(task_row, include_groups=False).reset_index()
    per_task = (tasks.drop(columns="seed")
                .groupby(["family", "model", "candidate_id", "task_id", "destination", "market", "fold_id"],
                         as_index=False).mean(numeric_only=True))
    per_task["high_n"] = per_task["high_n"].round().astype(int)

    sn = per_task.loc[per_task["model"] == "seasonal_naive", ["task_id", "MASE7"]].rename(columns={"MASE7": "sn"})
    per_task = per_task.merge(sn, on="task_id", how="left")
    per_task["beats_sn"] = per_task["MASE7"] < per_task["sn"]

    def macro(frame: pd.DataFrame, metric: str) -> float:
        return float(frame.groupby("market")[metric].mean().mean())

    rows = []
    for (family, model, candidate), part in per_task.groupby(["family", "model", "candidate_id"]):
        macau = part.loc[part["destination"] == "Macau"]
        hk = part.loc[part["destination"] == "HongKong"]
        seed_part = tasks.loc[(tasks["model"] == model) & (tasks["destination"] == "Macau")]
        seed_scores = [macro(s, "MASE7") for _, s in seed_part.groupby("seed")]
        q = macau["MASE7"].quantile([0.25, 0.5, 0.75, 0.9])
        rows.append({
            "family": FAMILY[family], "model": model, "candidate_id": candidate,
            "seeds": len(seed_scores), "tasks_done": int(part["task_id"].nunique()), "tasks_expected": OUTER_TASKS,
            "Macau_MASE7": macro(macau, "MASE7"),
            "Macau_MASE7_seed_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else np.nan,
            "Macau_median_MASE7": float(q.loc[0.5]),
            "Macau_IQR_MASE7": f"{q.loc[0.25]:.3f}-{q.loc[0.75]:.3f}",
            "Macau_worst10pct_MASE7": float(macau.loc[macau["MASE7"] >= q.loc[0.9], "MASE7"].mean()),
            "Macau_RMSSE7": macro(macau, "RMSSE7"),
            "Macau_WAPE": macro(macau, "WAPE"),
            "Macau_bias_pct": macro(macau, "bias_pct"),
            "Macau_high_MASE7": macro(macau.dropna(subset=["high_MASE7"]), "high_MASE7"),
            "Macau_MASE7_h1": macro(macau, "MASE7_h1"), "Macau_MASE7_h7": macro(macau, "MASE7_h7"),
            "Macau_MASE7_h14": macro(macau, "MASE7_h14"), "Macau_MASE7_h28": macro(macau, "MASE7_h28"),
            "Macau_win_rate_vs_SN": float(macau["beats_sn"].mean()),
            "HK_MASE7": macro(hk, "MASE7"),
            "All29_MASE7": macro(part, "MASE7"),
        })
    table = pd.DataFrame(rows)
    sn_macau = float(table.loc[table["model"] == "seasonal_naive", "Macau_MASE7"].iloc[0])
    table["Macau_rel_SN"] = table["Macau_MASE7"] / sn_macau
    table = table.sort_values("Macau_MASE7", kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    if not (table["tasks_done"] == OUTER_TASKS).all():
        raise ValueError("a model does not cover all 957 outer tasks")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(tasks, OUTPUT / "task_metrics_by_seed.parquet")
    _atomic_parquet(per_task, OUTPUT / "task_metrics.parquet")
    _atomic_parquet(table, OUTPUT / "main_table.parquet")
    table.to_csv(OUTPUT / "main_table.csv", index=False)
    status = {
        "split": "outer_backtest O01-O33", "history": "all", "models": len(table),
        "tasks_per_model": OUTER_TASKS, "sources": sources,
        "protected_targets_sha256": _sha(TARGETS),
        "frozen_baseline_config_sha256": _sha(FROZEN_BASELINE),
        "frozen_foundation_config_sha256": _sha(FROZEN_FOUNDATION),
        "aggregation": "task MASE7 (seed mean) -> fold mean within market -> equal-weight market mean",
        "primary": "Macau 27 markets; Hong Kong 2 series reported separately",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["rank", "family", "model", "candidate_id", "seeds", "tasks_done", "Macau_MASE7", "Macau_MASE7_seed_sd",
            "Macau_rel_SN", "Macau_median_MASE7", "Macau_IQR_MASE7", "Macau_worst10pct_MASE7", "Macau_RMSSE7",
            "Macau_WAPE", "Macau_high_MASE7", "Macau_win_rate_vs_SN", "HK_MASE7", "All29_MASE7"]
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(table[["rank", "model", "Macau_MASE7_h1", "Macau_MASE7_h7", "Macau_MASE7_h14", "Macau_MASE7_h28",
                 "Macau_bias_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
