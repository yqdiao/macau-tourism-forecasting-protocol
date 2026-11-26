"""Build Macau reopen/stress task panels and a day-level cache for RQ2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CORE_IDS, FROZEN_BASELINE, LABEL, FAMILY, MARKET_GROUP, REPO, RAW, RQ1, LOGS,
    log, ensure_dirs,
)

TARGETS = REPO / "shared/data/targets.parquet"
QALL = RQ1 / "raw/q_all.parquet"


def _read_many(paths, columns):
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        keep = [c for c in columns if c in frame.columns]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def prediction_files(config_id: str, split: str, history: str = "all") -> list[Path]:
    root = REPO / "track1/results"
    if split == "outer_backtest" and history == "all":
        if config_id in {"granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"}:
            return sorted((root / "foundation_outer_default").glob(f"*/{config_id}/predictions.parquet"))
        if config_id == "GL03":
            return sorted((root / "light_models/outer_backtest/global_lightgbm/GL03").rglob("predictions.parquet"))
        if config_id == "LL01":
            return sorted((root / "light_models/outer_backtest/local_lightgbm/LL01").rglob("predictions.parquet"))
        if config_id == "NL01":
            # seed 42 only for multi-seed
            return sorted((root / "light_models/outer_backtest/global_linear/NL01/seed42").rglob("predictions.parquet"))
        cand = FROZEN_BASELINE[config_id]
        return sorted((root / f"outer_server_all/predictions/{config_id}/{cand}/42").glob("*.parquet"))
    if split == "pandemic_stress" and history == "all":
        if config_id in {"granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"}:
            return sorted((root / f"sensitivity/pandemic_stress/all/{config_id}/seed42").rglob("predictions.parquet"))
        if config_id == "GL03":
            return sorted((root / "light_models/pandemic_stress/global_lightgbm/GL03/seed42").rglob("predictions.parquet"))
        if config_id == "LL01":
            return sorted((root / "light_models/pandemic_stress/local_lightgbm/LL01/seed42").rglob("predictions.parquet"))
        if config_id == "NL01":
            return sorted((root / "light_models/pandemic_stress/global_linear/NL01/seed42").rglob("predictions.parquet"))
        cand = FROZEN_BASELINE[config_id]
        return sorted((root / f"sensitivity/pandemic_stress/all/baselines/predictions/{config_id}/{cand}/42").glob("*.parquet"))
    if split == "outer_backtest" and history in {"180", "365"}:
        if config_id in {"granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"}:
            return sorted((root / f"sensitivity/outer_backtest/{history}/{config_id}/seed42").rglob("predictions.parquet"))
        if config_id == "GL03":
            return sorted((root / f"light_models/outer_backtest/{history}/global_lightgbm/GL03").rglob("predictions.parquet"))
        if config_id == "LL01":
            return sorted((root / f"light_models/outer_backtest/{history}/local_lightgbm/LL01").rglob("predictions.parquet"))
        if config_id == "NL01":
            return sorted((root / f"light_models/outer_backtest/{history}/global_linear/NL01").rglob("predictions.parquet"))
        cand = FROZEN_BASELINE.get(config_id)
        if cand is None:
            return []
        return sorted((root / f"sensitivity/outer_backtest/{history}/baselines/predictions/{config_id}/{cand}/42").glob("*.parquet"))
    return []


def score_days(days: pd.DataFrame, qall: pd.DataFrame) -> pd.DataFrame:
    days = days.copy()
    days["origin_date"] = pd.to_datetime(days["origin_date"]).dt.date
    days = days.merge(qall, on=["market", "origin_date"], how="left")
    days["error"] = days["prediction"] - days["actual"]
    days["abs_error"] = days["error"].abs()
    days["sq"] = days["error"] ** 2
    days["ape"] = np.where(days["actual"] > 0, days["abs_error"] / days["actual"], np.nan)
    g = days.groupby(["config_id", "market", "fold_id", "origin_date", "task_id"], sort=False)
    task = g.agg(
        n=("horizon_day", "size"),
        mae=("abs_error", "mean"),
        mse=("sq", "mean"),
        mape=("ape", "mean"),
        q_all=("q_all", "first"),
        zero_days=("actual", lambda s: int((s == 0).sum())),
        positive_days=("actual", lambda s: int((s > 0).sum())),
        sum_abs=("abs_error", "sum"),
        sum_actual=("actual", "sum"),
    ).reset_index()
    task["rmse"] = np.sqrt(task["mse"])
    task["mase"] = task["mae"] / task["q_all"]
    task["mape_pct"] = task["mape"] * 100.0
    task["label"] = task["config_id"].map(LABEL)
    task["family"] = task["config_id"].map(FAMILY)
    task["market_group"] = task["market"].map(MARKET_GROUP)
    return task, days


def load_day_predictions(config_id: str, split: str, history: str, targets: pd.DataFrame) -> pd.DataFrame:
    files = prediction_files(config_id, split, history)
    if not files:
        raise FileNotFoundError(f"no predictions for {config_id} {split} history={history}")
    cols = ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day",
            "target_date", "prediction", "seed", "candidate_id", "history_length",
            "training_history_days", "effective_context_days", "history_start"]
    frame = _read_many(files, cols)
    if "destination" in frame.columns:
        frame = frame[frame["destination"] == "Macau"]
    if "seed" in frame.columns and frame["seed"].nunique() > 1:
        frame = frame[frame["seed"] == 42]
    if config_id in FROZEN_BASELINE and "candidate_id" in frame.columns:
        frame = frame[frame["candidate_id"] == FROZEN_BASELINE[config_id]]
    frame = frame.merge(targets, on=["task_id", "horizon_day"], how="inner")
    frame["config_id"] = config_id
    frame["split_name"] = split
    frame["history_setting"] = history
    return frame


def build_main_panels() -> None:
    ensure_dirs()
    qall = pd.read_parquet(QALL)
    qall["origin_date"] = pd.to_datetime(qall["origin_date"]).dt.date
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "destination"])
    targets = targets[targets["destination"] == "Macau"][["task_id", "horizon_day", "actual", "split_name"]]

    task_rows, day_rows, notes = [], [], []
    for split, expected_tasks in [("outer_backtest", 891), ("pandemic_stress", 351)]:
        split_targets = targets[targets["split_name"] == split][["task_id", "horizon_day", "actual"]]
        for config_id in CORE_IDS:
            log(f"scoring {split} {config_id}")
            days = load_day_predictions(config_id, split, "all", split_targets)
            task, days = score_days(days, qall)
            n_tasks = int(task["task_id"].nunique())
            zeros = int((days["actual"] == 0).sum())
            pos = int((days["actual"] > 0).sum())
            notes.append({
                "split_name": split, "config_id": config_id, "tasks": n_tasks,
                "day_rows": len(days), "zero_obs": zeros, "positive_obs": pos,
                "expected_tasks": expected_tasks, "ok": n_tasks == expected_tasks,
            })
            if n_tasks != expected_tasks:
                log(f"WARNING {config_id} {split}: tasks={n_tasks} expected={expected_tasks}")
            task["split_name"] = split
            task_rows.append(task)
            # keep day panel only for modules that need it (all core, main history)
            keep = days[["config_id", "split_name", "history_setting", "task_id", "market", "fold_id",
                         "origin_date", "target_date", "horizon_day", "actual", "prediction",
                         "abs_error", "sq", "ape", "q_all"]].copy()
            day_rows.append(keep)

    task = pd.concat(task_rows, ignore_index=True)
    days = pd.concat(day_rows, ignore_index=True)
    task.to_parquet(RAW / "task_metrics_rq2.parquet", index=False)
    days.to_parquet(RAW / "day_metrics_rq2.parquet", index=False)
    pd.DataFrame(notes).to_csv(LOGS / "panel_coverage_check.csv", index=False)
    log(f"wrote task panel {len(task)} rows, day panel {len(days)} rows")

    # Cross-check reopen against RQ1 task_metrics MASE
    rq1 = pd.read_parquet(RQ1 / "raw/task_metrics.parquet")
    rq1 = rq1[rq1["config_id"].isin(CORE_IDS)]
    mine = task[task["split_name"] == "outer_backtest"]
    rows = []
    for cid in CORE_IDS:
        a = float(rq1[rq1.config_id == cid].groupby("market").mase.mean().mean())
        b = float(mine[mine.config_id == cid].groupby("market").mase.mean().mean())
        rows.append({"config_id": cid, "rq1_mase": a, "rq2_mase": b, "abs_diff": abs(a - b)})
    chk = pd.DataFrame(rows)
    chk.to_csv(LOGS / "rq1_reopen_mase_check.csv", index=False)
    log("RQ1 MASE check max abs diff = %.6g" % chk.abs_diff.max())


def build_history_panels(models: list[str] | None = None) -> None:
    """Score Clean-180 / Clean-365 / Full-all for representative models."""
    ensure_dirs()
    models = models or [
        "granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai",
        "ets", "seasonal_naive", "svr", "lightgbm",
    ]
    qall = pd.read_parquet(QALL)
    qall["origin_date"] = pd.to_datetime(qall["origin_date"]).dt.date
    # Own denominators from normalized daily for 180/365 windows
    daily = pd.DataFrame(json.loads(line) for line in open(REPO / "shared/data/normalized_daily.jsonl"))
    daily = daily[daily.destination == "Macau"].copy()
    daily["date"] = pd.to_datetime(daily["date"])

    def q_window(market: str, origin, days: int) -> float:
        series = daily[daily.market == market].sort_values("date").set_index("date").arrivals.astype(float)
        origin = pd.Timestamp(origin)
        hist = series.loc[:origin].iloc[-days:]
        if len(hist) <= 7:
            return np.nan
        return float((hist - hist.shift(7)).dropna().abs().mean())

    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "destination"])
    targets = targets[(targets.destination == "Macau") & (targets.split_name == "outer_backtest")][
        ["task_id", "horizon_day", "actual"]
    ]
    rows = []
    for history in ["all", "365", "180"]:
        for config_id in models:
            files = prediction_files(config_id, "outer_backtest", history)
            if not files:
                log(f"MISSING history={history} {config_id}")
                continue
            log(f"history score {history} {config_id} files={len(files)}")
            days = load_day_predictions(config_id, "outer_backtest", history, targets)
            task, _ = score_days(days, qall)
            # own denom
            origins = task[["market", "origin_date"]].drop_duplicates()
            own = []
            window = {"all": None, "365": 365, "180": 180}[history]
            for r in origins.itertuples(index=False):
                if window is None:
                    # full history q equals q_all construction; reuse q_all
                    qq = qall[(qall.market == r.market) & (qall.origin_date == pd.Timestamp(r.origin_date).date())]
                    val = float(qq.q_all.iloc[0]) if len(qq) else np.nan
                else:
                    val = q_window(r.market, r.origin_date, window)
                own.append({"market": r.market, "origin_date": pd.Timestamp(r.origin_date).date(), "q_own": val})
            own = pd.DataFrame(own)
            task = task.merge(own, on=["market", "origin_date"], how="left")
            task["mase_own"] = task["mae"] / task["q_own"]
            task["mase_fixed"] = task["mase"]  # already q_all
            task["history_setting"] = history
            rows.append(task)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RAW / "history_task_metrics.parquet", index=False)
    log(f"wrote history panel {len(out)} rows")


if __name__ == "__main__":
    build_main_panels()
    build_history_panels()
