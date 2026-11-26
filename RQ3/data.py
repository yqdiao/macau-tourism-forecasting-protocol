"""Load sealed RQ1/RQ2 panels for RQ3 (read-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import POOL, RQ1, RQ2, MARKET_GROUP, log


def load_day_panel(models: list[str] | None = None) -> pd.DataFrame:
    models = models or (POOL + ["ets", "LL01"])
    days = pd.read_parquet(RQ2 / "raw/day_metrics_rq2.parquet")
    days = days[days.config_id.isin(models)].copy()
    days["origin_date"] = pd.to_datetime(days["origin_date"])
    days["target_date"] = pd.to_datetime(days["target_date"])
    days["market_group"] = days.market.map(lambda m: MARKET_GROUP.get(m, "Small"))
    return days


def load_task_panel(models: list[str] | None = None) -> pd.DataFrame:
    models = models or (POOL + ["ets", "LL01"])
    task = pd.read_parquet(RQ2 / "raw/task_metrics_rq2.parquet")
    task = task[task.config_id.isin(models)].copy()
    task["origin_date"] = pd.to_datetime(task["origin_date"])
    task["market_group"] = task.market.map(lambda m: MARKET_GROUP.get(m, "Small"))
    return task


def load_timing() -> pd.DataFrame:
    return pd.read_csv(RQ1 / "tables/table4_timing_and_price_ratio.csv")


def load_annual_volume() -> pd.DataFrame:
    return pd.read_csv(RQ1 / "tables/table4_annual_volume.csv")


def load_holiday_flags() -> set:
    from pathlib import Path as _P
    hol = pd.read_parquet(_P("/root/DYQ/macau-tourism-forecasting-protocol/shared/data/holiday_calendar.parquet"))
    hol["date"] = pd.to_datetime(hol["date"]).dt.date
    return set(hol.loc[hol.role.isin(["mainland", "macau"]), "date"])


def origin_schedule(days: pd.DataFrame, split: str) -> pd.DataFrame:
    """Unique origins with target_end for completed-history rule."""
    d = days[(days.split_name == split) & (days.config_id == "ets")]
    g = (
        d.groupby(["fold_id", "origin_date"], sort=False)
        .agg(target_end=("target_date", "max"), n_markets=("market", "nunique"))
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    g["origin_idx"] = np.arange(len(g))
    return g


def completed_history_mask(schedule: pd.DataFrame, origin_idx: int) -> np.ndarray:
    """H: prior origins whose target_end <= current origin_date."""
    cur = schedule.iloc[origin_idx]
    ends = schedule["target_end"].to_numpy()
    origins = schedule["origin_date"].to_numpy()
    return (np.arange(len(schedule)) < origin_idx) & (ends <= cur.origin_date)


def build_prediction_cube(days: pd.DataFrame, split: str, models: list[str]):
    """Return dict keyed by model -> DataFrame indexed like ets subset with prediction column."""
    base = days[(days.split_name == split) & (days.config_id == "ets")][
        ["market", "fold_id", "origin_date", "target_date", "horizon_day", "task_id", "actual", "q_all", "market_group"]
    ].copy()
    key = ["market", "fold_id", "origin_date", "horizon_day"]
    out = {"_base": base}
    for m in models:
        part = days[(days.split_name == split) & (days.config_id == m)][key + ["prediction"]].rename(
            columns={"prediction": m}
        )
        out[m] = part
    # wide merge
    wide = base.copy()
    for m in models:
        wide = wide.merge(out[m], on=key, how="left", validate="one_to_one")
    assert wide[models].notna().all().all(), "missing predictions in cube"
    return wide


def score_task_from_days(df: pd.DataFrame, pred_col: str = "yhat") -> pd.DataFrame:
    """df must have market, fold_id, origin_date, task_id, actual, q_all, yhat."""
    err = df[pred_col].to_numpy(float) - df["actual"].to_numpy(float)
    tmp = df[["market", "fold_id", "origin_date", "task_id", "q_all"]].copy()
    tmp["abs_error"] = np.abs(err)
    tmp["sq"] = err ** 2
    tmp["ape"] = np.where(df.actual.to_numpy(float) > 0, tmp.abs_error / df.actual.to_numpy(float), np.nan)
    g = tmp.groupby(["market", "fold_id", "origin_date", "task_id"], sort=False)
    task = g.agg(
        mae=("abs_error", "mean"),
        mse=("sq", "mean"),
        mape=("ape", "mean"),
        q_all=("q_all", "first"),
        n=("abs_error", "size"),
        positive_days=("ape", lambda s: int(np.isfinite(s).sum())),
    ).reset_index()
    task["rmse"] = np.sqrt(task.mse)
    task["mase"] = task.mae / task.q_all
    task["mape_pct"] = task.mape * 100.0
    return task
