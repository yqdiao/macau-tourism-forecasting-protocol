"""Bootstrap CIs and DM helpers for RQ3 schemes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import B, BLOCK_REOPEN, BLOCK_STRESS, SEED, block_indices, dm_hln, market_then_equal, origin_means, stars


def summarize_task(task: pd.DataFrame) -> dict:
    return {
        "MASE": market_then_equal(task, "mase"),
        "MAPE": market_then_equal(task, "mape_pct"),
        "MAE": market_then_equal(task, "mae"),
        "RMSE": market_then_equal(task, "rmse"),
        "n_tasks": int(task.task_id.nunique()),
    }


def bootstrap_levels(task: pd.DataFrame, split: str) -> dict:
    block = BLOCK_REOPEN if split == "outer_backtest" else BLOCK_STRESS
    out = {}
    rng = np.random.default_rng(SEED)
    for metric, col in [("MASE", "mase"), ("MAPE", "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
        om = origin_means(task, col)
        if len(om) < 3:
            out[f"{metric}_ci95_low"] = np.nan
            out[f"{metric}_ci95_high"] = np.nan
            continue
        ii = block_indices(len(om), block, B, rng)
        boots = om.to_numpy()[ii].mean(1)
        out[f"{metric}_ci95_low"] = float(np.quantile(boots, 0.025))
        out[f"{metric}_ci95_high"] = float(np.quantile(boots, 0.975))
    return out


def pairwise_dm(task_a: pd.DataFrame, task_b: pd.DataFrame, metric: str = "mase") -> tuple[float, float, str]:
    oa = origin_means(task_a, metric)
    ob = origin_means(task_b, metric)
    idx = oa.index.intersection(ob.index)
    if len(idx) < 3:
        return np.nan, np.nan, ""
    dm, p = dm_hln(oa.loc[idx].to_numpy(), ob.loc[idx].to_numpy(), lag=2)
    return dm, p, stars(p)


def ratio_ci_vs(task: pd.DataFrame, ref: pd.DataFrame, metric: str, split: str) -> tuple[float, float, float]:
    block = BLOCK_REOPEN if split == "outer_backtest" else BLOCK_STRESS
    om = origin_means(task, metric)
    oe = origin_means(ref, metric)
    idx = om.index.intersection(oe.index)
    om, oe = om.loc[idx], oe.loc[idx]
    if len(om) < 3:
        return np.nan, np.nan, np.nan
    r = float(om.mean() / oe.mean())
    rng = np.random.default_rng(SEED)
    ii = block_indices(len(om), block, B, rng)
    boots = om.to_numpy()[ii].mean(1) / oe.to_numpy()[ii].mean(1)
    return r, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
