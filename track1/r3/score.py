"""Common-scale MASE. Positive paired difference means the new model is better."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import TARGETS
from data import load_origins
from scale import prefix_scales


def scale_frame(bundles: dict) -> pd.DataFrame:
    rows = []
    for bundle in bundles.values():
        for i, (market, task_id) in enumerate(zip(bundle.markets, bundle.task_ids)):
            q = float(prefix_scales(bundle.values[i])[-1])
            rows.append({
                "task_id": task_id, "split": bundle.split, "fold": bundle.fold, "market": market,
                "origin": bundle.origin.isoformat(), "q_ref": q, "scale_valid": bool(np.isfinite(q) and q > 0),
            })
    return pd.DataFrame(rows)


def load_targets() -> pd.DataFrame:
    frame = pd.read_parquet(
        TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length", "destination"])
    frame = frame[(frame["destination"] == "Macau") & (frame["history_length"] == "all")]
    return frame.rename(columns={"split_name": "split"})


def task_metrics(predictions: pd.DataFrame, targets: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(targets[["task_id", "horizon_day", "actual"]], on=["task_id", "horizon_day"], how="left")
    if merged["actual"].isna().any():
        raise RuntimeError("predictions missing actuals")
    merged = merged.merge(scales[["task_id", "q_ref", "scale_valid", "market", "fold", "origin", "split"]],
                          on="task_id", how="left", suffixes=("", "_s"))
    merged["abs_err"] = (merged["prediction"] - merged["actual"]).abs()
    grouped = merged.groupby(["model_id", "seed", "task_id"], as_index=False).agg(
        mae=("abs_err", "mean"), q_ref=("q_ref", "first"), scale_valid=("scale_valid", "first"),
        market=("market", "first"), fold=("fold", "first"), origin=("origin", "first"), split=("split", "first"),
        native=("status", lambda s: bool((s == "native").all())),
        signed_bias=("prediction", "mean"), actual_mean=("actual", "mean"))
    grouped["signed_scaled_bias"] = np.where(grouped["scale_valid"], (grouped["signed_bias"] - grouped["actual_mean"]) / grouped["q_ref"], np.nan)
    grouped["MASE7_common"] = np.where(grouped["scale_valid"], grouped["mae"] / grouped["q_ref"], np.nan)
    return grouped


def macro_mase(metrics: pd.DataFrame) -> float:
    seed_mean = metrics.groupby(["market", "task_id"], as_index=False)["MASE7_common"].mean()
    by_market = seed_mean.groupby("market")["MASE7_common"].mean()
    if by_market.isna().any():
        raise RuntimeError("macro MASE has markets with no valid tasks")
    return float(by_market.mean())


def block_bootstrap(metrics: pd.DataFrame, baseline: pd.DataFrame, n: int = 2000, seed: int = 20261018) -> dict:
    """Resample origin blocks of length 3, keeping all 27 markets."""
    a = metrics.groupby(["fold", "market"], as_index=False)["MASE7_common"].mean()
    b = baseline.groupby(["fold", "market"], as_index=False)["MASE7_common"].mean()
    merged = a.merge(b, on=["fold", "market"], suffixes=("_new", "_base"))
    folds = sorted(merged["fold"].unique())
    matrix_new = merged.pivot(index="fold", columns="market", values="MASE7_common_new").loc[folds].to_numpy()
    matrix_base = merged.pivot(index="fold", columns="market", values="MASE7_common_base").loc[folds].to_numpy()
    diff = matrix_base.mean(axis=1) - matrix_new.mean(axis=1)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(folds) - 2)
    draws = []
    for _ in range(n):
        pick = rng.choice(starts, size=int(np.ceil(len(folds) / 3)), replace=True)
        idx = np.concatenate([np.arange(s, s + 3) for s in pick])[:len(folds)]
        draws.append(float(diff[idx].mean()))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"diff_base_minus_new": float(diff.mean()), "low": float(lo), "high": float(hi)}
