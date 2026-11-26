"""Shared-anchor LightGBM fits. Local models are 27 separate trees."""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data import OriginBundle, seasonal_naive
from features import (
    ANNUAL_COLUMNS, CALENDAR_COLUMNS, GL_NUMERIC, PHASE_COLUMNS, annual_columns,
    legal_anchors, log_target_and_features, phase_columns, recent_level, scaled_target,
    week4_baseline,
)
from scale import prefix_scales

TREE_PARAMS = dict(num_leaves=63, n_estimators=1000, learning_rate=0.03, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def _booster(objective: str, seed: int, threads: int, n_estimators: int | None = None):
    import lightgbm as lgb
    params = dict(TREE_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return lgb.LGBMRegressor(
        objective=objective, random_state=seed, n_jobs=threads, verbosity=-1,
        device_type="cpu", **params,
    )


def _rows_for_market(bundle: OriginBundle, market_index: int, spec: dict, calendar) -> dict:
    y = bundle.values[market_index]
    z = np.log1p(y)
    anchors = legal_anchors(len(y))
    q_all = prefix_scales(y)
    q_anchor = q_all[anchors]
    market = bundle.markets[market_index]
    features, log_target, columns, target_index, level_rep = log_target_and_features(
        z, y, anchors, calendar, market, np.datetime64(bundle.start), True)
    level = recent_level(y, anchors)
    week = week4_baseline(y, anchors)
    mode = spec["target"]
    if mode == "log":
        target = log_target
        keep = np.ones(len(target), dtype=bool)
        baseline = None
        q_row = None
    else:
        baseline = level if mode == "mean28" else week
        target, target_index, q_row = scaled_target(y, anchors, baseline, q_anchor)
        keep = np.isfinite(q_row) & (q_row > 0) & np.isfinite(target)
    frame_x = features
    if spec.get("annual"):
        extra = annual_columns(y, anchors, level, target_index)
        frame_x = np.column_stack([frame_x, extra])
        columns = columns + ANNUAL_COLUMNS
    if spec.get("phase"):
        extra = phase_columns(calendar, bundle.start, target_index, getattr(calendar, "_phase_lookup", None))
        frame_x = np.column_stack([frame_x, extra])
        columns = columns + PHASE_COLUMNS
    return {
        "x": frame_x[keep], "y": target[keep].astype(np.float64), "columns": columns,
        "anchors": np.repeat(anchors, 28)[keep], "q": None if q_row is None else q_row[keep],
        "n_raw": int(len(target)), "n_kept": int(keep.sum()), "market": market,
    }


def _predict_market(model, columns, bundle, market_index, spec, calendar, categories):
    y = bundle.values[market_index]
    z = np.log1p(y)
    anchor = np.array([len(y) - 1])
    market = bundle.markets[market_index]
    features, _, _, target_index, level_rep = log_target_and_features(
        z, y, anchor, calendar, market, np.datetime64(bundle.start), True, need_target=False)
    level = recent_level(y, anchor)
    columns_now = list(GL_NUMERIC) + list(CALENDAR_COLUMNS)
    x = features
    if spec.get("annual"):
        x = np.column_stack([x, annual_columns(y, anchor, level, target_index)])
        columns_now += ANNUAL_COLUMNS
    if spec.get("phase"):
        x = np.column_stack([x, phase_columns(calendar, bundle.start, target_index, getattr(calendar, "_phase_lookup", None))])
        columns_now += PHASE_COLUMNS
    if columns_now != columns:
        raise RuntimeError("prediction columns diverged from training columns")
    frame = pd.DataFrame(x, columns=columns)
    frame["market_id"] = pd.Categorical(np.full(28, market_index), categories=categories)
    pred = np.asarray(model.predict(frame), dtype=np.float64)
    q = prefix_scales(y)[-1]
    if spec["target"] == "log":
        raw = np.expm1(pred + np.log1p(level[0]))
    else:
        base = level if spec["target"] == "mean28" else week4_baseline(y, anchor)
        raw = base + q * pred
    return raw


def _omega(anchors: np.ndarray, n: int, half_life) -> np.ndarray:
    age = (n - 1 - 28) - anchors
    if half_life in (None, "inf"):
        return np.ones(len(anchors), dtype=np.float64)
    return np.power(2.0, -age / float(half_life))


def fit_trees(bundle: OriginBundle, spec: dict, calendar, *, pooled: bool, seed: int = 42,
              threads: int = 4, n_estimators: int | None = None) -> dict:
    started = time.perf_counter()
    if spec.get("phase"):
        from features import phase_lookup
        calendar._phase_lookup = getattr(calendar, "_phase_lookup", None) or phase_lookup(calendar)
    parts = [_rows_for_market(bundle, i, spec, calendar) for i in range(27)]
    for part, i in zip(parts, range(27)):
        omega = _omega(part["anchors"], bundle.values.shape[1], spec.get("half_life", "inf"))
        part["omega"] = omega
        part["w_raw"] = omega / omega.sum() if omega.sum() else omega
    columns = parts[0]["columns"]
    categories = list(range(27))
    models = []
    groups = [list(range(27))] if pooled else [[i] for i in range(27)]
    weight_stats = []
    for group in groups:
        xs, ys, ms, ws = [], [], [], []
        for i in group:
            part = parts[i]
            if part["n_kept"] == 0:
                continue
            xs.append(part["x"])
            ys.append(part["y"])
            ms.append(np.full(part["n_kept"], i, dtype=np.int32))
            ws.append(part["w_raw"])
        if not xs:
            raise RuntimeError("no training rows")
        w = np.concatenate(ws)
        w = w / w.mean()
        frame = pd.DataFrame(np.vstack(xs), columns=columns)
        frame["market_id"] = pd.Categorical(np.concatenate(ms), categories=categories)
        model = _booster("regression_l1" if spec["l1"] else "regression", seed, threads, n_estimators)
        model.fit(frame, np.concatenate(ys), sample_weight=w, categorical_feature=["market_id"])
        models.append((group, model))
        weight_stats.append({"n": int(len(w)), "sum": float(w.sum()), "min": float(w.min()), "max": float(w.max()),
                             "ess": float((w.sum() ** 2) / np.sum(w * w))})
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions, statuses = {}, {}
    lookup = {}
    for group, model in models:
        for i in group:
            lookup[i] = model
    for i, market in enumerate(bundle.markets):
        try:
            raw = _predict_market(lookup[i], columns, bundle, i, spec, calendar, categories)
            if raw.shape != (28,) or not np.isfinite(raw).all():
                raise ValueError("non-finite forecast")
            predictions[market] = raw
            statuses[market] = "native"
        except Exception as exc:  # noqa: BLE001
            predictions[market] = seasonal_naive(bundle.values[i])
            statuses[market] = f"fallback:{type(exc).__name__}"
    return {
        "predictions": predictions, "status": statuses, "fit_seconds": fit_seconds,
        "inference_seconds": time.perf_counter() - started, "rows": int(sum(p["n_kept"] for p in parts)),
        "dropped": int(sum(p["n_raw"] - p["n_kept"] for p in parts)), "weight_stats": weight_stats,
        "columns": columns,
    }
