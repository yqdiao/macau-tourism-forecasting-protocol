"""Registered lightweight global models (track1/configs/light_model_plan.yaml).

Both families forecast all 27 Macau source markets of one origin at once and
see only values dated on or before that origin plus the frozen holiday calendar.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 28
MAX_ROWS = 1_500_000
KEEP_RECENT_ANCHORS = 365
MIN_ANCHOR = 55  # the 56-day window statistic needs 56 observed days


@dataclass(frozen=True)
class MarketHistory:
    task_id: str
    market: str
    start: date
    values: np.ndarray

    @property
    def end(self) -> date:
        return self.start + timedelta(days=len(self.values) - 1)


@dataclass
class OriginForecast:
    predictions: dict[str, np.ndarray]
    fit_seconds: float
    inference_seconds: float
    detail: dict


class Calendar:
    """Daily flags from the frozen statutory calendar."""

    def __init__(self, path: Path):
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        self.macau = set(frame.loc[frame["role"] == "macau", "date"])
        self.mainland = set(frame.loc[frame["role"] == "mainland", "date"])
        source = frame.loc[frame["role"] == "source"]
        self.source = {market: set(group["date"]) for market, group in source.groupby("market")}
        self._block = {}
        for day in sorted(self.mainland):
            if day - timedelta(days=1) in self.mainland:
                continue
            length, cursor = 0, day
            while cursor in self.mainland:
                length += 1
                cursor += timedelta(days=1)
            for offset in range(length):
                self._block[day + timedelta(days=offset)] = length
        self._mainland_sorted = np.array(sorted(self.mainland), dtype="datetime64[D]")

    def features(self, market: str, days: np.ndarray) -> np.ndarray:
        """Columns: dow, month, macau, mainland, source, days_to_mainland, block_length."""
        days = days.astype("datetime64[D]")
        dates = pd.DatetimeIndex(days)
        python_days = [d.date() for d in dates]
        source = self.source.get(market, set())
        macau = np.fromiter((d in self.macau for d in python_days), dtype=np.float32, count=len(days))
        mainland = np.fromiter((d in self.mainland for d in python_days), dtype=np.float32, count=len(days))
        local = np.fromiter((d in source for d in python_days), dtype=np.float32, count=len(days))
        block = np.fromiter((self._block.get(d, 0) for d in python_days), dtype=np.float32, count=len(days))
        position = np.searchsorted(self._mainland_sorted, days)
        after = self._mainland_sorted[np.minimum(position, len(self._mainland_sorted) - 1)]
        before = self._mainland_sorted[np.maximum(position - 1, 0)]
        to_after = (after - days).astype(int)
        to_before = (before - days).astype(int)
        signed = np.where(np.abs(to_before) < np.abs(to_after), to_before, to_after)
        signed = np.clip(signed, -7, 7).astype(np.float32)
        return np.column_stack([dates.dayofweek.to_numpy(np.float32), dates.month.to_numpy(np.float32),
                                macau, mainland, local, signed, block])


CALENDAR_COLUMNS = ["dow", "month", "macau_holiday", "mainland_holiday", "source_holiday",
                    "days_to_mainland_holiday", "mainland_block_length"]


def _check_aligned(histories: list[MarketHistory]) -> date:
    ends = {item.end for item in histories}
    if len(ends) != 1:
        raise ValueError(f"histories end on different days: {sorted(ends)}")
    if any(np.any(~np.isfinite(item.values)) or np.any(item.values < 0) for item in histories):
        raise ValueError("histories must be finite and non-negative")
    return ends.pop()


# --------------------------------------------------------------------------- GL

def _gl_features(z: np.ndarray, y: np.ndarray, anchors: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows for (anchor, horizon) pairs and the anchor log level."""
    level = np.log1p(np.array([y[t - 27:t + 1].mean() for t in anchors]))
    lags = z[anchors[:, None] - np.arange(28)[None, :]] - level[:, None]
    cumsum = np.concatenate([[0.0], np.cumsum(z)])
    cumsq = np.concatenate([[0.0], np.cumsum(z * z)])
    stats = []
    for window in (7, 14, 28, 56, 365):
        lo = np.maximum(anchors + 1 - window, 0)
        count = anchors + 1 - lo
        mean = (cumsum[anchors + 1] - cumsum[lo]) / count
        var = np.maximum((cumsq[anchors + 1] - cumsq[lo]) / count - mean * mean, 0.0)
        stats.extend([mean - level, np.sqrt(var)])
    ycum = np.concatenate([[0.0], np.cumsum(y)])
    lo365 = np.maximum(anchors - 364, 0)
    mean365 = (ycum[anchors + 1] - ycum[lo365]) / (anchors + 1 - lo365)
    ratio = level - np.log1p(mean365)
    anchor_block = np.column_stack([lags, *stats, ratio])
    rows = np.repeat(np.arange(len(anchors)), len(horizons))
    h = np.tile(horizons, len(anchors))
    t = anchors[rows]
    k_min = np.ceil(h / 7).astype(int)
    weekly = []
    for k in range(4):
        idx = t + h - 7 * (k_min + k)
        weekly.append(np.where(idx >= 0, z[np.maximum(idx, 0)], np.nan) - level[rows])
    features = np.column_stack([anchor_block[rows], np.column_stack(weekly), h.astype(np.float64)])
    return features.astype(np.float32), level


GL_NUMERIC = ([f"lag{k}" for k in range(28)]
              + [f"{kind}{w}" for w in (7, 14, 28, 56, 365) for kind in ("mean", "std")]
              + ["ratio28_365"] + [f"weekday_k{k}" for k in range(4)] + ["horizon"])


def _gl_anchors(n: int, window: int | str, cap_old: int | None) -> np.ndarray:
    last = n - 1 - HORIZON
    first = MIN_ANCHOR if window == "all" else max(MIN_ANCHOR, n - 1 - int(window))
    anchors = np.arange(first, last + 1)
    if cap_old is None or len(anchors) <= KEEP_RECENT_ANCHORS:
        return anchors
    recent = anchors[-KEEP_RECENT_ANCHORS:]
    old = anchors[:-KEEP_RECENT_ANCHORS]
    stride = max(1, math.ceil(len(old) / max(cap_old, 1)))
    return np.concatenate([old[::-1][::stride][::-1], recent])


def global_lightgbm(histories: list[MarketHistory], params: dict, calendar: Calendar | None,
                    *, pooled: bool = True, threads: int = 4, seed: int = 42) -> OriginForecast:
    import lightgbm as lgb

    origin = _check_aligned(histories)
    markets = [item.market for item in histories]
    groups = [histories] if pooled else [[item] for item in histories]
    horizons = np.arange(1, HORIZON + 1)
    predictions, fit_seconds, inference_seconds, rows_used = {}, 0.0, 0.0, 0
    strides = {}
    for group in groups:
        started = time.perf_counter()
        n_anchor_total = sum(len(_gl_anchors(len(item.values), params["anchor_window"], None)) for item in group)
        cap_old = None
        if n_anchor_total * HORIZON > MAX_ROWS:
            budget_anchors = MAX_ROWS // HORIZON - KEEP_RECENT_ANCHORS * len(group)
            cap_old = max(1, budget_anchors // len(group))
        x_parts, y_parts, m_parts = [], [], []
        for item in group:
            y = item.values.astype(np.float64)
            z = np.log1p(y)
            anchors = _gl_anchors(len(y), params["anchor_window"], cap_old)
            strides[item.market] = len(anchors)
            features, level = _gl_features(z, y, anchors, horizons)
            rows = np.repeat(np.arange(len(anchors)), HORIZON)
            target_index = anchors[rows] + np.tile(horizons, len(anchors))
            target = z[target_index] - level[rows]
            if calendar is not None:
                target_days = (np.datetime64(item.start) + target_index).astype("datetime64[D]")
                features = np.column_stack([features, calendar.features(item.market, target_days)])
            x_parts.append(features)
            y_parts.append(target.astype(np.float32))
            m_parts.append(np.full(len(target), markets.index(item.market), dtype=np.int32))
        x = np.vstack(x_parts)
        target = np.concatenate(y_parts)
        columns = GL_NUMERIC + (CALENDAR_COLUMNS if calendar is not None else [])
        frame = pd.DataFrame(x, columns=columns)
        frame["market_id"] = pd.Categorical(np.concatenate(m_parts), categories=range(len(markets)))
        model = lgb.LGBMRegressor(objective="regression", n_estimators=int(params["n_estimators"]),
                                  learning_rate=float(params["learning_rate"]),
                                  num_leaves=int(params["num_leaves"]), min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, n_jobs=threads, verbosity=-1)
        model.fit(frame, target, categorical_feature=["market_id"])
        fit_seconds += time.perf_counter() - started
        rows_used += len(frame)

        started = time.perf_counter()
        for item in group:
            y = item.values.astype(np.float64)
            z = np.log1p(y)
            anchor = np.array([len(y) - 1])
            features, level = _gl_features(z, y, anchor, horizons)
            if calendar is not None:
                days = (np.datetime64(item.start) + anchor[0] + horizons).astype("datetime64[D]")
                features = np.column_stack([features, calendar.features(item.market, days)])
            query = pd.DataFrame(features, columns=columns)
            query["market_id"] = pd.Categorical(np.full(HORIZON, markets.index(item.market)),
                                                categories=range(len(markets)))
            raw = np.expm1(model.predict(query) + level[0])
            predictions[item.market] = raw
        inference_seconds += time.perf_counter() - started
    return OriginForecast(predictions, fit_seconds, inference_seconds,
                          {"origin": origin.isoformat(), "pooled": pooled, "calendar": calendar is not None,
                           "training_rows": rows_used, "anchors_per_market": strides})


# --------------------------------------------------------------------------- NL

def _linear_module(kind: str, lookback: int):
    import torch
    from torch import nn

    class NLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(lookback, HORIZON)

        def forward(self, x):
            last = x[:, -1:]
            return self.linear(x - last) + last

    class DLinear(nn.Module):
        def __init__(self, kernel: int = 25):
            super().__init__()
            self.kernel = kernel
            self.trend = nn.Linear(lookback, HORIZON)
            self.seasonal = nn.Linear(lookback, HORIZON)

        def forward(self, x):
            pad = (self.kernel - 1) // 2
            padded = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, pad)], dim=1)
            trend = padded.unfold(1, self.kernel, 1).mean(dim=-1)
            return self.trend(trend) + self.seasonal(x - trend)

    if kind == "nlinear":
        return NLinear()
    if kind == "dlinear":
        return DLinear()
    raise ValueError(f"unknown linear model {kind}")


def global_linear(histories: list[MarketHistory], params: dict, *, seed: int,
                  threads: int = 2, max_epochs: int = 50, patience: int = 5,
                  batch_size: int = 256, learning_rate: float = 1e-3) -> OriginForecast:
    import copy
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    origin = _check_aligned(histories)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    lookback = int(params["lookback"])
    started = time.perf_counter()
    x_parts, y_parts, end_parts = [], [], []
    for item in histories:
        z = np.log1p(item.values.astype(np.float64)).astype(np.float32)
        ends = np.arange(lookback - 1, len(z) - HORIZON)
        if len(ends) < 10:
            raise ValueError(f"{item.market}: too little history for lookback {lookback}")
        x_parts.append(z[ends[:, None] - np.arange(lookback - 1, -1, -1)[None, :]])
        y_parts.append(z[ends[:, None] + np.arange(1, HORIZON + 1)[None, :]])
        end_parts.append(ends)
    x = np.vstack(x_parts)
    y = np.vstack(y_parts)
    ends = np.concatenate(end_parts)
    order = np.argsort(ends, kind="stable")
    x, y, ends = x[order], y[order], ends[order]
    validation = max(1, int(round(len(x) * 0.1)))
    train_x, train_y = torch.from_numpy(x[:-validation]), torch.from_numpy(y[:-validation])
    valid_x, valid_y = torch.from_numpy(x[-validation:]), torch.from_numpy(y[-validation:])
    model = _linear_module(params["model"], lookback)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_state, best_loss, stale, epochs_run = None, float("inf"), 0, 0
    for _ in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            current = float(loss_fn(model(valid_x), valid_y))
        epochs_run += 1
        if current < best_loss - 1e-8:
            best_loss, stale, best_state = current, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("linear model produced no finite validation loss")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.eval()
    predictions = {}
    with torch.no_grad():
        for item in histories:
            z = np.log1p(item.values.astype(np.float64)).astype(np.float32)
            window = torch.from_numpy(z[-lookback:][None, :])
            predictions[item.market] = np.expm1(model(window).numpy().reshape(-1).astype(np.float64))
    return OriginForecast(predictions, fit_seconds, time.perf_counter() - started,
                          {"origin": origin.isoformat(), "epochs_run": epochs_run,
                           "best_validation_mse": best_loss, "training_windows": int(len(train_x)),
                           "validation_windows": int(validation)})
