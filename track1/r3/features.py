"""Feature rows and baselines. Windows follow the original 28-day inclusive level."""
from __future__ import annotations

import sys

import numpy as np

from common import ORIGINAL_CODE
from scale import level_28, prefix_scales

sys.path.insert(0, str(ORIGINAL_CODE.parent))
import light_models  # noqa: E402

HORIZONS = np.arange(1, 29)
GL_NUMERIC = light_models.GL_NUMERIC
CALENDAR_COLUMNS = light_models.CALENDAR_COLUMNS
ANNUAL_COLUMNS = ["annual364", "annual365", "missing_annual364", "missing_annual365"]
PHASE_COLUMNS = ["holiday_day_index", "holiday_days_remaining", "pre_holiday_3d", "post_holiday_3d"]


def legal_anchors(n: int) -> np.ndarray:
    last = n - 1 - 28
    return np.arange(light_models.MIN_ANCHOR, last + 1)


def week4_baseline(y: np.ndarray, anchors: np.ndarray, horizons: np.ndarray = HORIZONS) -> np.ndarray:
    rows = np.repeat(np.arange(len(anchors)), len(horizons))
    h = np.tile(horizons, len(anchors))
    t = anchors[rows]
    k0 = np.ceil(h / 7.0).astype(np.int64)
    cols = []
    for k in range(4):
        idx = t + h - 7 * (k0 + k)
        if np.any(idx < 0) or np.any(idx > t):
            raise RuntimeError("week4 baseline looked past the anchor")
        cols.append(y[idx])
    return np.median(np.column_stack(cols), axis=1)


def log_target_and_features(z, y, anchors, calendar, market, start_ordinal, with_calendar: bool, need_target: bool = True):
    features, level = light_models._gl_features(z, y, anchors, HORIZONS)
    rows = np.repeat(np.arange(len(anchors)), 28)
    target_index = anchors[rows] + np.tile(HORIZONS, len(anchors))
    # Prediction anchors sit on the origin, so target indexes lie in the future and must not be read.
    if need_target:
        target = z[target_index] - level[rows]
    else:
        target = np.zeros(len(target_index), dtype=np.float64)
    columns = list(GL_NUMERIC)
    if with_calendar:
        days = (np.datetime64(start_ordinal) + target_index).astype("datetime64[D]")
        features = np.column_stack([features, calendar.features(market, days)])
        columns += list(CALENDAR_COLUMNS)
    return features.astype(np.float32), target.astype(np.float64), columns, target_index, level[rows]


def recent_level(y: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    levels = level_28(y)
    return np.repeat(levels[anchors], 28)


def scaled_target(y, anchors, baseline: np.ndarray, q_at_anchor: np.ndarray):
    rows = np.repeat(np.arange(len(anchors)), 28)
    target_index = anchors[rows] + np.tile(HORIZONS, len(anchors))
    q = np.repeat(q_at_anchor, 28)
    u = (y[target_index] - baseline) / q
    return u, target_index, q


def annual_columns(y, anchors, level_repeated: np.ndarray, target_index: np.ndarray) -> np.ndarray:
    out = np.full((len(target_index), 4), np.nan, dtype=np.float32)
    for col, lag in ((0, 364), (1, 365)):
        idx = target_index - lag
        ok = idx >= 0
        logged = np.full(len(idx), np.nan)
        logged[ok] = np.log1p(y[idx[ok]]) - np.log1p(level_repeated[ok])
        out[:, col] = logged
        out[:, col + 2] = (~ok).astype(np.float32)
    return out


def phase_lookup(calendar) -> tuple[np.ndarray, int, int]:
    """Ordinal-indexed phase table. Dates outside the calendar stay NaN."""
    from datetime import timedelta
    covered = set(calendar.macau) | set(calendar.mainland) | {d for s in calendar.source.values() for d in s}
    lo, hi = min(covered), max(covered)
    lo_ord, hi_ord = lo.toordinal(), hi.toordinal()
    span = hi_ord - lo_ord + 1
    table = np.full((span, 4), np.nan, dtype=np.float32)
    blocks = []
    cursor = lo
    while cursor <= hi:
        if cursor in calendar.mainland and (cursor - timedelta(days=1)) not in calendar.mainland:
            length, day = 0, cursor
            while day in calendar.mainland:
                length += 1
                day += timedelta(days=1)
            blocks.append((cursor, length))
        cursor += timedelta(days=1)
    starts = np.array([b[0].toordinal() for b in blocks], dtype=np.int64)
    ends = np.array([(s + timedelta(days=n - 1)).toordinal() for s, n in blocks], dtype=np.int64)
    for s, n in blocks:
        for offset in range(n):
            pos = s.toordinal() + offset - lo_ord
            table[pos, 0] = offset + 1
            table[pos, 1] = n - offset
            table[pos, 2] = 0.0
            table[pos, 3] = 0.0
    for pos in range(span):
        if np.isfinite(table[pos, 0]):
            continue
        day = lo_ord + pos
        nxt = starts[starts > day]
        prv = ends[ends < day]
        table[pos, 0] = 0.0
        table[pos, 1] = 0.0
        table[pos, 2] = 1.0 if len(nxt) and 1 <= int(nxt[0] - day) <= 3 else 0.0
        table[pos, 3] = 1.0 if len(prv) and 1 <= int(day - prv[-1]) <= 3 else 0.0
    return table, lo_ord, hi_ord


def phase_columns(calendar, start_date, target_index: np.ndarray, lookup=None) -> np.ndarray:
    table, lo_ord, hi_ord = lookup if lookup is not None else phase_lookup(calendar)
    ords = start_date.toordinal() + target_index.astype(np.int64)
    out = np.full((len(ords), 4), np.nan, dtype=np.float32)
    inside = (ords >= lo_ord) & (ords <= hi_ord)
    out[inside] = table[ords[inside] - lo_ord]
    return out
