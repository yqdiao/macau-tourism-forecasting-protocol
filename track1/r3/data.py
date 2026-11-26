"""Load causal Macau histories from the original task inputs. No targets."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import CALENDAR, ORIGINAL_CODE, TASK_INPUTS

sys.path.insert(0, str(ORIGINAL_CODE.parent))
import light_models  # noqa: E402


@dataclass
class OriginBundle:
    split: str
    fold: str
    origin: date
    start: date
    markets: list[str]
    task_ids: list[str]
    values: np.ndarray  # (27, T), each row ends on origin


def load_origins() -> dict[tuple[str, str], OriginBundle]:
    frame = pd.read_parquet(
        TASK_INPUTS,
        columns=["task_id", "split_name", "fold_id", "destination", "market", "origin_date",
                 "history_length", "history_start", "history"],
    )
    frame = frame[(frame["destination"] == "Macau") & (frame["history_length"] == "all")]
    bundles = {}
    for (split, fold), group in frame.groupby(["split_name", "fold_id"], sort=False):
        group = group.sort_values("market")
        if len(group) != 27:
            raise RuntimeError(f"{split} {fold}: expected 27 markets, got {len(group)}")
        origin = pd.Timestamp(group["origin_date"].iloc[0]).date()
        start = pd.Timestamp(group["history_start"].iloc[0]).date()
        rows = [np.asarray(v, dtype=np.float64) for v in group["history"]]
        if any(pd.Timestamp(s).date() != start for s in group["history_start"]):
            raise RuntimeError(f"{split} {fold}: history starts differ")
        if len({len(v) for v in rows}) != 1:
            raise RuntimeError(f"{split} {fold}: history lengths differ")
        bundles[(split, fold)] = OriginBundle(
            split, fold, origin, start, group["market"].tolist(), group["task_id"].astype(str).tolist(),
            np.vstack(rows),
        )
    return bundles


def load_calendar():
    return light_models.Calendar(CALENDAR)


def seasonal_naive(values_1d: np.ndarray) -> np.ndarray:
    """y[o+h-7*ceil(h/7)] with o the last observed index."""
    o = len(values_1d) - 1
    out = np.empty(28, dtype=np.float64)
    for h in range(1, 29):
        out[h - 1] = values_1d[o + h - 7 * int(np.ceil(h / 7))]
    return out


def date_at(start: date, index: int) -> date:
    return start + timedelta(days=int(index))
