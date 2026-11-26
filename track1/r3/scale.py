"""Causal seasonal scales. Evaluation q_ref uses the full history through the origin."""
from __future__ import annotations

import numpy as np


def prefix_scales(y: np.ndarray) -> np.ndarray:
    """q[t] = mean |y[i]-y[i-7]| for i=7..t. NaN when fewer than one pair."""
    n = len(y)
    q = np.full(n, np.nan, dtype=np.float64)
    if n <= 7:
        return q
    diff = np.abs(y[7:] - y[:-7])
    csum = np.cumsum(diff)
    idx = np.arange(7, n)
    q[idx] = csum[idx - 7] / (idx - 6)
    return q


def level_28(y: np.ndarray) -> np.ndarray:
    """Inclusive mean of the 28 days ending at each t. NaN before t=27."""
    n = len(y)
    level = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(y)
    idx = np.arange(27, n)
    prev = np.where(idx >= 28, csum[idx - 28], 0.0)
    level[idx] = (csum[idx] - prev) / 28.0
    return level
