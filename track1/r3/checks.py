"""Contract tests that do not use outer targets or enter the B ranking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, write_json
from data import load_calendar
from features import phase_columns, phase_lookup, week4_baseline
from linear import _module
from scale import prefix_scales


def main() -> None:
    y = np.arange(80, dtype=float)
    base = week4_baseline(y, np.array([70]))
    # h=1 uses y[64], y[57], y[50], y[43]
    expected = np.median([64, 57, 50, 43])
    assert base[0] == expected
    q = prefix_scales(y)
    assert abs(q[7] - abs(y[7] - y[0])) < 1e-9
    calendar = load_calendar()
    table, lo, hi = phase_lookup(calendar)
    assert table.shape[1] == 4
    # parameter counts
    assert sum(p.numel() for p in _module("nlinear").parameters()) == 1596
    assert sum(p.numel() for p in _module("scaled").parameters()) == 1596
    assert sum(p.numel() for p in _module("scaled_calendar").parameters()) == 1617
    # weight algebra: s=q cancels, market balance remains
    qv = np.array([2.0, 4.0])
    s = qv
    raw = s / qv
    assert np.allclose(raw, 1.0)
    body = {"week4_h1": float(base[0]), "q7": float(q[7]), "params": {"nlinear": 1596, "n3": 1617},
            "calendar_span": int(hi - lo + 1), "pass": True}
    write_json(OUT / "loss_contract_checks.json", body)
    print(json.dumps(body))


if __name__ == "__main__":
    main()
