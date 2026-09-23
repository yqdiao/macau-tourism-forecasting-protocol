"""Load the frozen, JSON-compatible YAML forecasting calendar."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "configs" / "protocol.yaml"
SPLIT_ORDER = ("pandemic_stress", "recovery_tuning", "outer_backtest")
EXPECTED_COUNTS = (13, 12, 33)


@dataclass(frozen=True)
class ProtocolFold:
    split_name: str
    fold_id: str
    origin: date
    destinations: tuple[str, ...]


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    protocol_id: str
    horizon_days: int
    seasonal_period_days: int
    history_lengths: tuple[int | None, ...]
    folds: tuple[ProtocolFold, ...]


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> Protocol:
    path = Path(path)
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    horizon = config.get("horizon_days")
    if horizon != 28 or config.get("seasonal_period_days") != 7:
        raise ValueError("This study requires horizon=28 and seasonal_period=7")
    if config.get("main_history") != "all" or config.get("sensitivity_histories") != [180, 365]:
        raise ValueError("Expected all as main history and 180/365 as sensitivity histories")
    if config.get("primary_metric") != "task_MASE7_market_fold_macro":
        raise ValueError("Unknown primary metric")
    splits = config.get("splits", {})
    if set(splits) != set(SPLIT_ORDER):
        raise ValueError("Protocol must contain exactly A, B and outer splits")
    folds: list[ProtocolFold] = []
    for split_name, expected_count, prefix in zip(SPLIT_ORDER, EXPECTED_COUNTS, ("A", "B", "O")):
        block = splits[split_name]
        destinations = tuple(block["destinations"])
        expected_destinations = ("Macau", "HongKong") if prefix == "O" else ("Macau",)
        if destinations != expected_destinations or block["use_for_tuning"] != (prefix == "B"):
            raise ValueError(f"{split_name}: destinations or tuning permission changed")
        entries = block["folds"]
        if len(entries) != expected_count:
            raise ValueError(f"{split_name}: expected {expected_count} folds")
        previous: date | None = None
        for number, entry in enumerate(entries, 1):
            fold_id = f"{prefix}{number:02d}"
            origin = date.fromisoformat(entry["origin"])
            if entry["id"] != fold_id or (previous is not None and origin <= previous):
                raise ValueError(f"{split_name}: invalid fold id or date order")
            if prefix in ("B", "O") and previous is not None and (origin - previous).days != 28:
                raise ValueError(f"{split_name}: expected a 28-day step")
            folds.append(ProtocolFold(split_name, fold_id, origin, destinations))
            previous = origin
    if (folds[0].origin, folds[12].origin, folds[13].origin, folds[24].origin,
            folds[25].origin, folds[-1].origin) != (
            date(2020, 1, 21), date(2022, 12, 10), date(2023, 1, 7),
            date(2023, 11, 11), date(2023, 12, 31), date(2026, 6, 14)):
        raise ValueError("Protocol boundary dates changed")
    if folds[24].origin + timedelta(days=horizon) >= folds[25].origin:
        raise ValueError("Tuning targets must finish before the first outer origin")
    if folds[-1].origin + timedelta(days=horizon) > date(2026, 7, 31):
        raise ValueError("Outer target window exceeds the observed data end")
    return Protocol(path, hashlib.sha256(raw).hexdigest(), config["protocol_id"],
                    horizon, 7, (180, 365, None), tuple(folds))
