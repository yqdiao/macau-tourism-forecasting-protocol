"""Prepare sanitized data metadata and physically separated backtest artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import (
    HONG_KONG_WORKBOOK,
    HISTORY_LENGTHS,
    MACAU_DIR,
    load_all,
    write_data_snapshot,
    write_task_artifacts,
)


def _parse_history(raw: str) -> tuple[int | None, ...]:
    result = tuple(
        None if token.strip() == "all" else int(token.strip())
        for token in raw.split(",")
        if token.strip()
    )
    if not result or any(value is not None and value < 1 for value in result):
        raise ValueError("history must contain positive days or all")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build task_inputs.parquet and separately protected targets.parquet"
    )
    parser.add_argument("--macau-dir", type=Path, default=MACAU_DIR)
    parser.add_argument("--hong-kong-workbook", type=Path, default=HONG_KONG_WORKBOOK)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--stride-days", type=int, default=28)
    parser.add_argument("--history", default="180,365,all")
    parser.add_argument("--destinations", default="Macau,HongKong")
    args = parser.parse_args()
    if args.stride_days < 1:
        raise ValueError("stride-days must be positive")
    history = _parse_history(args.history)
    destinations = tuple(
        token.strip() for token in args.destinations.split(",") if token.strip()
    )
    write_data_snapshot(args.output, args.macau_dir, args.hong_kong_workbook)
    series = load_all(args.macau_dir, args.hong_kong_workbook)
    manifest = write_task_artifacts(
        series,
        args.output,
        stride_days=args.stride_days,
        history_lengths=history,
        destinations=destinations,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
