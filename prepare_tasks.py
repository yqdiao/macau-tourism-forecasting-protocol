"""Prepare sanitized data metadata and physically separated backtest artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import (
    HONG_KONG_WORKBOOK,
    MACAU_DIR,
    load_all,
    write_data_snapshot,
    write_task_artifacts,
)
from protocol_config import DEFAULT_PROTOCOL, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build task_inputs.parquet and separately protected targets.parquet"
    )
    parser.add_argument("--macau-dir", type=Path, default=MACAU_DIR)
    parser.add_argument("--hong-kong-workbook", type=Path, default=HONG_KONG_WORKBOOK)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    write_data_snapshot(args.output, args.macau_dir, args.hong_kong_workbook, protocol)
    series = load_all(args.macau_dir, args.hong_kong_workbook)
    manifest = write_task_artifacts(
        series,
        args.output,
        protocol=protocol,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
