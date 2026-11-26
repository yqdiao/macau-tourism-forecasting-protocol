#!/usr/bin/env python3
"""R3 command line. Example: python track1/r3/cli.py discover"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (  # noqa: E402
    B_FOLDS, O_FOLDS, PROBE_FOLDS, discover, load_origins, reproduce_baselines, run_jobs, score_stored, specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory R3 light-model study")
    parser.add_argument("stage", choices=["discover", "reproduce-baselines", "smoke", "probe", "develop-v1",
                                          "score-v1", "help-stages"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "help-stages":
        print("discover reproduce-baselines smoke probe develop-v1 score-v1")
        return
    if args.stage == "discover":
        print(discover()["r3_code_hash"])
        return
    bundles = load_origins()
    if args.stage == "reproduce-baselines":
        print(reproduce_baselines(bundles).to_string(index=False))
        return
    if args.stage == "smoke":
        run_jobs(bundles, ["R3V1_NP", "R3V1_L"], ["B01"], smoke=True)
        return
    if args.stage == "probe":
        run_jobs(bundles, ["R3V1_NP", "R3V1_NR", "R3V1_G", "R3V1_L"], PROBE_FOLDS, smoke=False)
        return
    if args.stage == "develop-v1":
        run_jobs(bundles, ["R3V1_NP", "R3V1_NR", "R3V1_G", "R3V1_L"], B_FOLDS, smoke=False)
        return
    if args.stage == "score-v1":
        metrics = score_stored(bundles, ["R3V1_NP", "R3V1_NR", "R3V1_G", "R3V1_L"], "recovery_tuning")
        from score import macro_mase
        for model, part in metrics.groupby("model_id"):
            print(model, macro_mase(part))
        return
    raise SystemExit(args.stage)


if __name__ == "__main__":
    main()
