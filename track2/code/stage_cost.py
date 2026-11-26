"""Per-1,000-task cost along the registered upgrade path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, TRACK1, write_json

GPU_USD_PER_HOUR = 0.53
CPU_USD_PER_VCPU_HOUR = 0.04
PRICE_DATE = "2026-10-05"


def _walls() -> dict[str, float]:
    rows = [json.loads(line) for line in (TRACK1 / "compute_benchmark" / "benchmark.jsonl").read_text().splitlines() if line.strip()]
    walls = {}
    for model in ("granite_patchtst_fm", "chronos_2", "timesfm_3"):
        sample = [row["wall_seconds"] for row in rows if row.get("model") == model and not row.get("warmup") and row.get("returncode") == 0]
        walls[model] = float(np.median(sample))
    return walls


def main() -> None:
    walls = _walls()
    # One origin batch covers 29 series. Outer Macau uses 27 of them.
    seconds_per_task = {model: wall / 29.0 for model, wall in walls.items()}
    gpu_usd_per_task = {model: seconds / 3600.0 * GPU_USD_PER_HOUR for model, seconds in seconds_per_task.items()}
    m1 = gpu_usd_per_task["granite_patchtst_fm"]
    trio = sum(gpu_usd_per_task.values())
    # ETS is a one-thread CPU model. Use the registered 0.04 USD/vCPU-hour and the
    # median fit+infer time is not in this GPU file; Track 1 froze ETS fit at about 0.5s.
    ets = 0.508 / 3600.0 * CPU_USD_PER_VCPU_HOUR
    c1 = ets * 3
    # The frozen weights are 1, 0, 0, so the selected point forecast does not need M2 or M3.
    # The coordinator still runs all three members because fallback may use them.
    scores = pd.read_csv(RESULTS / "evaluation" / "score_summary.csv")
    macau = scores[(scores["destination"] == "Macau") & (scores["history"] == "all") & (scores["split"] == "outer_backtest")]
    mase = dict(zip(macau["experiment"], macau["mase"]))
    # ETS accuracy is not a Track 2 experiment id. Read it from the Track 1 main table.
    main_table = pd.read_csv(Path(__file__).resolve().parents[2] / "track1" / "artifacts" / "scores" / "outer_main_table" / "main_table.csv")
    ets_mase = float(main_table.loc[main_table["model"] == "ets", "Macau_MASE7"].iloc[0])
    naive_mase = float(main_table.loc[main_table["model"] == "seasonal_naive", "Macau_MASE7"].iloc[0])
    named = {
        "seasonal_naive": naive_mase,
        "ETS03": ets_mase,
        "T2-C1": mase["T2-C1"],
        "M1": mase["T2-01"],
        "T2-C2": mase["T2-C2"],
        "frozen_ensemble": mase["T2-04"],
        "T2-05": mase["T2-05"],
        "T2-A1": mase["T2-04"],
        "T2-A2": mase["T2-04"],
    }
    costs = {
        "seasonal_naive": None,
        "ETS03": ets,
        "T2-C1": c1,
        "M1": m1,
        "T2-C2": m1,
        "frozen_ensemble": m1,
        "T2-05": m1,
        "T2-A1": trio,
        "T2-A2": trio,
    }
    rows = []
    previous = None
    for name in named:
        cost = costs[name]
        row = {
            "system": name,
            "mase": named[name],
            "usd_per_task": cost,
            "usd_per_1000_tasks": None if cost is None else cost * 1000,
        }
        if previous is not None and cost is not None and costs[previous] is not None:
            row["delta_mase"] = named[name] - named[previous]
            row["delta_usd_per_1000"] = (cost - costs[previous]) * 1000
        elif previous is not None:
            row["delta_mase"] = named[name] - named[previous]
        rows.append(row)
        previous = name
    table = pd.DataFrame(rows)
    output = RESULTS / "evaluation"
    table.to_csv(output / "cost_chain.csv", index=False)
    write_json(output / "cost_inputs.json", {
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "cpu_usd_per_vcpu_hour": CPU_USD_PER_VCPU_HOUR,
        "price_date": PRICE_DATE,
        "median_origin_wall_seconds": walls,
        "usd_per_task": {model: gpu_usd_per_task[model] for model in gpu_usd_per_task},
        "note": "Frozen point forecast uses only M1 because the refit weights are 1, 0, 0. T2-A1 and T2-A2 still add all three member medians because fallback can call the other two. Seasonal naive has no isolated timing in the benchmark file. ETS fit time 0.508s is the Track 1 frozen B-group average. A2 API tokens are added after llm_calls.jsonl is complete.",
    })
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
