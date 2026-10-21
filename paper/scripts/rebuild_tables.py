#!/usr/bin/env python3
"""Validate released evidence and rebuild summary tables using the standard library.

This aggregates existing task scores. It does not generate forecasts or refit models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).resolve().parents[1]


def read(name):
    with (PAPER / "data" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value not in ("", None, "nan", "NaN") else math.nan


def avg(values):
    finite = [number(v) for v in values if math.isfinite(number(v))]
    return mean(finite) if finite else math.nan


def groups(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def market_mean(rows, metric, market_field):
    return avg(avg(x[metric] for x in g) for g in groups(rows, [market_field]).values())


def same(actual, expected, label):
    a, b = number(actual), number(expected)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if not math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8):
        raise AssertionError(f"{label}: rebuilt {a!r}, released {b!r}")
    return abs(a - b)


def save(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "rebuilt")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((PAPER / "evidence_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        p = PAPER / artifact["path"]
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == artifact["sha256"], f"Checksum mismatch: {artifact['path']}"
    checks = {"checksum_files": len(manifest["artifacts"]), "source_level": "released derived scores"}

    main_rows = read("main_model_market_metrics.csv")
    assert len(main_rows) == 405
    assert len(groups(main_rows, ["model", "market"])) == 405
    main_summary = []
    for (model,), g in sorted(groups(main_rows, ["model"]).items()):
        assert len(g) == 27 and all(int(x["n_tasks"]) == 33 for x in g)
        row = {"model": model, "tasks": 891, "markets": 27, "origins": 33,
               "history": "all", "horizon_days": 28,
               "selection_identity": g[0]["selection_identity"],
               "replicate_aggregation": g[0]["replicate_aggregation"]}
        row.update({metric: avg(x[metric] for x in g) for metric in ["mase", "mae", "rmse"]})
        main_summary.append(row)
    assert len(main_summary) == 15
    released_main = {r["model"]: r for r in read("main_accuracy_summary.csv")}
    for r in main_summary:
        for metric in ["mase", "mae", "rmse"]:
            same(r[metric], released_main[r["model"]][metric], f"main {r['model']} {metric}")
    save(out / "main_accuracy_summary.csv", main_summary)
    checks["main_configurations"] = len(main_summary)
    checks["main_market_rows"] = len(main_rows)

    r3 = read("r3_task_metrics.csv")
    assert len(groups(r3, ["model_id", "split", "task_id"])) == len(r3)
    r3_expected = {(r["model_id"], r["split"]): r for r in read("r3_summary.csv")}
    r3_out, r3_delta = [], 0.0
    for key, g in sorted(groups(r3, ["model_id", "split"]).items()):
        row = {"model_id": key[0], "split": key[1], "selection_identity": "post_outer_exploratory",
               "n_tasks": len(g), "n_markets": len({x["market"] for x in g}),
               "n_origins": len({x["origin"] for x in g})}
        assert row["n_tasks"] == {"outer_backtest": 891, "pandemic_stress": 351, "recovery_tuning": 324}[key[1]]
        for metric in ["mae", "mase", "rmse"]:
            row[metric] = market_mean(g, metric, "market")
            r3_delta = max(r3_delta, same(row[metric], r3_expected[key][metric], f"R3 {key} {metric}"))
        for count in ["n_tasks", "n_markets", "n_origins"]:
            same(row[count], r3_expected[key][count], f"R3 {key} {count}")
        r3_out.append(row)
    assert len(r3_out) == len(r3_expected) == 42
    save(out / "r3_summary.csv", r3_out)
    checks.update(r3_task_rows=len(r3), r3_summaries=len(r3_out), r3_max_absolute_metric_difference=r3_delta)

    llm = read("llm_task_metrics.csv")
    assert len(groups(llm, ["experiment", "strategy", "task_id"])) == len(llm)
    llm_groups = groups(llm, ["experiment", "strategy"])
    direct = llm_groups[("M5_4B_thinking180", "Qwen3_4B_thinking")]
    extreme_task = max(direct, key=lambda x: number(x["max_abs_error"]))["task_id"]
    assert extreme_task == "a66eb1bfd3810b8dd562c918"
    llm_out, llm_delta = [], 0.0
    for expected in read("llm_summary.csv"):
        key = (expected["experiment"], expected["strategy"])
        g = llm_groups[key]
        scope = expected["scope"]
        if scope == "zero_native_scale":
            g = [x for x in g if number(x["q_evaluation"]) == 0]
        elif scope == "positive_native_scale":
            g = [x for x in g if number(x["q_evaluation"]) > 0]
        elif scope == "matched_890_excluding_extreme_task_sensitivity":
            g = [x for x in g if x["task_id"] != extreme_task]
        else:
            assert scope == "full"
        row = {"experiment": key[0], "strategy": key[1], "scope": scope,
               "tasks": len(g), "markets": len({x["market_id"] for x in g}),
               "origins": len({x["origin"] for x in g}),
               "valid_mase_tasks": sum(math.isfinite(number(x["mase_evaluation"])) for x in g),
               "zero_scale_tasks": sum(number(x["q_evaluation"]) == 0 for x in g),
               "mase_evaluation_history": expected["mase_evaluation_history"]}
        for count in ["tasks", "markets", "origins", "valid_mase_tasks", "zero_scale_tasks"]:
            same(row[count], expected[count], f"LLM {key} {scope} {count}")
        for metric in ["mae", "rmse", "mase_evaluation", "mase_all_reference", "wape"]:
            for suffix in ["market_equal", "task_equal"]:
                col = metric + "_" + suffix
                row[col] = market_mean(g, metric, "market_id") if suffix == "market_equal" else avg(x[metric] for x in g)
                llm_delta = max(llm_delta, same(row[col], expected[col], f"LLM {key} {scope} {col}"))
        llm_out.append(row)
    save(out / "llm_summary.csv", llm_out)
    checks.update(llm_task_rows=len(llm), llm_summaries=len(llm_out),
                  llm_max_absolute_metric_difference=llm_delta, extreme_task_excluded_only_in_sensitivity=extreme_task)

    m2 = {r["strategy"]: r for r in llm_out if r["experiment"] == "M2_all_bridge" and r["scope"] == "full"}
    delta = m2["P3-R"]["mase_evaluation_market_equal"] - m2["TSFM_equal_weight"]["mase_evaluation_market_equal"]
    pair = next(r for r in read("llm_paired_differences.csv") if r["comparison"] == "M2_all_bridge: P3-R minus TSFM_equal_weight" and r["metric"] == "mase_evaluation")
    same(delta, pair["difference_first_minus_second"], "M2 headline paired difference")
    assert number(pair["ci95_lower"]) < 0 < number(pair["ci95_upper"])
    checks.update(m2_mase_difference=delta, m2_mase_relative_reduction_percent=-100 * delta / m2["TSFM_equal_weight"]["mase_evaluation_market_equal"],
                  bootstrap_intervals="stored source audit intervals; not rerun by this script")

    for r in read("llm_call_resources.csv"):
        rebuilt = number(r["all_sent_seconds"]) / number(r["deliveries"]) * 1000 / 3600
        same(rebuilt, r["additional_call_hours_per_1000_deliveries"], f"request duration {r['experiment']} {r['strategy']}")
    checks["request_hour_coefficients_verified"] = True

    totals = read("market_totals_27.csv")
    assert len(totals) == 27 and [int(x["total_rank"]) for x in totals] == list(range(1, 28))
    top = {x["market"] for x in totals if x["selected_top10"] == "True"}
    assert len(top) == 10
    top_rows = read("top10_model_market_metrics.csv")
    assert len(top_rows) == 150 and {x["market"] for x in top_rows} == top
    checks["top10_share_of_27_market_total"] = sum(int(x["total_arrivals"]) for x in totals if x["market"] in top) / sum(int(x["total_arrivals"]) for x in totals)
    checks["status"] = "passed"
    (out / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    report = ["# Evidence validation", "", "Status: passed.", "",
              f"- {checks['checksum_files']} released files match the manifest.",
              "- 15 main configurations are reconstructed from 405 market rows.",
              f"- {len(r3_out)} R3 summaries are reconstructed from {len(r3)} task/configuration rows.",
              f"- {len(llm_out)} LLM summaries are reconstructed from {len(llm)} task/strategy rows.",
              "- Full 4B outputs, matched-task sensitivity, and undefined-MASE pressure cases remain distinct.",
              "- Request-duration coefficients match their registered numerators and delivery counts.",
              "", "These checks validate aggregation of supplied derived scores. They do not rerun model inference, rederive raw daily errors, or regenerate the stored bootstrap intervals.", ""]
    (out / "validation.md").write_text("\n".join(report))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
