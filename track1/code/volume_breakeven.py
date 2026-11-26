"""Per-task cost and volume breakeven from the registered plan and the isolated benchmark.

Reads outer targets only to form task MAE. Hong Kong tasks are reported, not fitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO / "track1/configs/volume_breakeven_plan.yaml"
BENCH = REPO / "track1/results/compute_benchmark/benchmark.jsonl"
METRICS = REPO / "track1/evaluation/outer_main_table/task_metrics.parquet"
METRICS_SEED = REPO / "track1/evaluation/outer_main_table/task_metrics_by_seed.parquet"
INPUTS = REPO / "shared/data/task_inputs.parquet"
TARGETS = REPO / "shared/data/targets.parquet"
FROZEN = REPO / "track1/results/foundation_outer_frozen"
JOURNAL = Path("/root/DYQ/run-logs/b_search_nodes.jsonl")
OUTPUT = REPO / "track1/evaluation/volume_breakeven"

FOUNDATIONS = ["granite_patchtst_fm", "chronos_2", "timesfm_3", "moirai", "lag_llama", "patchtst", "informer"]
TRAINED = {"patchtst": "PT06", "informer": "INF07"}
C_ERR = np.logspace(-3, 2, 11)
BOOTSTRAP = 2000
SEED = 20261005


def _load_plan() -> dict:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "registered_before_analysis":
        raise SystemExit("volume plan is not registered")
    return plan


def costs(plan: dict) -> pd.DataFrame:
    """Mean isolated cost. Warmup runs are excluded."""
    gpu_hour = plan["pricing"]["primary"]["gpu"]["usd_per_hour"]
    cpu_hour = plan["pricing"]["primary"]["cpu"]["usd_per_vcpu_hour"]
    rows = []
    for record in map(json.loads, BENCH.read_text(encoding="utf-8").splitlines()):
        if record["warmup"]:
            continue
        if record["kind"] == "gpu":
            seconds = record["wall_seconds"] / record["tasks"]
            rows.append({"model": record["model"], "device": "gpu", "instance_seconds": seconds,
                         "allocated_vcpu_seconds": np.nan, "consumed_cpu_seconds": record["child_cpu_seconds"] / record["tasks"],
                         "peak_gpu_mb": record.get("peak_gpu_allocated_mb"), "peak_rss_mb": record.get("peak_rss_mb"),
                         "energy_wh_per_origin": record.get("energy_wh")})
        else:
            threads = record["threads_per_task"]
            for task in record["task_results"]:
                rows.append({"model": record["model"], "device": "cpu", "instance_seconds": np.nan,
                             "allocated_vcpu_seconds": task["wall_seconds"] * threads,
                             "consumed_cpu_seconds": task["cpu_seconds"],
                             "peak_gpu_mb": np.nan, "peak_rss_mb": task["rss_mb"], "energy_wh_per_origin": np.nan})
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "device"], as_index=False).agg(
        runs=("instance_seconds", "size"),
        instance_seconds=("instance_seconds", "mean"),
        allocated_vcpu_seconds=("allocated_vcpu_seconds", "mean"),
        consumed_cpu_seconds=("consumed_cpu_seconds", "mean"),
        peak_gpu_mb=("peak_gpu_mb", "median"),
        peak_rss_mb=("peak_rss_mb", "median"))
    summary["usd_per_task"] = np.where(
        summary["device"] == "gpu",
        summary["instance_seconds"] * gpu_hour / 3600,
        summary["allocated_vcpu_seconds"] * cpu_hour / 3600)
    summary["gpu_hours_per_1000"] = np.where(summary["device"] == "gpu", summary["instance_seconds"] * 1000 / 3600, 0.0)
    summary["vcpu_hours_per_1000"] = summary["allocated_vcpu_seconds"].fillna(0) * 1000 / 3600
    return summary


def _seed_mean_mae(model: str, candidate: str) -> pd.DataFrame:
    """MAE of the prediction averaged across the registered seeds."""
    frames = []
    for path in (FROZEN / model / candidate).glob("seed*/O*_*/predictions.parquet"):
        frames.append(pd.read_parquet(path, columns=["task_id", "horizon_day", "prediction", "destination"]))
    prediction = pd.concat(frames).groupby(["task_id", "horizon_day", "destination"], as_index=False)["prediction"].mean()
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length"])
    targets = targets[(targets["split_name"] == "outer_backtest") & (targets["history_length"] == "all")]
    merged = prediction.merge(targets, on=["task_id", "horizon_day"])
    if len(merged) != len(prediction):
        raise ValueError(f"{model}: seed-mean predictions do not match the outer targets")
    merged["abs_error"] = (merged["prediction"] - merged["actual"]).abs()
    return merged.groupby(["task_id", "destination"], as_index=False)["abs_error"].mean().rename(columns={"abs_error": "MAE"})


def task_table() -> pd.DataFrame:
    metrics = pd.read_parquet(METRICS, columns=["model", "task_id", "destination", "market", "fold_id", "MAE", "MASE7"])
    trained = []
    for model, candidate in TRAINED.items():
        mae = _seed_mean_mae(model, candidate)
        base = metrics.loc[metrics["model"] == model, ["task_id", "destination", "market", "fold_id", "MASE7"]]
        # MASE7 in the published table is the mean of per-seed MASE7; keep it for the relative metric.
        merged = base.merge(mae, on=["task_id", "destination"])
        merged.insert(0, "model", model)
        trained.append(merged)
    single = metrics.loc[~metrics["model"].isin(TRAINED)]
    tasks = pd.concat([single, pd.concat(trained)], ignore_index=True)

    inputs = pd.read_parquet(INPUTS, columns=["task_id", "history", "history_length", "split_name"])
    inputs = inputs[(inputs["split_name"] == "outer_backtest") & (inputs["history_length"] == "all")]
    volumes = []
    for task_id, history in zip(inputs["task_id"], inputs["history"]):
        values = np.asarray(history, dtype=float)
        volumes.append({"task_id": task_id, "V28": float(values[-28:].sum()),
                        "V365": float(28 * values[-365:].mean()) if len(values) >= 365 else np.nan})
    return tasks.merge(pd.DataFrame(volumes), on="task_id")


def _fit(log_volume: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
    positive = delta > 0
    if positive.sum() < 5:
        return np.nan, np.nan
    slope, intercept = np.polyfit(log_volume[positive], np.log(delta[positive]), 1)
    return float(intercept), float(slope)


def _predicted(intercept: float, slope: float, volume: np.ndarray) -> np.ndarray:
    return np.exp(intercept + slope * np.log(volume))


def _threshold(intercept: float, slope: float, extra_cost: float, c_err: float) -> float:
    """Volume where 28 * predicted positive delta * c_err equals extra cost. NaN if undefined."""
    if not np.isfinite(intercept) or extra_cost <= 0 or c_err <= 0 or slope <= 0:
        return np.nan
    scale = extra_cost / (28 * c_err * np.exp(intercept))
    if scale <= 0:
        return np.nan
    return float(np.exp(np.log(scale) / slope))


def _bootstrap(log_volume: np.ndarray, delta: np.ndarray, market_code: np.ndarray, extra: float,
               rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Cluster bootstrap. A market drawn twice contributes its tasks twice."""
    codes = np.unique(market_code)
    slices = [np.flatnonzero(market_code == code) for code in codes]
    slopes = np.empty(BOOTSTRAP)
    thresholds = {c_err: np.empty(BOOTSTRAP) for c_err in C_ERR}
    n_markets = len(codes)
    for draw in range(BOOTSTRAP):
        chosen = rng.integers(0, n_markets, n_markets)
        index = np.concatenate([slices[item] for item in chosen])
        intercept, slope = _fit(log_volume[index], delta[index])
        slopes[draw] = slope
        for c_err in C_ERR:
            thresholds[c_err][draw] = _threshold(intercept, slope, extra, c_err)
    return slopes, thresholds


def analyse_pair(tasks: pd.DataFrame, foundation: str, baseline: str, cost: dict, rng: np.random.Generator) -> dict:
    left = tasks.loc[tasks["model"] == foundation, ["task_id", "destination", "market", "fold_id", "MAE", "MASE7", "V28", "V365"]]
    right = tasks.loc[tasks["model"] == baseline, ["task_id", "MAE", "MASE7"]].rename(
        columns={"MAE": "MAE_base", "MASE7": "MASE7_base"})
    paired = left.merge(right, on="task_id")
    paired["delta_MAE"] = paired["MAE_base"] - paired["MAE"]
    paired["mase_ratio"] = paired["MASE7"] / paired["MASE7_base"]
    extra = cost[foundation] - cost[baseline]
    macau = paired.loc[paired["destination"] == "Macau"].copy()
    hong_kong = paired.loc[paired["destination"] != "Macau"]
    if len(macau) != 891:
        raise ValueError(f"{foundation} vs {baseline}: expected 891 Macau tasks, got {len(macau)}")

    log_v28 = np.log(macau["V28"].to_numpy())
    delta = macau["delta_MAE"].to_numpy()
    codes, market_code = np.unique(macau["market"].to_numpy(), return_inverse=True)
    intercept, slope = _fit(log_v28, delta)
    boot_slope, boot_v = _bootstrap(log_v28, delta, market_code, extra, rng)
    usable = macau["V365"].notna()
    intercept_365, slope_365 = _fit(np.log(macau.loc[usable, "V365"].to_numpy()), macau.loc[usable, "delta_MAE"].to_numpy())

    macro_delta = float(macau.groupby("market")["delta_MAE"].mean().mean())
    result = {
        "foundation": foundation, "baseline": baseline,
        "macau_tasks": int(len(macau)),
        "share_foundation_worse_or_equal": float((macau["delta_MAE"] <= 0).mean()),
        "macro_delta_MAE": macro_delta,
        "macro_mase_ratio": float(macau.groupby("market")["mase_ratio"].mean().mean()),
        "cost_foundation_usd": cost[foundation], "cost_baseline_usd": cost[baseline],
        "extra_cost_usd": extra,
        "loglog_intercept": intercept, "loglog_slope": slope,
        "slope_ci95": [float(np.nanpercentile(boot_slope, 2.5)), float(np.nanpercentile(boot_slope, 97.5))],
        "v365_tasks": int(usable.sum()), "v365_loglog_slope": slope_365, "v365_loglog_intercept": intercept_365,
        "dominant": bool(extra <= 0 and macro_delta > 0),
        "hong_kong_tasks": int(len(hong_kong)),
        "hong_kong_mean_delta_MAE": float(hong_kong["delta_MAE"].mean()) if len(hong_kong) else None,
        "hong_kong_median_V28": float(hong_kong["V28"].median()) if len(hong_kong) else None,
    }
    grid = []
    for c_err in C_ERR:
        draws = boot_v[c_err]
        grid.append({"c_err_usd": float(c_err),
                     "V28_star": _threshold(intercept, slope, extra, c_err),
                     "V28_star_ci95_low": float(np.nanpercentile(draws, 2.5)),
                     "V28_star_ci95_high": float(np.nanpercentile(draws, 97.5)),
                     "defined_share": float(np.isfinite(draws).mean()),
                     "V365_star": _threshold(intercept_365, slope_365, extra, c_err)})
    quintiles = []
    macau["quintile"] = pd.qcut(macau["V28"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    for label, frame in macau.groupby("quintile", observed=True):
        mean_delta = float(frame["delta_MAE"].mean())
        quintiles.append({
            "quintile": str(label), "tasks": int(len(frame)),
            "V28_min": float(frame["V28"].min()), "V28_median": float(frame["V28"].median()),
            "V28_max": float(frame["V28"].max()), "mean_delta_MAE": mean_delta,
            "share_not_improved": float((frame["delta_MAE"] <= 0).mean()),
            "c_err_star_usd": (extra / (28 * mean_delta)) if mean_delta > 0 else None,
        })
    return result, pd.DataFrame(grid), pd.DataFrame(quintiles), macau


def tuning_cost(plan: dict) -> pd.DataFrame:
    """Contended per-job wall time of the whole B search. Not used in the breakeven."""
    gpu_hour = plan["pricing"]["primary"]["gpu"]["usd_per_hour"]
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in map(json.loads, JOURNAL.read_text(encoding="utf-8").splitlines()):
        if record.get("event") == "job_done":
            totals[record["model"]] = totals.get(record["model"], 0.0) + record["elapsed_seconds"]
            counts[record["model"]] = counts.get(record["model"], 0) + 1
    parallelism = {"patchtst": 8, "informer": 12}
    rows = []
    for model, seconds in totals.items():
        rows.append({"model": model, "jobs": counts[model],
                     "sum_job_wall_seconds": seconds,
                     "approximate_exclusive_gpu_hours": seconds / parallelism[model] / 3600,
                     "approximate_usd": seconds / parallelism[model] / 3600 * gpu_hour,
                     "note": "jobs shared the GPUs; exclusive hours divide total job wall time by the launch parallelism"})
    return pd.DataFrame(rows)


def price_sensitivity(summary: pd.DataFrame, plan: dict, primary_extra_seconds: dict) -> pd.DataFrame:
    cpu_hour = plan["pricing"]["primary"]["cpu"]["usd_per_vcpu_hour"]
    alt_cpu = plan["pricing"]["primary"]["cpu"]["alternative"]["usd_per_vcpu_hour"]
    prices = {"primary_secure_0.53": plan["pricing"]["primary"]["gpu"]["usd_per_hour"],
              "community_0.33": plan["pricing"]["sensitivity"][0]["usd_per_hour"],
              "lambda_1.09": plan["pricing"]["sensitivity"][1]["usd_per_hour"]}
    granite = float(summary.loc[summary["model"] == "granite_patchtst_fm", "instance_seconds"].iloc[0])
    ets = float(summary.loc[summary["model"] == "ets", "allocated_vcpu_seconds"].iloc[0])
    rows = []
    for name, gpu_hour in prices.items():
        for cpu_name, cpu in (("quoted_0.04", cpu_hour), ("bundled_0.0589", alt_cpu)):
            extra = granite * gpu_hour / 3600 - ets * cpu / 3600
            rows.append({"price": name, "cpu_price": cpu_name, "extra_cost_usd": extra,
                         "granite_usd": granite * gpu_hour / 3600, "ets_usd": ets * cpu / 3600})
    return pd.DataFrame(rows)


def price_ratio(summary: pd.DataFrame, quintiles: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Largest GPU-hour / CPU-core-hour price ratio at which the quintile's mean error reduction still pays."""
    cpu_hour = plan["pricing"]["primary"]["cpu"]["usd_per_vcpu_hour"]
    seconds = summary.set_index("model")
    rows = []
    for record in quintiles.itertuples(index=False):
        foundation, baseline = record.pair.rsplit("_vs_", 1)
        gpu_seconds = float(seconds.loc[foundation, "instance_seconds"])
        cpu_seconds = float(seconds.loc[baseline, "allocated_vcpu_seconds"])
        equal_ratio = cpu_seconds / gpu_seconds
        for c_err in (0.001, 1.0, 100.0):
            if record.mean_delta_MAE > 0:
                ratio = (28 * record.mean_delta_MAE * c_err * 3600 / cpu_hour + cpu_seconds) / gpu_seconds
            else:
                ratio = np.nan
            rows.append({"pair": record.pair, "quintile": record.quintile, "c_err_usd": c_err,
                         "max_gpu_cpu_price_ratio": ratio, "cost_equal_ratio": equal_ratio})
    return pd.DataFrame(rows)


def main() -> None:
    plan = _load_plan()
    summary = costs(plan)
    tasks = task_table()
    cost = dict(zip(summary["model"], summary["usd_per_task"]))
    missing = [model for model in FOUNDATIONS + ["ets", "seasonal_naive"] if model not in cost]
    if missing:
        raise SystemExit(f"missing benchmark cost: {missing}")
    if not (tasks.groupby("model")["task_id"].nunique() == 957).all():
        raise SystemExit("task coverage is not 957 per model")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "cost_per_task.csv", index=False)
    rng = np.random.default_rng(SEED)
    headers, grids, quintiles = [], [], []
    primary = plan["comparisons"]["primary_pair"]
    pairs = [(primary["foundation"], primary["baseline"]), (primary["foundation"], plan["comparisons"]["cheap_reference"])]
    pairs += [(model, baseline) for model in FOUNDATIONS if model != primary["foundation"]
              for baseline in (primary["baseline"], plan["comparisons"]["cheap_reference"])]
    for foundation, baseline in pairs:
        header, grid, quintile, _ = analyse_pair(tasks, foundation, baseline, cost, rng)
        headers.append(header)
        grid.insert(0, "pair", f"{foundation}_vs_{baseline}")
        quintile.insert(0, "pair", f"{foundation}_vs_{baseline}")
        grids.append(grid)
        quintiles.append(quintile)
    pd.DataFrame(headers).to_csv(OUTPUT / "pair_summary.csv", index=False)
    pd.concat(grids).to_csv(OUTPUT / "volume_threshold_grid.csv", index=False)
    pd.concat(quintiles).to_csv(OUTPUT / "volume_quintiles.csv", index=False)
    tuning_cost(plan).to_csv(OUTPUT / "tuning_cost.csv", index=False)
    price_sensitivity(summary, plan, {}).to_csv(OUTPUT / "price_sensitivity.csv", index=False)
    price_ratio(summary, pd.concat(quintiles), plan).to_csv(OUTPUT / "price_ratio.csv", index=False)
    status = {"plan_status": plan["status"], "registered_at_utc": plan["registered_at_utc"],
              "bootstrap": BOOTSTRAP, "seed": SEED, "c_err_grid": C_ERR.tolist(),
              "curve": "log(delta_MAE) on log(V28), Macau tasks with delta_MAE>0; V* is conditional on an improvement",
              "multi_seed_mae": "MAE of the mean prediction across seeds 11, 29 and 42",
              "targets": "outer_backtest history=all", "created_at_utc": pd.Timestamp.utcnow().isoformat()}
    (OUTPUT / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(summary[["model", "device", "instance_seconds", "allocated_vcpu_seconds", "usd_per_task",
                   "gpu_hours_per_1000", "vcpu_hours_per_1000"]].round(6).to_string(index=False))
    print(pd.DataFrame(headers)[["foundation", "baseline", "macro_delta_MAE", "share_foundation_worse_or_equal",
                                 "extra_cost_usd", "loglog_slope", "dominant"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
