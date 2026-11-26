"""Fill the RQ1 tables that the first scoring pass did not write.

Uses the saved task metrics, the isolated timing log and the existing
day-level predictions. It does not retrain any model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
OUT = REPO / "RQ1"
sys.path.insert(0, str(OUT))
import build_rq1 as base

BLOCK, RESAMPLES, SEED = base.BLOCK, base.RESAMPLES, base.SEED
LABEL = {c[0]: c[1] for c in base.CONFIGS}
CORE14 = [c[0] for c in base.CONFIGS[:14]]


def origin_series(task, config_id, column):
    part = task[task.config_id == config_id].sort_values(["market", "origin_date"])
    return part.groupby("origin_date", sort=True)[column].mean()


def task_win_rates(task):
    ets = task.loc[task.config_id == "ets", ["task_id", "mase"]].rename(columns={"mase": "mase_ets"})
    merged = task.merge(ets, on="task_id", how="inner")
    rows = []
    for config_id, group in merged.groupby("config_id"):
        compared = int(group.task_id.nunique())
        wins = int((group.mase < group.mase_ets).sum()) if config_id != "ets" else np.nan
        rows.append({"config_id": config_id, "model": LABEL.get(config_id, config_id),
                     "tasks_compared": compared, "task_wins_vs_ETS": wins,
                     "task_wins_vs_ETS_text": "" if config_id == "ets" else f"{wins}/{compared}"})
    return pd.DataFrame(rows)


def market_dm(task):
    strongest = "LL01"
    rows = []
    ets = task[task.config_id == "ets"][["market", "origin_date", "mase", "mape", "rmse"]]
    ref_s = task[task.config_id == strongest][["market", "origin_date", "mase"]]
    for config_id, group in task.groupby("config_id"):
        joined = group.merge(ets, on=["market", "origin_date"], suffixes=("", "_ets"))
        joined = joined.merge(ref_s, on=["market", "origin_date"], suffixes=("", "_strong"))
        for market, part in joined.groupby("market"):
            part = part.sort_values("origin_date")
            row = {"config_id": config_id, "model": LABEL.get(config_id, config_id), "market": market,
                   "origins": len(part), "MASE": part.mase.mean(), "MAPE_pct": part.mape.mean(),
                   "RMSE": part.rmse.mean()}
            if config_id != "ets":
                stat, p = base.dm_hln(part.mase.to_numpy(), part.mase_ets.to_numpy())
                row.update({"DM_vs_ETS": stat, "p_vs_ETS": p, "sig_vs_ETS": base.stars(p)})
            if config_id != strongest:
                stat, p = base.dm_hln(part.mase.to_numpy(), part.mase_strong.to_numpy())
                row.update({"DM_vs_LightGBM_calendar": stat, "p_vs_LightGBM_calendar": p,
                            "sig_vs_LightGBM_calendar": base.stars(p)})
            rows.append(row)
    return pd.DataFrame(rows)


def forest_and_pareto(task, timing):
    ets = origin_series(task, "ets", "mase")
    rng = np.random.default_rng(SEED)
    idx = base.block_indices(len(ets), BLOCK, RESAMPLES, rng)
    rows = []
    for config_id in task.config_id.unique():
        series = origin_series(task, config_id, "mase").reindex(ets.index)
        change = series / ets - 1
        lo, hi = base.ci(change.to_numpy(), idx)
        rows.append({"config_id": config_id, "model": LABEL.get(config_id, config_id),
                     "relative_change_vs_ETS": float(change.mean()),
                     "relative_change_ci95_low": lo, "relative_change_ci95_high": hi})
    forest = pd.DataFrame(rows).sort_values("relative_change_vs_ETS")
    return forest


def timing_detail():
    repeats = []
    for line in open(REPO / "track1/results/compute_benchmark/benchmark.jsonl"):
        row = json.loads(line)
        if row.get("warmup") or row.get("returncode") not in (0, None):
            continue
        if row.get("kind") == "gpu":
            tasks = row["tasks"]
            wall = row["wall_seconds"] / tasks
            infer = max(row["wall_seconds"] - (row.get("load_seconds") or 0.0), 0.0) / tasks
            cpu = row["child_cpu_seconds"] / tasks
        else:
            macau = [item for item in row["task_results"] if item["market"] not in {"Mainland visitors", "Other visitors"}]
            wall = float(np.mean([item["allocated_vcpu_seconds"] for item in macau]))
            infer = float(np.mean([item["inference_seconds"] for item in macau]))
            cpu = wall
        repeats.append({"model": row["model"], "device": row["kind"], "repeat": row["repeat"],
                        "seconds_per_vector_with_load": wall, "inference_seconds_per_vector": infer,
                        "cpu_seconds_per_vector": cpu})
    frame = pd.DataFrame(repeats)
    summary = frame.groupby(["model", "device"], as_index=False).agg(
        repeats=("repeat", "nunique"),
        seconds_mean=("seconds_per_vector_with_load", "mean"),
        seconds_sd=("seconds_per_vector_with_load", "std"),
        inference_seconds_mean=("inference_seconds_per_vector", "mean"),
        inference_seconds_sd=("inference_seconds_per_vector", "std"),
        cpu_seconds_mean=("cpu_seconds_per_vector", "mean"))
    summary["cv"] = summary.seconds_sd / summary.seconds_mean
    summary["inference_cv"] = summary.inference_seconds_sd / summary.inference_seconds_mean
    ets = float(summary.loc[summary.model == "ets", "cpu_seconds_mean"].iloc[0])
    summary["rho_star_vs_ETS"] = np.where(summary.device == "gpu", ets / summary.seconds_mean, np.nan)
    summary["rho_star_inference_only"] = np.where(summary.device == "gpu", ets / summary.inference_seconds_mean, np.nan)
    return summary


def price_curve(timing):
    gpu = timing[timing.device == "gpu"].dropna(subset=["rho_star_vs_ETS"])
    rows = []
    for rho in np.linspace(1, 30, 30):
        for row in gpu.itertuples(index=False):
            rows.append({"model": row.model, "rho": float(rho),
                         "cost_ratio_vs_ETS": float(rho / row.rho_star_vs_ETS),
                         "inference_only_cost_ratio_vs_ETS": float(rho / row.rho_star_inference_only)})
    return pd.DataFrame(rows)


def volume_and_breakeven(timing, task):
    quotes = [("cloud_on_demand", 0.53, 0.04), ("cloud_spot_or_community", 0.33, 0.04), ("owned_or_lambda", 1.09, 0.04)]
    mae = task.groupby(["config_id", "market"]).mae.mean().groupby("config_id").mean()
    ets_mae = float(mae["ets"])
    ets_seconds = float(timing.loc[timing.model == "ets", "cpu_seconds_mean"].iloc[0])
    rows, breakeven = [], []
    for row in timing.itertuples(index=False):
        seconds = row.cpu_seconds_mean if row.device == "cpu" else row.seconds_mean
        infer = row.inference_seconds_mean if row.device == "gpu" else row.cpu_seconds_mean
        for n in (27, 1000, 10000):
            annual = n * 365 * seconds / 3600
            annual_infer = n * 365 * infer / 3600
            item = {"model": row.model, "device": row.device, "series": n,
                    "annual_hours_with_load": annual, "annual_hours_inference_only": annual_infer}
            for name, gpu_price, cpu_price in quotes:
                price = gpu_price if row.device == "gpu" else cpu_price
                item[f"annual_usd_{name}"] = annual * price
                item[f"annual_usd_inference_only_{name}"] = annual_infer * price
            rows.append(item)
        if row.model in mae.index and row.device == "gpu":
            for name, gpu_price, cpu_price in quotes:
                delta_cost = 27 * 365 * (row.seconds_mean * gpu_price - ets_seconds * cpu_price) / 3600
                delta_people = (ets_mae - float(mae[row.model])) * 27 * 365
                breakeven.append({"model": row.model, "scenario": name, "series": 27,
                                  "annual_extra_cost_usd": delta_cost,
                                  "annual_absolute_error_reduction_people": delta_people,
                                  "usd_per_person_error_avoided": delta_cost / delta_people if delta_people > 0 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(breakeven)


def pareto(timing, task):
    mae_mase = task.groupby("config_id").mase.mean()
    # market-then-equal MASE is already one row per task; use the saved overall definition
    overall = task.groupby(["config_id", "market"]).mase.mean().groupby("config_id").mean()
    quotes = [("cloud_on_demand", 0.53, 0.04), ("cloud_spot_or_community", 0.33, 0.04), ("owned_or_lambda", 1.09, 0.04)]
    ets_seconds = float(timing.loc[timing.model == "ets", "cpu_seconds_mean"].iloc[0])
    ets_cost = {name: ets_seconds * cpu for name, _gpu, cpu in quotes}
    rows = []
    named = timing[timing.model.isin(overall.index)]
    for row in named.itertuples(index=False):
        for name, gpu_price, cpu_price in quotes:
            price = gpu_price if row.device == "gpu" else cpu_price
            seconds = row.seconds_mean if row.device == "gpu" else row.cpu_seconds_mean
            cost = seconds * price
            rows.append({"scenario": name, "model": row.model, "MASE": float(overall[row.model]),
                         "cost_ratio_vs_ETS": cost / ets_cost[name]})
    frame = pd.DataFrame(rows)
    flags = []
    for _scenario, part in frame.groupby("scenario"):
        for i, row in part.iterrows():
            dominated = ((part.MASE <= row.MASE) & (part.cost_ratio_vs_ETS <= row.cost_ratio_vs_ETS)
                         & ~((part.MASE == row.MASE) & (part.cost_ratio_vs_ETS == row.cost_ratio_vs_ETS))).any()
            flags.append((i, not dominated))
    frame["on_pareto_front"] = pd.Series(dict(flags))
    return frame


def day_losses(files, targets, qall, value_column):
    columns = ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day",
               "prediction", "q0.1", "q0.5", "q0.9", "seed", "candidate_id"]
    frame = base.read_many(files, columns)
    if "destination" in frame.columns:
        frame = frame[frame.destination == "Macau"]
    frame = frame.merge(targets, on=["task_id", "horizon_day"], how="inner")
    frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
    frame = frame.merge(qall, on=["market", "origin_date"], how="left")
    return frame


def pinball(pred, actual, tau):
    return np.where(actual >= pred, tau * (actual - pred), (1 - tau) * (pred - actual))


def summarise_pinball(frame, pred_column, tau, qall_column="q_all"):
    loss = pinball(frame[pred_column].to_numpy(float), frame.actual.to_numpy(float), tau)
    part = frame[["market", "fold_id", "origin_date"]].copy()
    part["loss"] = loss
    part["scaled"] = loss / frame[qall_column].to_numpy(float)
    task = part.groupby(["market", "fold_id", "origin_date"], sort=False).mean(numeric_only=True)
    market = task.groupby("market").mean()
    origin = task.groupby("origin_date").mean().sort_index()
    return float(market.loss.mean()), float(market.scaled.mean()), origin


def table5_comparisons(qall, targets_outer, targets_stress):
    panels = [("reopening_891", base.FOUNDATION5, REPO / "track1/results/foundation_outer_default", targets_outer),
              ("pandemic_351", base.STRESS3, REPO / "track1/results/sensitivity/pandemic_stress/all", targets_stress)]
    stored = {}
    for panel, models, root, targets in panels:
        for model in models:
            files = list(root.glob(f"*/{model}/predictions.parquet")) if panel == "reopening_891" else list(root.joinpath(model).rglob("predictions.parquet"))
            frame = day_losses(files, targets, qall, "q0.1")
            for tau, column in ((0.1, "q0.1"), (0.5, "q0.5"), (0.9, "q0.9")):
                raw, scaled, origin = summarise_pinball(frame, column, tau)
                stored[(panel, model, tau)] = {"loss": raw, "scaled_loss": scaled, "origin_scaled": origin}
    rows = []
    rng = np.random.default_rng(SEED)
    for panel, models, _root, _targets in panels:
        for tau in (0.1, 0.5, 0.9):
            best = min(models, key=lambda m: stored[(panel, m, tau)]["scaled_loss"])
            reference = stored[(panel, best, tau)]["origin_scaled"]
            idx = base.block_indices(len(reference), BLOCK, RESAMPLES, rng)
            for model in models:
                item = stored[(panel, model, tau)]
                aligned = item["origin_scaled"].reindex(reference.index)
                if model == best:
                    stat, p, lo, hi = np.nan, np.nan, np.nan, np.nan
                else:
                    stat, p = base.dm_hln(aligned.to_numpy(), reference.to_numpy())
                    lo, hi = base.ci((aligned - reference).to_numpy(), idx)
                rows.append({"panel": panel, "model": model, "tau": tau, "loss": item["loss"],
                             "scaled_loss": item["scaled_loss"], "best_model_at_tau": best,
                             "delta_scaled_vs_best": item["scaled_loss"] - stored[(panel, best, tau)]["scaled_loss"],
                             "delta_ci95_low": lo, "delta_ci95_high": hi, "DM_vs_best": stat, "p_vs_best": p,
                             "significance": base.stars(p)})
    return pd.DataFrame(rows)


def point_decision(qall, targets_outer):
    rows = []
    for config_id, _label, _family, source in base.CONFIGS:
        if config_id not in CORE14:
            continue
        files = base.prediction_files(config_id, source)
        frame = day_losses(files, targets_outer, qall, "prediction")
        if "seed" in frame.columns and frame["seed"].nunique() > 1:
            frame = frame[frame["seed"] == 42]
        if "candidate_id" in frame.columns and source == "outer_server_all":
            chosen = pd.read_csv(REPO / "track1/evaluation/outer_main_table/main_table.csv")
            candidate = dict(zip(chosen.model, chosen.candidate_id))[config_id]
            frame = frame[frame.candidate_id == candidate]
        for tau in (0.1, 0.5, 0.9):
            raw, scaled, _origin = summarise_pinball(frame, "prediction", tau)
            rows.append({"model": LABEL[config_id], "config_id": config_id, "decision": "point_forecast",
                         "tau": tau, "loss": raw, "scaled_loss": scaled,
                         "loss_over_ETS": np.nan})
        print(f"point decision {config_id}", flush=True)
    out = pd.DataFrame(rows)
    ets = out[out.config_id == "ets"].set_index("tau").loss
    out["loss_over_ETS"] = out.apply(lambda r: r.loss / ets.loc[r.tau], axis=1)
    return out


def draw_plots(forest, curve, pareto_frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = OUT / "figures"
    figure.mkdir(exist_ok=True)
    plot = forest.sort_values("relative_change_vs_ETS")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.errorbar(plot.relative_change_vs_ETS, y,
                xerr=[plot.relative_change_vs_ETS - plot.relative_change_ci95_low,
                      plot.relative_change_ci95_high - plot.relative_change_vs_ETS],
                fmt="o", color="black", ecolor="gray", capsize=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot.model)
    ax.set_xlabel("Relative MASE change versus ETS")
    ax.set_title("Origin-block bootstrap, 4,000 resamples")
    fig.tight_layout()
    fig.savefig(figure / "forest_vs_ets.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, part in curve.groupby("model"):
        ax.plot(part.rho, part.cost_ratio_vs_ETS, label=model)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_xlabel("GPU price / vCPU price")
    ax.set_ylabel("Cost relative to ETS")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure / "price_sensitivity.png", dpi=150)
    plt.close(fig)
    for scenario, part in pareto_frame.groupby("scenario"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(part.cost_ratio_vs_ETS, part.MASE, color="gray")
        front = part[part.on_pareto_front].sort_values("cost_ratio_vs_ETS")
        ax.plot(front.cost_ratio_vs_ETS, front.MASE, color="black")
        for row in part.itertuples(index=False):
            ax.annotate(row.model, (row.cost_ratio_vs_ETS, row.MASE), fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("Cost relative to ETS")
        ax.set_ylabel("MASE")
        ax.set_title(scenario)
        fig.tight_layout()
        fig.savefig(figure / f"pareto_{scenario}.png", dpi=150)
        plt.close(fig)


def main():
    task = pd.read_parquet(OUT / "raw/task_metrics.parquet")
    qall = pd.read_parquet(OUT / "raw/q_all.parquet")
    targets = pd.read_parquet(REPO / "shared/data/targets.parquet",
                              columns=["task_id", "horizon_day", "actual", "split_name", "destination"])
    targets_outer = targets[(targets.split_name == "outer_backtest") & (targets.destination == "Macau")][
        ["task_id", "horizon_day", "actual"]]
    targets_stress = targets[(targets.split_name == "pandemic_stress") & (targets.destination == "Macau")][
        ["task_id", "horizon_day", "actual"]]
    task_win_rates(task).to_csv(OUT / "appendix_task_win_rates.csv", index=False)
    market_dm(task).to_csv(OUT / "appendix_market_dm.csv", index=False)
    timing = timing_detail()
    timing.to_csv(OUT / "table4_timing_sd_cv.csv", index=False)
    curve = price_curve(timing)
    curve.to_csv(OUT / "appendix_price_sensitivity_curve.csv", index=False)
    volume, breakeven = volume_and_breakeven(timing, task)
    volume.to_csv(OUT / "table4_annual_cost.csv", index=False)
    breakeven.to_csv(OUT / "table4_breakeven_per_person.csv", index=False)
    front = pareto(timing, task)
    front.to_csv(OUT / "figure4_pareto_coordinates.csv", index=False)
    forest = forest_and_pareto(task, timing)
    forest.to_csv(OUT / "appendix_forest_intervals.csv", index=False)
    print("scoring quantile comparisons", flush=True)
    table5_comparisons(qall, targets_outer, targets_stress).to_csv(OUT / "table5_vs_best.csv", index=False)
    print("scoring point-forecast decisions", flush=True)
    point_decision(qall, targets_outer).to_csv(OUT / "appendix_point_forecast_as_decision.csv", index=False)
    draw_plots(forest, curve, front)
    print("remaining RQ1 outputs written", flush=True)


if __name__ == "__main__":
    main()
