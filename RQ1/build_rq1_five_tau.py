"""Merge the light-model timing into Table 4 / Figure 4 and build the five-level tau curves.

Inputs: RQ1/raw/light_benchmark.jsonl (benchmark_light.py) and
RQ1/raw/five_quantile_runs (run_five_quantiles.py), plus the saved RQ1 tables.
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
import build_rq1_remaining as rem

TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]
QUOTES = [("cloud_on_demand", 0.53, 0.04), ("cloud_spot_or_community", 0.33, 0.04), ("owned_or_lambda", 1.09, 0.04)]
LABEL = {c[0]: c[1] for c in base.CONFIGS}


def light_timing() -> pd.DataFrame:
    rows = [json.loads(line) for line in (OUT / "raw/light_benchmark.jsonl").read_text().splitlines()]
    frame = pd.DataFrame([r for r in rows if not r["warmup"]])
    frame["per_vector"] = frame.allocated_vcpu_seconds / frame.tasks
    frame["inference_per_vector"] = frame.inference_seconds * frame.threads / frame.tasks
    out = frame.groupby("model", as_index=False).agg(
        repeats=("repeat", "size"), seconds_mean=("per_vector", "mean"), seconds_sd=("per_vector", "std"),
        inference_seconds_mean=("inference_per_vector", "mean"),
        inference_seconds_sd=("inference_per_vector", "std"))
    out["device"] = "cpu"
    out["cpu_seconds_mean"] = out.seconds_mean
    out["cv"] = out.seconds_sd / out.seconds_mean
    out["inference_cv"] = out.inference_seconds_sd / out.inference_seconds_mean
    out["rho_star_vs_ETS"] = np.nan
    out["rho_star_inference_only"] = np.nan
    return out


def merged_timing() -> pd.DataFrame:
    old = pd.read_csv(OUT / "table4_timing_sd_cv.csv")
    new = light_timing()
    old = old[~old.model.isin(new.model)]
    timing = pd.concat([old, new[old.columns]], ignore_index=True)
    timing["display_name"] = timing.model.map(lambda m: LABEL.get(m, m))
    timing["timing_source"] = np.where(timing.model.isin(new.model), "RQ1/raw/light_benchmark.jsonl",
                                       "track1/results/compute_benchmark/benchmark.jsonl")
    return timing


def annual_and_breakeven(timing: pd.DataFrame, task: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mae = task.groupby(["config_id", "market"]).mae.mean().groupby("config_id").mean()
    ets_mae = float(mae["ets"])
    ets_seconds = float(timing.loc[timing.model == "ets", "cpu_seconds_mean"].iloc[0])
    annual, breakeven = [], []
    for row in timing.itertuples(index=False):
        with_load = row.seconds_mean if row.device == "gpu" else row.cpu_seconds_mean
        inference = row.inference_seconds_mean
        for n in (27, 1000, 10000):
            hours = n * 365 * with_load / 3600
            hours_inf = n * 365 * inference / 3600
            item = {"model": row.model, "display_name": row.display_name, "device": row.device, "series": n,
                    "annual_hours_with_load": hours, "annual_hours_inference_only": hours_inf}
            for name, gpu_price, cpu_price in QUOTES:
                price = gpu_price if row.device == "gpu" else cpu_price
                item[f"annual_usd_{name}"] = hours * price
                item[f"annual_usd_inference_only_{name}"] = hours_inf * price
            annual.append(item)
        if row.model in mae.index and row.model != "ets":
            people = (ets_mae - float(mae[row.model])) * 27 * 365
            for name, gpu_price, cpu_price in QUOTES:
                price = gpu_price if row.device == "gpu" else cpu_price
                extra = 27 * 365 * (with_load * price - ets_seconds * cpu_price) / 3600
                breakeven.append({"model": row.model, "display_name": row.display_name, "device": row.device,
                                  "scenario": name, "series": 27, "annual_extra_cost_usd": extra,
                                  "annual_absolute_error_reduction_people": people,
                                  "usd_per_person_error_avoided": extra / people if people > 0 else np.nan,
                                  "note": "cheaper than ETS and more accurate" if extra < 0 < people else ""})
    return pd.DataFrame(annual), pd.DataFrame(breakeven)


def pareto(timing: pd.DataFrame, task: pd.DataFrame) -> pd.DataFrame:
    frame = rem.pareto(timing, task)
    frame["display_name"] = frame.model.map(lambda m: LABEL.get(m, m))
    frame["in_table3"] = frame.model.isin(base.CORE)
    return frame


def point_curve(qall, targets) -> pd.DataFrame:
    chosen = pd.read_csv(REPO / "track1/evaluation/outer_main_table/main_table.csv")
    candidate = dict(zip(chosen.model, chosen.candidate_id))
    rows = []
    for config_id, label, family, source in base.CONFIGS[:14]:
        frame = rem.day_losses(base.prediction_files(config_id, source), targets, qall, "prediction")
        if "seed" in frame.columns and frame["seed"].nunique() > 1:
            frame = frame[frame["seed"] == 42]
        if source == "outer_server_all":
            frame = frame[frame.candidate_id == candidate[config_id]]
        for tau in TAUS:
            raw, scaled, _ = rem.summarise_pinball(frame, "prediction", tau)
            rows.append({"config_id": config_id, "model": label, "family": family, "tau": tau,
                         "loss": raw, "scaled_loss": scaled})
    out = pd.DataFrame(rows)
    ets = out[out.config_id == "ets"].set_index("tau").loss
    out["loss_over_ETS"] = out.loss / out.tau.map(ets)
    return out


def quantile_curve(qall, targets_outer, targets_stress) -> pd.DataFrame:
    rows = []
    runs = OUT / "raw/five_quantile_runs"
    panels = [("reopening_891", "outer_backtest", base.FOUNDATION5, targets_outer),
              ("pandemic_351", "pandemic_stress", base.STRESS3, targets_stress)]
    columns = ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day",
               "q0.1", "q0.25", "q0.5", "q0.75", "q0.9"]
    for panel, split, models, targets in panels:
        for model in models:
            files = sorted((runs / split / model).glob("*/predictions.parquet"))
            if not files:
                raise RuntimeError(f"{model} {split}: no five-quantile predictions")
            frame = base.read_many(files, columns)
            if "destination" in frame.columns:
                frame = frame[frame.destination == "Macau"]
            frame = frame.merge(targets, on=["task_id", "horizon_day"], how="inner")
            frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
            frame = frame.merge(qall, on=["market", "origin_date"], how="left")
            if frame[columns[6:]].isna().any().any():
                raise RuntimeError(f"{model} {split}: missing quantiles")
            crossed = float((frame["q0.25"] > frame["q0.5"] + 1e-6).mean()
                            + (frame["q0.75"] < frame["q0.5"] - 1e-6).mean())
            for tau in TAUS:
                raw, scaled, _ = rem.summarise_pinball(frame, f"q{tau:g}", tau)
                rows.append({"panel": panel, "model": model, "display_name": LABEL[model], "tau": tau,
                             "loss": raw, "scaled_loss": scaled, "tasks": int(frame.task_id.nunique()),
                             "quantile_source": ("interpolated between model deciles"
                                                 if model == "timesfm_3" and tau in (0.25, 0.75)
                                                 else "model output"),
                             "share_crossing_median": crossed})
    return pd.DataFrame(rows)


def price_ratio(timing: pd.DataFrame) -> pd.DataFrame:
    ets = float(timing.loc[timing.model == "ets", "cpu_seconds_mean"].iloc[0])
    rows = []
    for row in timing.itertuples(index=False):
        seconds = row.seconds_mean if row.device == "gpu" else row.cpu_seconds_mean
        rho_star = ets / seconds if row.device == "gpu" and seconds else np.nan
        item = {"model": row.model, "device": row.device, "seconds_per_vector": seconds,
                "runs": int(row.repeats), "rho_star_vs_ETS": rho_star}
        for name, gpu_price, cpu_price in QUOTES:
            rho = gpu_price / cpu_price
            item[f"rho_{name}"] = rho
            item[f"cost_ratio_{name}"] = rho / rho_star if rho_star else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def check_against_table5(curve: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OUT / "table5_asymmetric_loss.csv")
    joined = curve.merge(old[["panel", "model", "tau", "loss"]], on=["panel", "model", "tau"], how="inner",
                         suffixes=("", "_table5"))
    joined["abs_diff"] = (joined.loss - joined.loss_table5).abs()
    return joined[["panel", "model", "tau", "loss", "loss_table5", "abs_diff"]]


def plots(timing, pareto_frame, point, quant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    for scenario, part in pareto_frame.groupby("scenario"):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        colors = np.where(part.in_table3, "black", "lightgray")
        ax.scatter(part.cost_ratio_vs_ETS, part.MASE, c=colors)
        front = part[part.on_pareto_front].sort_values("cost_ratio_vs_ETS")
        ax.plot(front.cost_ratio_vs_ETS, front.MASE, color="tab:red")
        for row in part.itertuples(index=False):
            ax.annotate(row.display_name, (row.cost_ratio_vs_ETS, row.MASE), fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("Cost relative to ETS (log scale)")
        ax.set_ylabel("MASE")
        ax.set_title(scenario)
        fig.tight_layout()
        fig.savefig(fig_dir / f"pareto_{scenario}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, part in point.groupby("model"):
        ax.plot(part.tau, part.scaled_loss, marker="o", label=model)
    ax.set_xlabel("tau")
    ax.set_ylabel("Scaled asymmetric loss (point forecast as decision)")
    ax.set_xticks(TAUS)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "five_tau_point_forecast_as_decision.png", dpi=150)
    plt.close(fig)
    for panel, part in quant.groupby("panel"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, line in part.groupby("display_name"):
            ax.plot(line.tau, line.scaled_loss, marker="o", label=model)
        ax.set_xlabel("tau")
        ax.set_ylabel("Scaled quantile loss (model quantiles)")
        ax.set_xticks(TAUS)
        ax.set_title(panel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"five_tau_quantile_{panel}.png", dpi=150)
        plt.close(fig)


def main():
    task = pd.read_parquet(OUT / "raw/task_metrics.parquet")
    qall = pd.read_parquet(OUT / "raw/q_all.parquet")
    targets = pd.read_parquet(REPO / "shared/data/targets.parquet",
                              columns=["task_id", "horizon_day", "actual", "split_name", "destination"])
    outer = targets[(targets.split_name == "outer_backtest") & (targets.destination == "Macau")][["task_id", "horizon_day", "actual"]]
    stress = targets[(targets.split_name == "pandemic_stress") & (targets.destination == "Macau")][["task_id", "horizon_day", "actual"]]

    timing = merged_timing()
    timing.to_csv(OUT / "table4_timing_sd_cv.csv", index=False)
    price_ratio(timing).to_csv(OUT / "table4_timing_and_price_ratio.csv", index=False)
    annual, breakeven = annual_and_breakeven(timing, task)
    annual.to_csv(OUT / "table4_annual_cost.csv", index=False)
    breakeven.to_csv(OUT / "table4_breakeven_per_person.csv", index=False)
    front = pareto(timing, task)
    front.to_csv(OUT / "figure4_pareto_coordinates.csv", index=False)
    print("timing merged", flush=True)

    point = point_curve(qall, outer)
    point.to_csv(OUT / "appendix_five_tau_point_forecast_as_decision.csv", index=False)
    point_decision = point.rename(columns={"config_id": "config_id", "model": "model"}).copy()
    point_decision["decision"] = "point_forecast"
    point_decision = point_decision[["model", "config_id", "decision", "tau", "loss", "scaled_loss", "loss_over_ETS"]]
    point_decision.to_csv(OUT / "appendix_point_forecast_as_decision.csv", index=False)
    quant = quantile_curve(qall, outer, stress)
    quant.to_csv(OUT / "appendix_five_tau_quantile_loss.csv", index=False)
    check_against_table5(quant).to_csv(OUT / "raw/five_quantile_vs_table5_check.csv", index=False)
    plots(timing, front, point, quant)
    print("five-tau curves written", flush=True)


if __name__ == "__main__":
    main()
