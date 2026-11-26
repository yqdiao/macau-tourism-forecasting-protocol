"""4.2.5 Hong Kong validation, information ablation, RQ2 boundary matrix."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from common import (
    B, BLOCK_REOPEN, BLOCK_STRESS, CORE_IDS, FROZEN_BASELINE, FOUNDATION5, HEADLINE, LABEL,
    RAW, TABLES, REPO, RQ1, SEED, block_indices, dm_hln, log, market_then_equal, origin_means,
    stars, ensure_dirs,
)


def score_hk_tasks():
    """66 HK outer tasks from foundation + baselines that include HongKong rows."""
    targets = pd.read_parquet(REPO / "shared/data/targets.parquet",
                              columns=["task_id", "horizon_day", "actual", "split_name", "destination", "market"])
    targets = targets[(targets.destination == "HongKong") & (targets.split_name == "outer_backtest")]
    qrows = []
    # build q_all for HK markets from normalized daily
    daily = pd.read_json(REPO / "shared/data/normalized_daily.jsonl", lines=True)
    daily = daily[daily.destination == "HongKong"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    origins = pd.to_datetime(targets.merge(
        pd.read_parquet(REPO / "shared/data/task_index.parquet", columns=["task_id", "origin_date"]),
        on="task_id")["origin_date"]).drop_duplicates().sort_values()
    # get origin via task_index
    ti = pd.read_parquet(REPO / "shared/data/task_index.parquet")
    ti = ti[(ti.destination == "HongKong") & (ti.split_name == "outer_backtest")][["task_id", "market", "origin_date", "fold_id"]]
    for market, group in daily.groupby("market"):
        series = group.sort_values("date").set_index("date").arrivals.astype(float)
        for origin in pd.to_datetime(ti.origin_date.unique()):
            hist = series.loc[:origin]
            if len(hist) <= 7:
                continue
            q = float((hist - hist.shift(7)).dropna().abs().mean())
            qrows.append({"market": market, "origin_date": origin.date(), "q_all": q})
    qall = pd.DataFrame(qrows)

    rows = []
    # foundation models
    for model in FOUNDATION5 + ["ets", "sarima", "seasonal_naive", "svr", "lightgbm", "transformer"]:
        if model in FOUNDATION5:
            files = sorted((REPO / "track1/results/foundation_outer_default").glob(f"*/{model}/predictions.parquet"))
        else:
            cand = FROZEN_BASELINE[model]
            files = sorted((REPO / f"track1/results/outer_server_all/predictions/{model}/{cand}/42").glob("*.parquet"))
        if not files:
            log(f"HK missing {model}"); continue
        cols = ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day", "prediction", "candidate_id", "seed"]
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        frame = frame[frame.destination == "HongKong"]
        if "seed" in frame.columns and frame.seed.nunique() > 1:
            frame = frame[frame.seed == 42]
        if model in FROZEN_BASELINE and "candidate_id" in frame.columns:
            frame = frame[frame.candidate_id == FROZEN_BASELINE[model]]
        frame = frame.merge(targets[["task_id", "horizon_day", "actual"]], on=["task_id", "horizon_day"], how="inner")
        frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
        frame = frame.merge(qall, on=["market", "origin_date"], how="left")
        frame["abs_error"] = (frame.prediction - frame.actual).abs()
        frame["sq"] = (frame.prediction - frame.actual) ** 2
        frame["ape"] = np.where(frame.actual > 0, frame.abs_error / frame.actual, np.nan)
        g = frame.groupby(["market", "fold_id", "origin_date", "task_id"], sort=False)
        task = g.agg(mae=("abs_error", "mean"), mse=("sq", "mean"), mape=("ape", "mean"), q_all=("q_all", "first")).reset_index()
        task["rmse"] = np.sqrt(task.mse); task["mase"] = task.mae / task.q_all; task["mape_pct"] = task.mape * 100
        task["config_id"] = model
        rows.append(task)
        log(f"HK scored {model} tasks={task.task_id.nunique()}")
    return pd.concat(rows, ignore_index=True)


def run():
    ensure_dirs()
    hk = score_hk_tasks()
    hk.to_parquet(RAW / "hongkong_task_metrics.parquet", index=False)
    ets = hk[hk.config_id == "ets"]
    rng = np.random.default_rng(SEED)
    s11 = []
    for cid in hk.config_id.unique():
        part = hk[hk.config_id == cid]
        item = {"config_id": cid, "model": LABEL.get(cid, cid), "tasks": int(part.task_id.nunique())}
        for col, name in [("mase", "MASE"), ("mape_pct", "MAPE"), ("mae", "MAE"), ("rmse", "RMSE")]:
            item[name] = market_then_equal(part, col)
            # ratio vs HK ETS with block bootstrap on origins
            om = origin_means(part, col); oe = origin_means(ets, col)
            idx = om.index.intersection(oe.index)
            om, oe = om.loc[idx], oe.loc[idx]
            item[f"R_{name}_vs_ETS"] = float(om.mean() / oe.mean())
            ii = block_indices(len(om), BLOCK_REOPEN, B, rng)
            ratios = om.to_numpy()[ii].mean(1) / oe.to_numpy()[ii].mean(1)
            item[f"R_{name}_ci95_low"] = float(np.quantile(ratios, 0.025))
            item[f"R_{name}_ci95_high"] = float(np.quantile(ratios, 0.975))
            dm, p = dm_hln(om, oe, lag=2)
            item[f"DM_{name}"] = dm; item[f"p_{name}"] = p; item[f"sig_{name}"] = stars(p)
        s11.append(item)
    s11 = pd.DataFrame(s11)
    # rank correlation vs Macau reopen
    macau = pd.read_parquet(RAW / "task_metrics_rq2.parquet")
    macau = macau[macau.split_name == "outer_backtest"]
    common = [c for c in s11.config_id if c in set(macau.config_id)]
    macau_rank = {c: market_then_equal(macau[macau.config_id == c], "mase") for c in common}
    hk_rank = {c: float(s11.loc[s11.config_id == c, "MASE"].iloc[0]) for c in common}
    xs = [macau_rank[c] for c in common]; ys = [hk_rank[c] for c in common]
    rho, rp = spearmanr(xs, ys); tau, tp = kendalltau(xs, ys)
    s11["spearman_rho_vs_macau_MASE_rank"] = rho
    s11["kendall_tau_vs_macau_MASE_rank"] = tau
    s11.to_csv(TABLES / "table_S11_hongkong_validation_and_macau_control.csv", index=False)
    log(f"HK vs Macau Spearman rho={rho:.3f}")

    # S12 ablation from table3_calendar_related_extension + core univariate
    ext = pd.read_csv(RQ1 / "tables/table3_calendar_related_extension.csv")
    core = pd.read_csv(RQ1 / "tables/table3_accuracy.csv")
    # timing for cost multiples if available
    timing = pd.read_csv(RQ1 / "tables/table4_timing_sd_cv.csv")
    s12 = []
    for _, r in ext.iterrows():
        base = core[core.config_id == r.model]
        if base.empty:
            continue
        b = base.iloc[0]
        item = {
            "model": r.model, "display_name": LABEL.get(r.model, r.model),
            "input_mode": r.input_mode,
            "tasks": r.tasks,
            "MASE_extended": r.MASE, "MAPE_extended": r.MAPE_pct, "RMSE_extended": r.RMSE,
            "MASE_univariate": b.MASE, "MAPE_univariate": b.MAPE_pct, "RMSE_univariate": b.RMSE,
            "MAE_univariate": b.MAE,
            "pct_change_MASE": 100 * (r.MASE / b.MASE - 1),
            "pct_change_MAPE": 100 * (r.MAPE_pct / b.MAPE_pct - 1),
            "pct_change_RMSE": 100 * (r.RMSE / b.RMSE - 1),
            "note": "extension row is calendar+related/cross-learning; no calendar-only row in source",
        }
        # LL01 vs lightgbm as calendar local ablation
        s12.append(item)
    # add LL01 vs lightgbm univariate comparison
    ll = core[core.config_id == "LL01"].iloc[0]
    lg = core[core.config_id == "lightgbm"].iloc[0]
    s12.append({
        "model": "LL01_vs_lightgbm", "display_name": "LightGBM+Calendar vs LightGBM local",
        "input_mode": "local_calendar_features_vs_univariate_lightgbm",
        "tasks": 891,
        "MASE_extended": ll.MASE, "MAPE_extended": ll.MAPE_pct, "RMSE_extended": ll.RMSE,
        "MASE_univariate": lg.MASE, "MAPE_univariate": lg.MAPE_pct, "RMSE_univariate": lg.RMSE,
        "MAE_univariate": lg.MAE, "MAE_extended": ll.MAE,
        "pct_change_MASE": 100 * (ll.MASE / lg.MASE - 1),
        "pct_change_MAPE": 100 * (ll.MAPE_pct / lg.MAPE_pct - 1),
        "pct_change_RMSE": 100 * (ll.RMSE / lg.RMSE - 1),
        "note": "LL01 adds calendar features to local LightGBM",
    })
    # timing multiples if both present
    for row in s12:
        mid = row["model"] if row["model"] in set(timing.model) else None
        if mid is not None and "ets" in set(timing.model):
            row["seconds_per_vector"] = float(timing.loc[timing.model == mid, "seconds_mean"].iloc[0])
    pd.DataFrame(s12).to_csv(TABLES / "table_S12_exogenous_information_ablation_4metrics.csv", index=False)

    # S13 boundary matrix across 10 dimensions
    task = pd.read_parquet(RAW / "task_metrics_rq2.parquet")
    days = pd.read_parquet(RAW / "day_metrics_rq2.parquet")
    hist = pd.read_parquet(RAW / "history_task_metrics.parquet") if (RAW / "history_task_metrics.parquet").exists() else None
    s6 = pd.read_csv(TABLES / "table_S6_market_size_groups_4metrics.csv")
    s8 = pd.read_csv(TABLES / "table_S8_horizon_weekly_blocks_and_same_weekday.csv")
    s9 = pd.read_csv(TABLES / "table_S9_high_demand_holiday_underprediction_risk.csv")

    def status(r, lo, hi):
        if r < 1 and hi < 1:
            return "significantly_better"
        if r < 1 and lo < 1 < hi:
            return "point_better_CI_crosses_1"
        if r > 1:
            return "worse_than_ETS"
        return "approx_parity"

    dims = []
    # helpers to compute R with CI for a filtered task set
    def add_dim(name, subset_task, models=HEADLINE):
        ets = subset_task[subset_task.config_id == "ets"]
        rng = np.random.default_rng(SEED + hash(name) % 1_000_000)
        block = BLOCK_REOPEN if subset_task.fold_id.astype(str).str.startswith("O").any() else BLOCK_STRESS
        # detect stress by fold A
        if subset_task.fold_id.astype(str).str.startswith("A").any():
            block = BLOCK_STRESS
        for cid in models:
            part = subset_task[subset_task.config_id == cid]
            if part.empty or ets.empty:
                continue
            for col, metric in [("mase", "MASE"), ("mape_pct", "MAPE"), ("mae", "MAE"), ("rmse", "RMSE")]:
                om = origin_means(part, col); oe = origin_means(ets, col)
                idx = om.index.intersection(oe.index)
                if len(idx) < 3:
                    continue
                om, oe = om.loc[idx], oe.loc[idx]
                r = float(om.mean() / oe.mean())
                ii = block_indices(len(om), min(block, max(1, len(om) - 1)) if len(om) > 1 else 1, B, rng)
                ratios = om.to_numpy()[ii].mean(1) / oe.to_numpy()[ii].mean(1)
                lo, hi = float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))
                dims.append({"dimension": name, "config_id": cid, "model": LABEL[cid], "metric": metric,
                             "R_vs_ETS": r, "ci95_low": lo, "ci95_high": hi, "status": status(r, lo, hi)})

    add_dim("1_reopening_overall", task[task.split_name == "outer_backtest"])
    add_dim("2_pandemic_stress", task[task.split_name == "pandemic_stress"])
    for g in ["Large", "Medium", "Small"]:
        sub = task[(task.split_name == "outer_backtest") & (task.market_group == g)]
        # market_group may be missing if not in panel - add it
        if "market_group" not in task.columns:
            pass
        add_dim(f"{3 if g=='Large' else 4 if g=='Medium' else 5}_market_{g.lower()}", 
                task[(task.split_name == "outer_backtest")].assign(
                    market_group=task[task.split_name=="outer_backtest"].market.map(
                        {**{m:'Large' for m in ['CN_Chinese_mainland','HK_Hong_Kong_SAR']}}
                    )
                ) if False else task[(task.split_name == "outer_backtest") & (task.market.isin(
                    {"Large": ["CN_Chinese_mainland","HK_Hong_Kong_SAR"],
                     "Medium": ["TW_Taiwan_region","KR_Republic_of_Korea","PH_Philippines","JP_Japan","MY_Malaysia","ID_Indonesia","US_United_States_of_America","TH_Thailand"],
                     "Small": [m for m in task.market.unique() if m not in ["CN_Chinese_mainland","HK_Hong_Kong_SAR","TW_Taiwan_region","KR_Republic_of_Korea","PH_Philippines","JP_Japan","MY_Malaysia","ID_Indonesia","US_United_States_of_America","TH_Thailand"]]}
                    [g]))])
    if hist is not None:
        for code, label in [("180", "6_clean180"), ("all", "7_full_with_covid")]:
            sub = hist[hist.history_setting == code].copy()
            sub["mase"] = sub["mase_fixed"]
            keep = [c for c in ["config_id", "market", "fold_id", "origin_date", "task_id",
                                "mae", "mape_pct", "rmse", "mase"] if c in sub.columns]
            add_dim(label, sub[keep])
    # Week4 from day aggregation already in s8 — rebuild task-like from days
    d = days[days.split_name == "outer_backtest"].copy()
    d = d[d.horizon_day >= 22]
    t = d.groupby(["config_id", "market", "fold_id", "origin_date", "task_id"], sort=False).agg(
        mae=("abs_error", "mean"), mse=("sq", "mean"), mape=("ape", "mean"), q_all=("q_all", "first")
    ).reset_index()
    t["rmse"] = np.sqrt(t.mse); t["mase"] = t.mae / t.q_all; t["mape_pct"] = t.mape * 100
    add_dim("8_week4_horizon", t)
    # holidays
    hol = pd.read_parquet(REPO / "shared/data/holiday_calendar.parquet")
    hol["date"] = pd.to_datetime(hol["date"]).dt.date
    official = set(hol.loc[hol.role.isin(["mainland", "macau"]), "date"])
    d2 = days[days.split_name == "outer_backtest"].copy()
    d2["target_date"] = pd.to_datetime(d2.target_date).dt.date
    d2 = d2[d2.target_date.isin(official)]
    t2 = d2.groupby(["config_id", "market", "fold_id", "origin_date", "task_id"], sort=False).agg(
        mae=("abs_error", "mean"), mse=("sq", "mean"), mape=("ape", "mean"), q_all=("q_all", "first")
    ).reset_index()
    t2["rmse"] = np.sqrt(t2.mse); t2["mase"] = t2.mae / t2.q_all; t2["mape_pct"] = t2.mape * 100
    add_dim("9_exante_official_holiday", t2)
    # HK
    add_dim("10_hongkong_external", hk)
    pd.DataFrame(dims).to_csv(TABLES / "table_S13_rq2_boundary_summary_matrix.csv", index=False)
    log("wrote S11-S13")


if __name__ == "__main__":
    run()
