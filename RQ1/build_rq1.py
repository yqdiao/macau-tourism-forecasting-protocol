"""Build the RQ1 tables from the completed 891-task and 351-task experiments.

No model is retrained. Day-level predictions, the isolated timing log and the
Qwen3-4B deliveries are scored with one common denominator q_all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
OUT = REPO / "RQ1"
RAW = OUT / "raw"
BLOCK, RESAMPLES, SEED = 3, 4000, 20261116

# 23 completed point-forecast configurations. The first 14 are the Table 3 core.
CONFIGS = [
    ("granite_patchtst_fm", "Granite", "Foundation", "foundation_outer_default"),
    ("timesfm_3", "TimesFM 3.0", "Foundation", "foundation_outer_default"),
    ("chronos_2", "Chronos-2", "Foundation", "foundation_outer_default"),
    ("moirai", "Moirai 2.0", "Foundation", "foundation_outer_default"),
    ("lag_llama", "Lag-Llama", "Foundation", "foundation_outer_default"),
    ("ets", "ETS", "Statistical", "outer_server_all"),
    ("sarima", "SARIMA", "Statistical", "outer_server_all"),
    ("seasonal_naive", "Seasonal Naive", "Statistical", "outer_server_all"),
    ("svr", "SVR", "Machine learning", "outer_server_all"),
    ("lightgbm", "LightGBM (Local)", "Machine learning", "outer_server_all"),
    ("GL03", "LightGBM (Global)", "Machine learning", "light_global"),
    ("LL01", "LightGBM+Calendar (Local)", "Machine learning", "light_local"),
    ("transformer", "Small Transformer", "Deep learning", "outer_server_all"),
    ("NL01", "Global NLinear", "Deep learning", "light_nlinear"),
    ("gradient_boosting", "GBDT", "Machine learning", "outer_server_all"),
    ("random_forest", "Random Forest", "Machine learning", "outer_server_all"),
    ("ridge", "Ridge", "Machine learning", "outer_server_all"),
    ("linear_regression", "Linear Regression", "Machine learning", "outer_server_all"),
    ("lasso", "Lasso", "Machine learning", "outer_server_all"),
    ("elasticnet", "Elastic Net", "Machine learning", "outer_server_all"),
    ("parrot_random_forest_800", "Random Forest 800", "Machine learning", "outer_server_all"),
    ("patchtst", "PatchTST", "Deep learning", "foundation_outer_default"),
    ("informer", "Informer", "Deep learning", "foundation_outer_default"),
]
CORE = {c[0] for c in CONFIGS[:14]}
FOUNDATION5 = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"]
STRESS3 = ["granite_patchtst_fm", "timesfm_3", "chronos_2"]


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def dm_hln(loss_a, loss_b, h=1):
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    t = len(d)
    if t < 3 or np.allclose(d, d[0]):
        return np.nan, np.nan
    mean = d.mean()
    gamma = [np.sum((d[k:] - mean) * (d[:t - k] - mean)) / t for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / t
    if var <= 0:
        return np.nan, np.nan
    stat = mean / np.sqrt(var) * np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=t - 1))


def block_indices(t, block, resamples, rng):
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, t - block + 1, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)
    return idx[:, :t]


def ci(values, idx):
    means = np.asarray(values, float)[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def q_all_table():
    rows = [json.loads(line) for line in open(REPO / "shared/data/normalized_daily.jsonl")]
    frame = pd.DataFrame(rows)
    frame = frame[frame.destination == "Macau"]
    frame["date"] = pd.to_datetime(frame.date)
    origins = pd.read_parquet(REPO / "shared/data/task_index.parquet", columns=["origin_date", "destination", "split_name"])
    origins = origins[(origins.destination == "Macau") & (origins.split_name.isin(["outer_backtest", "pandemic_stress"]))]
    origins = pd.to_datetime(origins.origin_date).drop_duplicates().sort_values()
    out = []
    for market, group in frame.groupby("market"):
        series = group.sort_values("date").set_index("date").arrivals.astype(float)
        for origin in origins:
            hist = series.loc[:origin]
            if len(hist) <= 7:
                continue
            diff = (hist - hist.shift(7)).dropna().abs()
            out.append({"market": market, "origin_date": origin.date(), "q_all": float(diff.mean()),
                        "history_days": int(len(hist))})
    return pd.DataFrame(out)


def read_many(paths, columns):
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        keep = [c for c in columns if c in frame.columns]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def prediction_files(model, source):
    root = REPO / "track1/results"
    if source == "outer_server_all":
        return list((root / "outer_server_all/predictions" / model).rglob("*.parquet"))
    if source == "foundation_outer_default":
        return list((root / "foundation_outer_default").glob(f"*/{model}/predictions.parquet"))
    if source == "light_local":
        return list((root / "light_models/outer_backtest/local_lightgbm/LL01").rglob("predictions.parquet"))
    if source == "light_global":
        return list((root / "light_models/outer_backtest/global_lightgbm/GL03").rglob("predictions.parquet"))
    if source == "light_nlinear":
        return list((root / "light_models/outer_backtest/global_linear/NL01").rglob("predictions.parquet"))
    raise KeyError(source)


def task_frame(days, qall):
    days = days[days.destination == "Macau"].copy()
    days["origin_date"] = pd.to_datetime(days.origin_date).dt.date
    days = days.merge(qall, on=["market", "origin_date"], how="left")
    if days.q_all.isna().any():
        raise RuntimeError("q_all missing for some predictions")
    days["error"] = days.prediction - days.actual
    days["abs_error"] = days.error.abs()
    days["sq"] = days.error ** 2
    days["ape"] = np.where(days.actual > 0, days.abs_error / days.actual, np.nan)
    group = days.groupby(["config_id", "market", "fold_id", "origin_date", "task_id"], sort=False)
    task = group.agg(n=("horizon_day", "size"), mae=("abs_error", "mean"), mse=("sq", "mean"),
                     mape=("ape", "mean"), bias=("error", "mean"), q_all=("q_all", "first"),
                     zero_days=("actual", lambda s: int((s == 0).sum())),
                     positive_days=("actual", lambda s: int((s > 0).sum())),
                     sum_abs=("abs_error", "sum"), sum_actual=("actual", "sum")).reset_index()
    task["rmse"] = np.sqrt(task.mse)
    task["mase"] = task.mae / task.q_all
    task["mape"] = task.mape * 100
    task["scaled_bias"] = task.bias / task.q_all
    return task


def hierarchy(task):
    market = task.groupby(["config_id", "market"], sort=False).agg(
        origins=("origin_date", "nunique"), mase=("mase", "mean"), mape=("mape", "mean"),
        rmse=("rmse", "mean"), mae=("mae", "mean"), scaled_bias=("scaled_bias", "mean")).reset_index()
    overall = market.groupby("config_id", sort=False).agg(
        markets=("market", "nunique"), mase=("mase", "mean"), mape=("mape", "mean"),
        rmse=("rmse", "mean"), mae=("mae", "mean"), scaled_bias=("scaled_bias", "mean")).reset_index()
    return market, overall


def origin_loss(task, column):
    return task.groupby(["config_id", "fold_id"], sort=True)[column].mean().unstack(0).sort_index()


def load_all(qall, targets):
    columns = ["task_id", "destination", "market", "fold_id", "origin_date", "target_date",
               "horizon_day", "prediction", "model", "candidate_id", "seed", "mase7_scale"]
    pieces = []
    wanted = pd.read_csv(REPO / "track1/evaluation/outer_main_table/main_table.csv")
    chosen = dict(zip(wanted.model, wanted.candidate_id))
    for config_id, _label, _family, source in CONFIGS:
        files = prediction_files(config_id if source != "outer_server_all" else config_id, source)
        frame = read_many(files, columns)
        if source == "outer_server_all":
            frame = frame[frame.candidate_id == chosen[config_id]]
        if "seed" in frame.columns and frame["seed"].nunique() > 1:
            frame = frame[frame["seed"] == 42]
        if "destination" in frame.columns:
            frame = frame[frame["destination"] == "Macau"]
        frame["config_id"] = config_id
        frame["destination"] = frame.get("destination", "Macau")
        if "destination" not in frame.columns:
            frame["destination"] = "Macau"
        pieces.append(frame)
        print(f"loaded {config_id}: rows={len(frame)}", flush=True)
    days = pd.concat(pieces, ignore_index=True)
    days = days.merge(targets, on=["task_id", "horizon_day"], how="left", suffixes=("", "_target"))
    if days.actual.isna().any():
        raise RuntimeError(f"unmatched predictions: {int(days.actual.isna().sum())}")
    return task_frame(days, qall)


def qwen_tasks(qall, targets):
    rows = []
    path = REPO / "RQ1/raw/qwen3_4b_D-S1_deliveries.jsonl"
    for line in open(path):
        item = json.loads(line)
        if item["destination"] != "Macau" or item["mode"] != "D":
            continue
        origin = pd.Timestamp(item["origin"]).date()
        for horizon, value in enumerate(item["forecast"], 1):
            rows.append({"market": item["market_id"], "origin_date": origin, "horizon_day": horizon,
                         "replicate": item["replicate"], "prediction": float(value)})
    rep = pd.DataFrame(rows)
    extreme_pairs = set(map(tuple, rep.loc[rep.prediction > 1e12, ["market", "origin_date"]].drop_duplicates().to_numpy()))
    mean = rep.groupby(["market", "origin_date", "horizon_day"], as_index=False).prediction.mean()
    lookup = targets[(targets.split_name == "outer_backtest") & (targets.destination == "Macau")].copy()
    lookup["origin_date"] = pd.to_datetime(lookup.origin_date).dt.date
    index = pd.read_parquet(REPO / "shared/data/task_index.parquet",
                            columns=["task_id", "split_name", "destination", "market", "origin_date", "fold_id", "history_length"])
    index = index[(index.split_name == "outer_backtest") & (index.destination == "Macau") & (index.history_length == "all")]
    index["origin_date"] = pd.to_datetime(index.origin_date).dt.date
    mean = mean.merge(lookup[["market", "origin_date", "horizon_day", "actual"]],
                      on=["market", "origin_date", "horizon_day"], how="left")
    mean = mean.merge(index[["market", "origin_date", "task_id", "fold_id"]], on=["market", "origin_date"], how="left")
    if mean.actual.isna().any():
        raise RuntimeError(f"Qwen rows without targets: {int(mean.actual.isna().sum())}")
    mean["destination"] = "Macau"
    mean["config_id"] = "qwen3_4b"
    mean["target_date"] = pd.to_datetime(mean.origin_date) + pd.to_timedelta(mean.horizon_day, unit="D")
    scored = task_frame(mean, qall)
    extreme_ids = set(scored.loc[scored.apply(lambda r: (r.market, r.origin_date) in extreme_pairs, axis=1), "task_id"])
    return scored, extreme_ids


def asymmetric(files, targets, qall, label):
    columns = ["task_id", "destination", "market", "fold_id", "origin_date", "target_date", "horizon_day",
               "prediction", "q0.1", "q0.5", "q0.9"]
    frame = read_many(files, columns)
    frame = frame[frame.destination == "Macau"]
    frame = frame.merge(targets, on=["task_id", "horizon_day"], how="left")
    frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
    frame = frame.merge(qall, on=["market", "origin_date"], how="left")
    rows = []
    for tau, column in ((0.1, "q0.1"), (0.5, "q0.5"), (0.9, "q0.9")):
        pred = frame[column].to_numpy(float)
        actual = frame.actual.to_numpy(float)
        loss = np.where(actual >= pred, tau * (actual - pred), (1 - tau) * (pred - actual))
        part = frame[["market", "fold_id", "q_all"]].copy()
        part["loss"] = loss
        part["scaled"] = loss / part.q_all
        by_origin = part.groupby(["market", "fold_id"], sort=False).agg(loss=("loss", "mean"), scaled=("scaled", "mean")).reset_index()
        by_market = by_origin.groupby("market", sort=False).agg(loss=("loss", "mean"), scaled=("scaled", "mean"))
        rows.append({"panel": label, "tau": tau, "model": frame.model.iloc[0] if "model" in frame else label,
                     "loss": float(by_market.loss.mean()), "scaled_loss": float(by_market.scaled.mean()),
                     "origins": part.groupby("fold_id").loss.mean()})
    return rows


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("building q_all", flush=True)
    qall = q_all_table()
    qall.to_parquet(RAW / "q_all.parquet", index=False)
    targets = pd.read_parquet(REPO / "shared/data/targets.parquet",
                              columns=["task_id", "horizon_day", "actual", "origin_date", "market", "destination", "split_name"])
    targets_outer = targets[(targets.split_name == "outer_backtest") & (targets.destination == "Macau")][
        ["task_id", "horizon_day", "actual"]]
    print("loading predictions", flush=True)
    task = load_all(qall, targets_outer)
    labels = pd.DataFrame(CONFIGS, columns=["config_id", "label", "family", "source"])
    task = task.merge(labels, on="config_id", how="left")
    task.to_parquet(RAW / "task_metrics.parquet", index=False)
    zero_days = int(task.zero_days.sum())
    positive_days = int(task.positive_days.sum())
    market, overall = hierarchy(task)
    market.to_parquet(RAW / "market_metrics.parquet", index=False)

    # Table 3 and the 23-config appendix use the same scores.
    origin_mase = origin_loss(task, "mase")
    origin_mape = origin_loss(task, "mape")
    origin_rmse = origin_loss(task, "rmse")
    ets = origin_mase["ets"]
    core_ids = [c[0] for c in CONFIGS[:14]]
    non_foundation = [c[0] for c in CONFIGS[:14] if c[2] != "Foundation"]
    strongest = overall.set_index("config_id").loc[non_foundation, "mase"].idxmin()
    rng = np.random.default_rng(SEED)
    idx = block_indices(len(origin_mase), BLOCK, RESAMPLES, rng)

    def paired(column, reference):
        stat, p = dm_hln(column.to_numpy(), reference.to_numpy())
        lo, hi = ci((column - reference).to_numpy(), idx)
        return stat, p, lo, hi

    rows = []
    wins = market.pivot(index="market", columns="config_id", values="mase")
    for config_id, label, family, _source in CONFIGS:
        score = overall.set_index("config_id").loc[config_id]
        stat_e, p_e, lo_e, hi_e = paired(origin_mase[config_id], ets) if config_id != "ets" else (np.nan, np.nan, np.nan, np.nan)
        stat_s, p_s, lo_s, hi_s = paired(origin_mase[config_id], origin_mase[strongest]) if config_id != strongest else (np.nan, np.nan, np.nan, np.nan)
        beat = int((wins[config_id] < wins["ets"]).sum()) if config_id != "ets" else np.nan
        mean_lo, mean_hi = ci(origin_mase[config_id].to_numpy(), idx)
        rows.append({
            "config_id": config_id, "model": label, "family": family, "in_table3": config_id in CORE,
            "tasks": int((task.config_id == config_id).sum()), "markets": int(score.markets),
            "MASE": score.mase, "MAPE_pct": score.mape, "RMSE": score.rmse, "MAE": score.mae,
            "DM_vs_ETS": stat_e, "p_vs_ETS": p_e, "sig_vs_ETS": stars(p_e),
            "DM_vs_strongest": stat_s, "p_vs_strongest": p_s, "sig_vs_strongest": stars(p_s),
            "strongest_baseline": labels.set_index("config_id").loc[strongest, "label"],
            "wins_vs_ETS": beat, "wins_vs_ETS_text": "" if config_id == "ets" else f"{beat}/27",
            "MASE_ci95_low": mean_lo, "MASE_ci95_high": mean_hi,
            "delta_vs_ETS_ci95_low": lo_e, "delta_vs_ETS_ci95_high": hi_e,
            "relative_change_vs_ETS": score.mase / overall.set_index("config_id").loc["ets", "mase"] - 1,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "appendix_23configs.csv", index=False)
    table[table.in_table3].drop(columns=["in_table3"]).to_csv(OUT / "table3_accuracy.csv", index=False)

    # Granite vs TimesFM on three metrics.
    duel = []
    for metric, frame in (("MASE", origin_mase), ("MAPE", origin_mape), ("RMSE", origin_rmse)):
        stat, p = dm_hln(frame["granite_patchtst_fm"].to_numpy(), frame["timesfm_3"].to_numpy())
        lo, hi = ci((frame["granite_patchtst_fm"] - frame["timesfm_3"]).to_numpy(), idx)
        duel.append({"metric": metric, "DM_Granite_minus_TimesFM": stat, "p_value": p, "significance": stars(p),
                     "delta_ci95_low": lo, "delta_ci95_high": hi})
    pd.DataFrame(duel).to_csv(OUT / "granite_vs_timesfm.csv", index=False)

    # Large-market share of the equal-weight mean MAE.
    shares = []
    for config_id in ("granite_patchtst_fm", "ets"):
        part = market[market.config_id == config_id].set_index("market").mae
        total = part.sum()
        shares.append({"model": config_id, "mainland_share": part["CN_Chinese_mainland"] / total,
                       "mainland_hongkong_share": (part["CN_Chinese_mainland"] + part["HK_Hong_Kong_SAR"]) / total,
                       "mainland_MAE": part["CN_Chinese_mainland"], "hongkong_MAE": part["HK_Hong_Kong_SAR"],
                       "equal_weight_mean_MAE": part.mean()})
    pd.DataFrame(shares).to_csv(OUT / "large_market_mae_share.csv", index=False)
    market.to_csv(OUT / "appendix_market_metrics.csv", index=False)

    # Matched-input extension rows.
    matched_rows = []
    for model, folder in (("timesfm_3", "timesfm_3"), ("chronos_2", "chronos_2"), ("moirai", "moirai")):
        files = list((REPO / "track1/results/zeroshot_matched_inputs" / folder).rglob("predictions.parquet"))
        frame = read_many(files, ["task_id", "market", "fold_id", "origin_date", "horizon_day", "prediction", "input_mode", "mase7_scale"])
        frame["destination"] = "Macau"
        frame["config_id"] = model + "|" + frame.input_mode.iloc[0]
        frame = frame.merge(targets_outer, on=["task_id", "horizon_day"], how="left")
        scored = task_frame(frame, qall)
        _m, summary = hierarchy(scored)
        matched_rows.append({"model": model, "input_mode": frame.input_mode.iloc[0], "tasks": len(scored),
                             **summary.iloc[0][["mase", "mape", "rmse"]].to_dict()})
    pd.DataFrame(matched_rows).rename(columns={"mase": "MASE", "mape": "MAPE_pct", "rmse": "RMSE"}).to_csv(
        OUT / "table3_calendar_related_extension.csv", index=False)

    # Qwen 891 and 890.
    qwen, extreme = qwen_tasks(qall, targets)
    qwen.to_parquet(RAW / "qwen3_4b_task_metrics.parquet", index=False)
    naive = task[task.config_id == "seasonal_naive"][["task_id", "market", "origin_date", "fold_id", "mase", "mape", "rmse", "mae"]]
    qwen_rows = []
    for name, subset in (("891 tasks", qwen), ("890 tasks, extreme removed", qwen[~qwen.task_id.isin(extreme)])):
        _m, summary = hierarchy(subset)
        same = naive[naive.task_id.isin(subset.task_id)]
        naive_summary = same.groupby("market")[["mase", "mape", "rmse"]].mean().mean()
        qwen_rows.append({"sample": name, "model": "Qwen3-4B", "tasks": len(subset), **summary.iloc[0][["mase", "mape", "rmse"]].to_dict()})
        qwen_rows.append({"sample": name, "model": "Seasonal Naive", "tasks": int(same.task_id.nunique()), **naive_summary.to_dict()})
    pd.DataFrame(qwen_rows).to_csv(OUT / "qwen3_4b_vs_seasonal_naive.csv", index=False)
    pd.Series(sorted(extreme), name="task_id").to_csv(RAW / "qwen_extreme_tasks.csv", index=False)

    # Timing panel from the completed isolated benchmark.
    cost = pd.read_csv(REPO / "track1/evaluation/volume_breakeven/cost_per_task.csv")
    ets_seconds = float(cost.loc[cost.model == "ets", "allocated_vcpu_seconds"].iloc[0])
    quotes = [("cloud_on_demand", 0.53, 0.04), ("cloud_spot_or_community", 0.33, 0.04), ("owned_or_lambda", 1.09, 0.04)]
    timing = []
    for _config, row in cost.iterrows():
        seconds = row.instance_seconds if row.device == "gpu" else row.allocated_vcpu_seconds
        rho_star = ets_seconds / seconds if row.device == "gpu" and seconds else np.nan
        item = {"model": row.model, "device": row.device, "seconds_per_vector": seconds,
                "runs": int(row.runs), "rho_star_vs_ETS": rho_star}
        for name, gpu_price, cpu_price in quotes:
            rho = gpu_price / cpu_price
            item[f"rho_{name}"] = rho
            item[f"cost_ratio_{name}"] = rho / rho_star if rho_star else np.nan
        timing.append(item)
    pd.DataFrame(timing).to_csv(OUT / "table4_timing_and_price_ratio.csv", index=False)
    scale_rows = []
    for _i, row in cost.iterrows():
        seconds = row.instance_seconds if row.device == "gpu" else row.allocated_vcpu_seconds
        if not np.isfinite(seconds):
            continue
        for n in (27, 1000, 10000):
            annual_hours = n * 365 * seconds / 3600
            scale_rows.append({"model": row.model, "series": n, "annual_hours": annual_hours, "unit": row.device})
    pd.DataFrame(scale_rows).to_csv(OUT / "table4_annual_volume.csv", index=False)

    # Asymmetric loss for the foundation quantiles.
    loss_rows = []
    for model in FOUNDATION5:
        files = list((REPO / "track1/results/foundation_outer_default").glob(f"*/{model}/predictions.parquet"))
        frame = read_many(files, ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day",
                                  "prediction", "q0.1", "q0.5", "q0.9", "model"])
        frame = frame.merge(targets_outer, on=["task_id", "horizon_day"], how="left")
        frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
        frame = frame.merge(qall, on=["market", "origin_date"], how="left")
        frame["model"] = model
        for tau, column in ((0.1, "q0.1"), (0.5, "q0.5"), (0.9, "q0.9")):
            pred = frame[column].to_numpy(float)
            actual = frame.actual.to_numpy(float)
            loss = np.where(actual >= pred, tau * (actual - pred), (1 - tau) * (pred - actual))
            part = frame[["market", "fold_id", "q_all"]].copy()
            part["loss"] = loss
            part["scaled"] = loss / part.q_all
            market_mean = part.groupby(["market", "fold_id"]).agg(loss=("loss", "mean"), scaled=("scaled", "mean")).groupby("market").mean()
            check = part.assign(abs_error=np.abs(pred - actual)).groupby(["market", "fold_id"]).mean(numeric_only=True)
            loss_rows.append({"panel": "reopening_891", "model": model, "tau": tau,
                              "loss": float(market_mean.loss.mean()), "scaled_loss": float(market_mean.scaled.mean()),
                              "mae_over_2": float((check.abs_error / 2).mean()) if tau == 0.5 else np.nan})
    stress_targets = targets[(targets.split_name == "pandemic_stress") & (targets.destination == "Macau")][
        ["task_id", "horizon_day", "actual"]]
    for model in STRESS3:
        files = list((REPO / "track1/results/sensitivity/pandemic_stress/all" / model).rglob("predictions.parquet"))
        frame = read_many(files, ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day",
                                  "prediction", "q0.1", "q0.5", "q0.9"])
        frame = frame[frame.destination == "Macau"].merge(stress_targets, on=["task_id", "horizon_day"], how="left")
        frame["origin_date"] = pd.to_datetime(frame.origin_date).dt.date
        frame = frame.merge(qall, on=["market", "origin_date"], how="left")
        for tau, column in ((0.1, "q0.1"), (0.5, "q0.5"), (0.9, "q0.9")):
            pred = frame[column].to_numpy(float)
            actual = frame.actual.to_numpy(float)
            loss = np.where(actual >= pred, tau * (actual - pred), (1 - tau) * (pred - actual))
            part = frame[["market", "fold_id"]].copy()
            part["loss"] = loss
            part["scaled"] = loss / frame.q_all.to_numpy()
            market_mean = part.groupby(["market", "fold_id"]).mean(numeric_only=True).groupby("market").mean()
            loss_rows.append({"panel": "pandemic_351", "model": model, "tau": tau,
                              "loss": float(market_mean.loss.mean()), "scaled_loss": float(market_mean.scaled.mean()),
                              "mae_over_2": np.nan})
    pd.DataFrame(loss_rows).to_csv(OUT / "table5_asymmetric_loss.csv", index=False)
    pd.Series({"zero_actual_days_excluded_from_mape": zero_days,
               "positive_actual_days_in_mape": positive_days,
               "tau_0.5_equals_mae_over_2": "checked per model in table5_asymmetric_loss.csv"}).to_csv(OUT / "quality_checks.csv")
    print("RQ1 tables written", flush=True)


if __name__ == "__main__":
    main()
