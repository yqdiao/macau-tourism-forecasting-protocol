"""4.2.4 Market size, horizon, holidays, 80% prediction intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    B, BLOCK_REOPEN, BLOCK_STRESS, CORE_IDS, FOUNDATION5, LABEL, LARGE, MEDIUM, SMALL,
    MARKET_GROUP, RAW, TABLES, REPO, RQ1, SEED, block_indices, dm_hln, log, market_then_equal,
    origin_means, stars, ensure_dirs,
)


def top10_markets(task: pd.DataFrame) -> list[str]:
    # by mean actual volume proxy: sum_actual in reopen
    vol = task[task.split_name == "outer_backtest"].groupby("market").sum_actual.mean().sort_values(ascending=False)
    return list(vol.head(10).index)


def holiday_flags() -> pd.DataFrame:
    hol = pd.read_parquet(REPO / "shared/data/holiday_calendar.parquet")
    hol["date"] = pd.to_datetime(hol["date"]).dt.date
    # ex-ante official: mainland OR macau statutory holidays (and HK public holidays as macau/source?)
    # Use mainland + macau roles as destination-relevant official holidays.
    official = hol[hol.role.isin(["mainland", "macau"])][["date"]].drop_duplicates()
    official["is_official_holiday"] = True
    return official


def wis_components(y, q10, q90):
    """WIS for central 80% interval = width/2 term style decomposition used in paper:
    dispersion = (q90-q10); over = 2/0.2 * (q10-y)_+ ; under = 2/0.2 * (y-q90)_+
    Interval score IS = (u-l) + 2/alpha*(l-y)_+ + 2/alpha*(y-u)_+ with alpha=0.2.
    """
    alpha = 0.2
    width = q90 - q10
    over = np.maximum(q10 - y, 0.0)
    under = np.maximum(y - q90, 0.0)
    iscore = width + (2.0 / alpha) * over + (2.0 / alpha) * under
    return width, (2.0 / alpha) * over, (2.0 / alpha) * under, iscore


def run():
    ensure_dirs()
    task = pd.read_parquet(RAW / "task_metrics_rq2.parquet")
    days = pd.read_parquet(RAW / "day_metrics_rq2.parquet")
    reopen = task[task.split_name == "outer_backtest"]
    ets = reopen[reopen.config_id == "ets"]

    # S6 market size groups
    rows = []
    for cid in CORE_IDS:
        part = reopen[reopen.config_id == cid]
        for gname, markets in [("Large", LARGE), ("Medium", MEDIUM), ("Small", SMALL)]:
            a = part[part.market.isin(markets)]
            e = ets[ets.market.isin(markets)]
            row = {"config_id": cid, "model": LABEL[cid], "market_group": gname, "n_markets": len(markets),
                   "tasks": int(a.task_id.nunique())}
            for col, name in [("mase", "MASE"), ("mape_pct", "MAPE"), ("mae", "MAE"), ("rmse", "RMSE")]:
                row[name] = market_then_equal(a, col)
                row[f"R_{name}_vs_ETS"] = row[name] / market_then_equal(e, col)
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES / "table_S6_market_size_groups_4metrics.csv", index=False)

    # S7 top10 bootstrap winners
    tops = top10_markets(task)
    rng = np.random.default_rng(SEED)
    s7 = []
    for market in tops:
        sub = reopen[reopen.market == market]
        origins = sorted(pd.to_datetime(sub.origin_date).unique())
        idx = block_indices(len(origins), BLOCK_REOPEN, B, rng)
        for metric, col in [("MASE", "mase"), ("MAPE", "MAPE" if False else "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
            # origin x model matrix
            mat = sub.copy()
            mat["origin_date"] = pd.to_datetime(mat.origin_date)
            wide = mat.pivot_table(index="origin_date", columns="config_id", values=col, aggfunc="mean").sort_index()
            wide = wide.reindex(origins)
            # point ranks
            point = wide.mean(axis=0).sort_values()
            best = point.index[0]; second = point.index[1]
            # bootstrap win freq
            wins = {c: 0 for c in wide.columns}
            vals = wide.to_numpy()
            cols = list(wide.columns)
            for bi in idx:
                means = vals[bi].mean(axis=0)
                wins[cols[int(np.nanargmin(means))]] += 1
            freq = {c: wins[c] / B for c in cols}
            best_f = max(freq, key=freq.get)
            s7.append({
                "market": market, "metric": metric, "point_best": best, "point_best_model": LABEL[best],
                "point_second": second, "point_second_model": LABEL[second],
                "bootstrap_winner": best_f, "bootstrap_winner_model": LABEL[best_f],
                "bootstrap_win_prob": freq[best_f],
                "status": "Clear Winner" if freq[best_f] >= 0.80 else "Statistical Tie",
                **{f"winprob_{c}": freq[c] for c in CORE_IDS},
            })
    pd.DataFrame(s7).to_csv(TABLES / "table_S7_top10_markets_bootstrap_winner_stability.csv", index=False)
    log("S6/S7 done")

    # S8 horizon weekly blocks + same-weekday degradation
    d = days[days.split_name == "outer_backtest"].copy()
    d["week_block"] = pd.cut(d.horizon_day, bins=[0, 7, 14, 21, 28], labels=["Week1", "Week2", "Week3", "Week4"])
    d["dow"] = ((d.horizon_day - 1) % 7)  # 0=Mon ... 6=Sun given Sunday origins
    s8 = []
    for cid in CORE_IDS:
        part = d[d.config_id == cid]
        ets_d = d[d.config_id == "ets"]
        for week in ["Week1", "Week2", "Week3", "Week4"]:
            a = part[part.week_block == week]
            e = ets_d[ets_d.week_block == week]
            # aggregate to task-week then market-equal: use day-level means per market-origin then equal
            def agg(frame):
                t = frame.groupby(["market", "origin_date"], sort=False).agg(
                    mae=("abs_error", "mean"), mse=("sq", "mean"), mape=("ape", "mean"), q_all=("q_all", "first")
                ).reset_index()
                t["rmse"] = np.sqrt(t.mse); t["mase"] = t.mae / t.q_all; t["mape_pct"] = t.mape * 100
                return t
            ta, te = agg(a), agg(e)
            row = {"config_id": cid, "model": LABEL[cid], "week_block": week}
            for col, name in [("mase", "MASE"), ("mape_pct", "MAPE"), ("mae", "MAE"), ("rmse", "RMSE")]:
                row[name] = market_then_equal(ta, col)
                row[f"R_{name}_vs_ETS"] = row[name] / market_then_equal(te, col)
            s8.append(row)
        # same-weekday degradation Week4/Week1
        deg = {}
        for col, name in [("abs_error", "MAE"), ("ape", "MAPE"), ("sq", "RMSE")]:
            ratios = []
            for dow in range(7):
                w1 = part[(part.dow == dow) & (part.horizon_day <= 7)]
                w4 = part[(part.dow == dow) & (part.horizon_day >= 22)]
                if col == "sq":
                    e1 = float(np.sqrt(w1[col].mean())); e4 = float(np.sqrt(w4[col].mean()))
                elif col == "ape":
                    e1 = float(w1[col].mean()); e4 = float(w4[col].mean())
                else:
                    e1 = float(w1[col].mean()); e4 = float(w4[col].mean())
                if e1 > 0:
                    ratios.append(e4 / e1)
            deg[name] = float(np.mean(ratios)) if ratios else np.nan
        # MASE degradation via abs_error/q_all daywise
        ratios = []
        for dow in range(7):
            w1 = part[(part.dow == dow) & (part.horizon_day <= 7)]
            w4 = part[(part.dow == dow) & (part.horizon_day >= 22)]
            e1 = float((w1.abs_error / w1.q_all).mean()); e4 = float((w4.abs_error / w4.q_all).mean())
            if e1 > 0:
                ratios.append(e4 / e1)
        deg["MASE"] = float(np.mean(ratios)) if ratios else np.nan
        s8.append({"config_id": cid, "model": LABEL[cid], "week_block": "SameWeekday_W4_over_W1", **{k: deg[k] for k in ["MASE","MAPE","MAE","RMSE"]}})
    pd.DataFrame(s8).to_csv(TABLES / "table_S8_horizon_weekly_blocks_and_same_weekday.csv", index=False)
    log("S8 done")

    # S9 high demand
    official = holiday_flags()
    d = days[days.split_name == "outer_backtest"].copy()
    d["target_date"] = pd.to_datetime(d["target_date"]).dt.date
    d = d.merge(official, left_on="target_date", right_on="date", how="left")
    d["is_official_holiday"] = d["is_official_holiday"].fillna(False)
    # definition 2: ets prediction in top 10% within market
    ets_day = d[d.config_id == "ets"][["market", "task_id", "horizon_day", "prediction"]].rename(columns={"prediction": "ets_pred"})
    d = d.merge(ets_day, on=["market", "task_id", "horizon_day"], how="left")
    thr2 = d.groupby("market").ets_pred.quantile(0.90)
    d["is_ref_high"] = d.ets_pred >= d.market.map(thr2)
    thr3 = d.groupby("market").actual.quantile(0.90)
    d["is_expost_p90"] = d.actual >= d.market.map(thr3)
    s9 = []
    for cid in CORE_IDS:
        part = d[d.config_id == cid]
        for rule, mask_col in [("exante_official_holiday", "is_official_holiday"),
                               ("exante_ets_top10", "is_ref_high"),
                               ("expost_actual_p90", "is_expost_p90"),
                               ("ordinary_non_holiday", None)]:
            if mask_col is None:
                sub = part[~part.is_official_holiday]
            else:
                sub = part[part[mask_col]]
            if sub.empty:
                continue
            y, yhat = sub.actual.to_numpy(), sub.prediction.to_numpy()
            err = yhat - y
            under = err < 0
            severe = np.where(y > 0, err / y < -0.20, False)
            row = {
                "config_id": cid, "model": LABEL[cid], "high_demand_rule": rule,
                "n_days": len(sub),
                "MAE": float(np.abs(err).mean()),
                "RMSE": float(np.sqrt((err ** 2).mean())),
                "MAPE_pct": float((np.abs(err)[y > 0] / y[y > 0]).mean() * 100) if (y > 0).any() else np.nan,
                "MASE": float((np.abs(err) / sub.q_all.to_numpy()).mean()),
                "underprediction_rate": float(under.mean()),
                "severe_under_rate_20pct": float(np.mean(severe)),
                "cond_shortfall_people": float((-err[under]).mean()) if under.any() else np.nan,
                "cond_shortfall_pct": float(((-err[under]) / y[under])[y[under] > 0].mean() * 100) if under.any() and (y[under] > 0).any() else np.nan,
            }
            s9.append(row)
    pd.DataFrame(s9).to_csv(TABLES / "table_S9_high_demand_holiday_underprediction_risk.csv", index=False)
    log("S9 done")

    # S10 prediction intervals from five_quantile_runs
    runs = RQ1 / "raw/five_quantile_runs"
    targets = pd.read_parquet(REPO / "shared/data/targets.parquet",
                              columns=["task_id", "horizon_day", "actual", "split_name", "destination"])
    targets = targets[targets.destination == "Macau"]
    hol = holiday_flags()
    s10 = []
    for split, models in [("outer_backtest", FOUNDATION5), ("pandemic_stress", ["granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"])]:
        split_targets = targets[targets.split_name == split][["task_id", "horizon_day", "actual"]]
        for model in models:
            files = sorted((runs / split / model).glob("*/predictions.parquet"))
            if not files:
                # require five-quantile predictions for each requested model
                log(f"S10 missing {split} {model}")
                continue
            cols = ["task_id", "destination", "market", "fold_id", "origin_date", "horizon_day", "target_date", "q0.1", "q0.9"]
            frame = pd.concat([pd.read_parquet(f, columns=[c for c in cols if c in pd.read_parquet(f).columns]) for f in files], ignore_index=True)
            if "destination" in frame.columns:
                frame = frame[frame.destination == "Macau"]
            frame = frame.merge(split_targets, on=["task_id", "horizon_day"], how="inner")
            frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
            frame = frame.merge(hol, left_on="target_date", right_on="date", how="left")
            frame["is_official_holiday"] = frame["is_official_holiday"].fillna(False)
            frame["week_block"] = pd.cut(frame.horizon_day, bins=[0, 7, 14, 21, 28], labels=["Week1", "Week2", "Week3", "Week4"])
            y = frame.actual.to_numpy(); lo = frame["q0.1"].to_numpy(); hi = frame["q0.9"].to_numpy()
            width, over, under, iscore = wis_components(y, lo, hi)
            cover = (y >= lo) & (y <= hi)
            for subset, mask in [
                ("overall", np.ones(len(frame), dtype=bool)),
                ("official_holiday", frame.is_official_holiday.to_numpy()),
                ("non_holiday", ~frame.is_official_holiday.to_numpy()),
                ("Week1", (frame.week_block == "Week1").to_numpy()),
                ("Week4", (frame.week_block == "Week4").to_numpy()),
            ]:
                if mask.sum() == 0:
                    continue
                s10.append({
                    "split_name": split, "model": model, "display_name": LABEL[model], "subset": subset,
                    "n_days": int(mask.sum()),
                    "coverage80": float(cover[mask].mean()),
                    "mean_width": float(width[mask].mean()),
                    "WIS": float(iscore[mask].mean()),
                    "WIS_dispersion": float(width[mask].mean()),
                    "WIS_overprediction_penalty": float(over[mask].mean()),
                    "WIS_underprediction_penalty": float(under[mask].mean()),
                })
            log(f"S10 {split} {model}")
    pd.DataFrame(s10).to_csv(TABLES / "table_S10_prediction_interval_80_coverage_and_wis.csv", index=False)
    log("wrote S6-S10")


if __name__ == "__main__":
    run()
