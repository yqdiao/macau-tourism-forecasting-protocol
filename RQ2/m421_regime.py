"""4.2.1 Regime comparison: reopen vs pandemic stress."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    B, BLOCK_REOPEN, BLOCK_STRESS, CORE_IDS, LABEL, FAMILY, MARKET_GROUP, RAW, TABLES, SEED,
    block_indices, dm_hln, log, market_then_equal, origin_means, stars, ensure_dirs,
)


def _panel():
    return pd.read_parquet(RAW / "task_metrics_rq2.parquet")


def joint_ratio_ci(task_m, task_e, metric, block, rng):
    om = origin_means(task_m, metric)
    oe = origin_means(task_e, metric)
    idx = om.index.intersection(oe.index)
    om, oe = om.loc[idx], oe.loc[idx]
    t = len(om)
    ii = block_indices(t, block, B, rng)
    ratios = om.to_numpy()[ii].mean(axis=1) / oe.to_numpy()[ii].mean(axis=1)
    return float(om.mean() / oe.mean()), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)), om, oe


def mean_of_ratios(task_m, task_e, metric):
    om = origin_means(task_m, metric)
    oe = origin_means(task_e, metric)
    idx = om.index.intersection(oe.index)
    return float((om.loc[idx] / oe.loc[idx]).mean())


def run():
    ensure_dirs()
    task = _panel()
    metrics = [("mase", "MASE"), ("mape_pct", "MAPE"), ("mae", "MAE"), ("rmse", "RMSE")]
    rng = np.random.default_rng(SEED)
    rows = []
    stress_folds = sorted(task[task.split_name == "pandemic_stress"].fold_id.unique())
    # stage assignment by chronological order of origins
    stress_origins = (
        task[task.split_name == "pandemic_stress"][["fold_id", "origin_date"]]
        .drop_duplicates().sort_values("origin_date")
    )
    stage_map = {}
    n = len(stress_origins)
    for i, r in enumerate(stress_origins.itertuples(index=False)):
        if i < n // 3:
            stage_map[r.fold_id] = "early"
        elif i < 2 * n // 3:
            stage_map[r.fold_id] = "mid"
        else:
            stage_map[r.fold_id] = "late"

    for cid in CORE_IDS:
        reopen = task[(task.split_name == "outer_backtest") & (task.config_id == cid)]
        stress = task[(task.split_name == "pandemic_stress") & (task.config_id == cid)]
        ets_r = task[(task.split_name == "outer_backtest") & (task.config_id == "ets")]
        ets_s = task[(task.split_name == "pandemic_stress") & (task.config_id == "ets")]
        item = {"config_id": cid, "model": LABEL[cid], "family": FAMILY[cid],
                "reopen_tasks": int(reopen.task_id.nunique()),
                "stress_tasks": int(stress.task_id.nunique())}
        for col, name in metrics:
            item[f"reopen_{name}"] = market_then_equal(reopen, col)
            item[f"stress_{name}"] = market_then_equal(stress, col)
            # ratio of means + CI
            r_pt, r_lo, r_hi, om_r, oe_r = joint_ratio_ci(reopen, ets_r, col, BLOCK_REOPEN, rng)
            s_pt, s_lo, s_hi, om_s, oe_s = joint_ratio_ci(stress, ets_s, col, BLOCK_STRESS, rng)
            item[f"R_{name}_reopen"] = r_pt
            item[f"R_{name}_reopen_ci95_low"] = r_lo
            item[f"R_{name}_reopen_ci95_high"] = r_hi
            item[f"R_{name}_stress"] = s_pt
            item[f"R_{name}_stress_ci95_low"] = s_lo
            item[f"R_{name}_stress_ci95_high"] = s_hi
            item[f"R_{name}_reopen_mean_of_ratios"] = mean_of_ratios(reopen, ets_r, col)
            item[f"R_{name}_stress_mean_of_ratios"] = mean_of_ratios(stress, ets_s, col)
            # D = R_stress - R_reopen with independent joint bootstraps
            rr = om_r.to_numpy()[block_indices(len(om_r), BLOCK_REOPEN, B, rng)].mean(1) / oe_r.to_numpy()[block_indices(len(oe_r), BLOCK_REOPEN, B, rng)].mean(1)
            # Fix: use same rng draws properly — regenerate paired independently
            rng2 = np.random.default_rng(SEED + hash((cid, name, "D")) % 10_000_000)
            ir = block_indices(len(om_r), BLOCK_REOPEN, B, rng2)
            is_ = block_indices(len(om_s), BLOCK_STRESS, B, rng2)
            boot_r = om_r.to_numpy()[ir].mean(1) / oe_r.to_numpy()[ir].mean(1)
            boot_s = om_s.to_numpy()[is_].mean(1) / oe_s.to_numpy()[is_].mean(1)
            d_boot = boot_s - boot_r
            item[f"D_{name}"] = s_pt - r_pt
            item[f"D_{name}_ci95_low"] = float(np.quantile(d_boot, 0.025))
            item[f"D_{name}_ci95_high"] = float(np.quantile(d_boot, 0.975))
            # DM vs ETS on origin means
            dm, p = dm_hln(om_r, oe_r, lag=2)
            item[f"DM_{name}_reopen_vs_ETS"] = dm
            item[f"p_{name}_reopen_vs_ETS"] = p
            item[f"sig_{name}_reopen_vs_ETS"] = stars(p)
            dm, p = dm_hln(om_s, oe_s, lag=2)
            item[f"DM_{name}_stress_vs_ETS"] = dm
            item[f"p_{name}_stress_vs_ETS"] = p
            item[f"sig_{name}_stress_vs_ETS"] = stars(p)
            # markets beating ETS
            m_r = reopen.groupby("market")[col].mean()
            e_r = ets_r.groupby("market")[col].mean()
            m_s = stress.groupby("market")[col].mean()
            e_s = ets_s.groupby("market")[col].mean()
            item[f"markets_beat_ETS_{name}_reopen"] = int((m_r < e_r.reindex(m_r.index)).sum())
            item[f"markets_beat_ETS_{name}_stress"] = int((m_s < e_s.reindex(m_s.index)).sum())
            # LOOO on stress
            looo = []
            for fold in stress_folds:
                sub = stress[stress.fold_id != fold]
                ets_sub = ets_s[ets_s.fold_id != fold]
                looo.append(market_then_equal(sub, col) / market_then_equal(ets_sub, col))
            item[f"R_{name}_stress_LOOO_min"] = float(np.min(looo))
            item[f"R_{name}_stress_LOOO_max"] = float(np.max(looo))
        rows.append(item)
        log(f"S1 row {cid}")

    s1 = pd.DataFrame(rows)
    s1.to_csv(TABLES / "table_S1_regime_comparison.csv", index=False)

    # S2 stage x market-group decomposition (delta vs ETS on MAE/MASE/MAPE/RMSE)
    dec = []
    stress = task[task.split_name == "pandemic_stress"].copy()
    stress["stage"] = stress.fold_id.map(stage_map)
    ets = stress[stress.config_id == "ets"]
    for cid in CORE_IDS:
        part = stress[stress.config_id == cid]
        for stage in ["early", "mid", "late"]:
            for group in ["Large", "Medium", "Small"]:
                a = part[(part.stage == stage) & (part.market_group == group)]
                b = ets[(ets.stage == stage) & (ets.market_group == group)]
                if a.empty or b.empty:
                    continue
                row = {"config_id": cid, "model": LABEL[cid], "stage": stage, "market_group": group,
                       "tasks": int(a.task_id.nunique())}
                for col, name in metrics:
                    row[f"{name}_model"] = market_then_equal(a, col)
                    row[f"{name}_ETS"] = market_then_equal(b, col)
                    row[f"delta_{name}"] = row[f"{name}_model"] - row[f"{name}_ETS"]
                dec.append(row)
    pd.DataFrame(dec).to_csv(TABLES / "table_S2_stress_stage_market_decomposition.csv", index=False)
    log("wrote S1/S2")
    return s1


if __name__ == "__main__":
    run()
