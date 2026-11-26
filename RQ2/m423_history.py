"""4.2.3 History length and MASE denominator control."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    B, BLOCK_REOPEN, LABEL, RAW, TABLES, SEED, block_indices, dm_hln, log, market_then_equal,
    origin_means, stars, ensure_dirs,
)

CONTEXT_LIMIT = {
    "granite_patchtst_fm": 512,
    "timesfm_3": None,
    "chronos_2": None,
    "moirai": None,
    "lag_llama": 512,
    "ets": None,
    "seasonal_naive": 7,
    "svr": None,
    "lightgbm": None,
}

HIST_LABEL = {"all": "Full-with-COVID", "365": "Clean-365", "180": "Clean-180"}


def run():
    ensure_dirs()
    path = RAW / "history_task_metrics.parquet"
    if not path.exists():
        raise SystemExit("run build_panels.build_history_panels first")
    task = pd.read_parquet(path)
    models = sorted(task.config_id.unique())
    rows = []
    rng = np.random.default_rng(SEED)
    base = task[task.history_setting == "all"]
    for cid in models:
        for hist in ["180", "365", "all"]:
            part = task[(task.config_id == cid) & (task.history_setting == hist)]
            if part.empty:
                rows.append({"config_id": cid, "model": LABEL.get(cid, cid), "history_setting": HIST_LABEL[hist],
                             "status": "missing"})
                continue
            ref = base[base.config_id == cid]
            item = {
                "config_id": cid, "model": LABEL.get(cid, cid),
                "history_setting": HIST_LABEL[hist], "history_code": hist,
                "tasks": int(part.task_id.nunique()),
                "MAE": market_then_equal(part, "mae"),
                "MAPE_pct": market_then_equal(part, "mape_pct"),
                "RMSE": market_then_equal(part, "rmse"),
                "MASE_fixed_ref_denom": market_then_equal(part, "mase_fixed"),
                "MASE_own_denom": market_then_equal(part, "mase_own"),
                "context_length_limit": CONTEXT_LIMIT.get(cid),
                "status": "ok",
            }
            if not ref.empty:
                for key, col in [("MAE", "mae"), ("MAPE_pct", "mape_pct"), ("RMSE", "rmse"),
                                 ("MASE_fixed_ref_denom", "mase_fixed")]:
                    item[f"pct_change_vs_Full_{key}"] = 100.0 * (item[key] / market_then_equal(ref, col) - 1.0)
                # DM vs Full on fixed MASE origin means
                om = origin_means(part, "mase_fixed")
                of = origin_means(ref, "mase_fixed")
                idx = om.index.intersection(of.index)
                dm, p = dm_hln(om.loc[idx], of.loc[idx], lag=2)
                item["DM_MASEfixed_vs_Full"] = dm
                item["p_MASEfixed_vs_Full"] = p
                item["sig_MASEfixed_vs_Full"] = stars(p)
            rows.append(item)
            log(f"S5 {cid} {hist}")
    s5 = pd.DataFrame(rows)
    # seasonal naive identity check
    sn = s5[s5.config_id == "seasonal_naive"].copy()
    if len(sn) >= 2:
        mae_range = sn.MAE.max() - sn.MAE.min()
        log(f"Seasonal Naive MAE range across histories = {mae_range:.6g} (expect ~0)")
    s5.to_csv(TABLES / "table_S5_history_length_and_denominator_control.csv", index=False)
    log("wrote S5")


if __name__ == "__main__":
    run()
