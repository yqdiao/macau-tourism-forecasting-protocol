"""4.2.2 Metric concordance, market decomposition, R(alpha) spectrum."""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kendalltau

from common import (
    B, BLOCK_REOPEN, BLOCK_STRESS, CORE_IDS, LABEL, LARGE, MEDIUM, SMALL, MARKET_GROUP,
    RAW, TABLES, SEED, block_indices, log, market_then_equal, origin_means, ensure_dirs,
)


def total_destination_metrics(days: pd.DataFrame) -> pd.DataFrame:
    """Sum 27 markets per origin-day then task-aggregate MAE/RMSE per origin."""
    g = days.groupby(["config_id", "split_name", "origin_date", "horizon_day", "target_date"], sort=False).agg(
        actual=("actual", "sum"), prediction=("prediction", "sum")
    ).reset_index()
    g["abs_error"] = (g.prediction - g.actual).abs()
    g["sq"] = (g.prediction - g.actual) ** 2
    task = g.groupby(["config_id", "split_name", "origin_date"], sort=False).agg(
        mae=("abs_error", "mean"), mse=("sq", "mean")
    ).reset_index()
    task["rmse"] = np.sqrt(task.mse)
    return task


def concordance(task: pd.DataFrame, days: pd.DataFrame, split: str) -> pd.DataFrame:
    part = task[task.split_name == split]
    totals = total_destination_metrics(days[days.split_name == split])
    scores = {}
    for cid in CORE_IDS:
        sub = part[part.config_id == cid]
        scores[cid] = {
            "MASE": market_then_equal(sub, "mase"),
            "MAPE": market_then_equal(sub, "mape_pct"),
            "MAE": market_then_equal(sub, "mae"),
            "RMSE": market_then_equal(sub, "rmse"),
            "Total_MAE": float(totals[totals.config_id == cid].mae.mean()),
            "Total_RMSE": float(totals[totals.config_id == cid].rmse.mean()),
        }
    metrics = ["MASE", "MAPE", "MAE", "RMSE", "Total_MAE", "Total_RMSE"]
    rows = []
    for a, b in combinations(metrics, 2):
        xa = [scores[c][a] for c in CORE_IDS]
        xb = [scores[c][b] for c in CORE_IDS]
        tau, p = kendalltau(xa, xb)
        # discordant pairs among 14 models
        disc = 0
        pairs = 0
        for i, j in combinations(range(len(CORE_IDS)), 2):
            pairs += 1
            da = np.sign(xa[i] - xa[j])
            db = np.sign(xb[i] - xb[j])
            if da != 0 and db != 0 and da != db:
                disc += 1
        rows.append({"split_name": split, "metric_a": a, "metric_b": b,
                     "kendall_tau": float(tau), "kendall_p": float(p),
                     "discordant_pairs": disc, "pairs": pairs,
                     "discordant_ratio": disc / pairs})
    # also store absolute scores
    score_rows = [{"split_name": split, "config_id": c, "model": LABEL[c], **scores[c]} for c in CORE_IDS]
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def contrib_pair(task: pd.DataFrame, split: str, model_a: str, model_b: str) -> pd.DataFrame:
    a = task[(task.split_name == split) & (task.config_id == model_a)]
    b = task[(task.split_name == split) & (task.config_id == model_b)]
    rows = []
    for market in sorted(a.market.unique()):
        am = a[a.market == market]
        bm = b[b.market == market]
        row = {"split_name": split, "model_a": model_a, "model_b": model_b, "market": market,
               "market_group": MARKET_GROUP.get(market, "?")}
        for col, name in [("mae", "MAE"), ("mase", "MASE"), ("mape_pct", "MAPE"), ("rmse", "RMSE")]:
            da = float(am[col].mean() - bm[col].mean())
            row[f"delta_{name}"] = da
        rows.append(row)
    frame = pd.DataFrame(rows)
    # group shares for MAE/MASE (equal-market: each market weight 1/27)
    for name in ["MAE", "MASE", "MAPE", "RMSE"]:
        total = frame[f"delta_{name}"].mean()  # equal-market mean of deltas = delta of means
        frame[f"share_{name}_pct"] = frame[f"delta_{name}"] / (total * 27) * 100.0 if total != 0 else np.nan
    # verify
    for name in ["MAE", "MASE", "MAPE", "RMSE"]:
        s = frame[f"share_{name}_pct"].sum()
        frame[f"share_{name}_sum_check"] = s
    return frame


def spectrum(task: pd.DataFrame, split: str, model_a: str, model_b: str) -> pd.DataFrame:
    a = task[(task.split_name == split) & (task.config_id == model_a)].groupby("market").mase.mean()
    b = task[(task.split_name == split) & (task.config_id == model_b)].groupby("market").mase.mean()
    q = task[(task.split_name == split) & (task.config_id == model_a)].groupby("market").q_all.mean()
    markets = a.index.intersection(b.index).intersection(q.index)
    a, b, q = a.loc[markets], b.loc[markets], q.loc[markets]
    rows = []
    alphas = np.round(np.arange(0, 1.0001, 0.02), 2)
    # bootstrap at market? protocol: origin block bootstrap of R(alpha)
    # Use origin-level weighted? Simpler: market-level with origin bootstrap of market means.
    base = task[task.split_name == split]
    origins = sorted(pd.to_datetime(base.origin_date).unique())
    block = BLOCK_REOPEN if split == "outer_backtest" else BLOCK_STRESS
    rng = np.random.default_rng(SEED + hash((split, model_a, model_b)) % 1_000_000)
    # precompute market x origin mase for A/B and q
    def mat(cid):
        part = base[base.config_id == cid].copy()
        part["origin_date"] = pd.to_datetime(part.origin_date)
        return part.pivot_table(index="origin_date", columns="market", values="mase", aggfunc="mean").sort_index()
    Ma, Mb = mat(model_a), mat(model_b)
    common_o = Ma.index.intersection(Mb.index)
    Ma, Mb = Ma.loc[common_o], Mb.loc[common_o]
    qv = q.reindex(Ma.columns).to_numpy()
    idx = block_indices(len(common_o), block, B, rng)
    for alpha in alphas:
        w = np.power(qv, alpha)
        w = w / w.sum()
        # point
        ra = (Ma.mean(axis=0).to_numpy() * w).sum()
        rb = (Mb.mean(axis=0).to_numpy() * w).sum()
        r = ra / rb
        # bootstrap
        boots = []
        for bi in idx:
            ra_b = (Ma.to_numpy()[bi].mean(axis=0) * w).sum()
            rb_b = (Mb.to_numpy()[bi].mean(axis=0) * w).sum()
            boots.append(ra_b / rb_b)
        boots = np.asarray(boots)
        rows.append({"split_name": split, "model_a": model_a, "model_b": model_b, "alpha": float(alpha),
                     "R_alpha": float(r), "ci95_low": float(np.quantile(boots, 0.025)),
                     "ci95_high": float(np.quantile(boots, 0.975))})
    frame = pd.DataFrame(rows)
    # alpha* where R crosses 1 (linear interp on grid)
    cross = frame.copy()
    sign = np.sign(cross.R_alpha.to_numpy() - 1.0)
    alpha_star = np.nan
    for i in range(len(sign) - 1):
        if sign[i] == 0:
            alpha_star = float(cross.alpha.iloc[i]); break
        if sign[i] * sign[i + 1] < 0:
            x0, x1 = cross.alpha.iloc[i], cross.alpha.iloc[i + 1]
            y0, y1 = cross.R_alpha.iloc[i], cross.R_alpha.iloc[i + 1]
            alpha_star = float(x0 + (1 - y0) * (x1 - x0) / (y1 - y0))
            break
    frame["alpha_star"] = alpha_star
    return frame


def extreme_concentration(days: pd.DataFrame, split: str) -> pd.DataFrame:
    part = days[days.split_name == split]
    rows = []
    for cid in CORE_IDS:
        d = part[part.config_id == cid].copy()
        # task-level rmse/mae
        t = d.groupby("task_id").agg(mae=("abs_error", "mean"), mse=("sq", "mean")).reset_index()
        t["rmse_over_mae"] = np.sqrt(t.mse) / t.mae
        abs_e = d.abs_error.to_numpy()
        sq = d.sq.to_numpy()
        order = np.argsort(-abs_e)
        n = len(abs_e)
        sae = abs_e.sum(); sse = sq.sum()
        row = {"split_name": split, "config_id": cid, "model": LABEL[cid],
               "mean_task_rmse_over_mae": float(t.rmse_over_mae.mean())}
        for pct in (1, 5, 10):
            k = max(1, int(round(n * pct / 100)))
            top = order[:k]
            row[f"top{pct}pct_share_SAE"] = float(abs_e[top].sum() / sae)
            row[f"top{pct}pct_share_SSE"] = float(sq[top].sum() / sse)
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    ensure_dirs()
    task = pd.read_parquet(RAW / "task_metrics_rq2.parquet")
    days = pd.read_parquet(RAW / "day_metrics_rq2.parquet")
    conc_rows, score_rows = [], []
    for split in ["outer_backtest", "pandemic_stress"]:
        c, s = concordance(task, days, split)
        conc_rows.append(c); score_rows.append(s)
    s3 = pd.concat(conc_rows, ignore_index=True)
    scores = pd.concat(score_rows, ignore_index=True)
    s3.to_csv(TABLES / "table_S3_metric_concordance_matrix.csv", index=False)
    scores.to_csv(RAW / "metric_scores_by_model.csv", index=False)

    pairs = [
        ("pandemic_stress", "granite_patchtst_fm", "ets"),
        ("pandemic_stress", "timesfm_3", "ets"),
        ("outer_backtest", "granite_patchtst_fm", "timesfm_3"),
    ]
    decs, specs = [], []
    for split, a, b in pairs:
        decs.append(contrib_pair(task, split, a, b))
        specs.append(spectrum(task, split, a, b))
        log(f"decomp/spectrum {split} {a}/{b}")
    # also spectrum for chronos/ets stress
    specs.append(spectrum(task, "pandemic_stress", "chronos_2", "ets"))
    dec = pd.concat(decs, ignore_index=True)
    spec = pd.concat(specs, ignore_index=True)
    ext = pd.concat([extreme_concentration(days, s) for s in ["outer_backtest", "pandemic_stress"]], ignore_index=True)
    # group rollups
    group_rows = []
    for (split, a, b), part in dec.groupby(["split_name", "model_a", "model_b"]):
        for gname, markets in [("Large", LARGE), ("Medium", MEDIUM), ("Small", SMALL),
                               ("Mainland", ["CN_Chinese_mainland"]), ("HongKong", ["HK_Hong_Kong_SAR"])]:
            sub = part[part.market.isin(markets)]
            row = {"split_name": split, "model_a": a, "model_b": b, "group": gname, "n_markets": len(sub)}
            for name in ["MAE", "MASE", "MAPE", "RMSE"]:
                row[f"delta_{name}"] = sub[f"delta_{name}"].mean()  # equal within group then? keep sum of market deltas / 27
                row[f"contrib_{name}"] = sub[f"delta_{name}"].sum() / 27.0
                row[f"share_{name}_pct"] = sub[f"share_{name}_pct"].sum()
            group_rows.append(row)
    groups = pd.DataFrame(group_rows)
    # write combined S4
    # store as multi-section CSV with a section column
    dec["section"] = "market_detail"
    groups["section"] = "group_summary"
    groups["market"] = groups["group"]
    spec["section"] = "spectrum"
    ext["section"] = "extreme_concentration"
    # align columns loosely via concat on separate files + one binder
    dec.to_csv(RAW / "s4_market_decomposition_detail.csv", index=False)
    groups.to_csv(RAW / "s4_group_contribution.csv", index=False)
    spec.to_csv(RAW / "s4_spectrum.csv", index=False)
    ext.to_csv(RAW / "s4_extreme_concentration.csv", index=False)
    # published S4: group summary + alpha_star + extremes key cols
    alpha_star = spec.groupby(["split_name", "model_a", "model_b"], as_index=False).alpha_star.first()
    alpha_star["section"] = "alpha_star"
    s4 = pd.concat([
        groups,
        alpha_star,
        ext,
        spec[spec.alpha.isin([0.0, 0.5, 1.0])],
    ], ignore_index=True, sort=False)
    s4.to_csv(TABLES / "table_S4_market_decomposition_and_spectrum.csv", index=False)
    log("wrote S3/S4")


if __name__ == "__main__":
    run()
