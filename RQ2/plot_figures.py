"""Figures 5–10 for RQ2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, LABEL, TABLES, FIGURES, RAW, HEADLINE, style_axes, save_fig, log, ensure_dirs


def fig5(s1: pd.DataFrame):
    # sort by reopen MASE ascending
    s1 = s1.sort_values("reopen_MASE")
    metrics = ["MASE", "MAPE", "MAE", "RMSE"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 7), sharey=True)
    y = np.arange(len(s1))
    for ax, metric, letter in zip(axes, metrics, "abcd"):
        r = s1[f"R_{metric}_reopen"]; rlo = s1[f"R_{metric}_reopen_ci95_low"]; rhi = s1[f"R_{metric}_reopen_ci95_high"]
        s = s1[f"R_{metric}_stress"]; slo = s1[f"R_{metric}_stress_ci95_low"]; shi = s1[f"R_{metric}_stress_ci95_high"]
        ax.errorbar(r, y - 0.12, xerr=[r - rlo, rhi - r], fmt="o", color="#222222", ms=4, label="Reopening", capsize=2)
        ax.errorbar(s, y + 0.12, xerr=[s - slo, shi - s], fmt="D", mfc="white", mec="#E66101", color="#E66101",
                    ms=4, label="Pandemic stress", capsize=2)
        ax.axvline(1.0, color="grey", ls="--", lw=1)
        ax.set_xlabel(f"R_{metric} vs ETS")
        ax.set_title(f"{letter}", loc="left", fontweight="bold")
        style_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s1.model)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "figure5_regime_forest_4metrics")
    plt.close(fig)


def fig6():
    groups = pd.read_csv(RAW / "s4_group_contribution.csv")
    spec = pd.read_csv(RAW / "s4_spectrum.csv")
    ext = pd.read_csv(RAW / "s4_extreme_concentration.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # a: stress granite-ets group contrib MAE vs MASE
    g = groups[(groups.split_name == "pandemic_stress") & (groups.model_a == "granite_patchtst_fm") & (groups.model_b == "ets")]
    g = g[g.group.isin(["Mainland", "HongKong", "Medium", "Small"])]
    x = np.arange(len(g))
    axes[0, 0].bar(x - 0.15, g.contrib_MAE, 0.3, label="ΔMAE contrib", color="#E66101")
    axes[0, 0].bar(x + 0.15, g.contrib_MASE, 0.3, label="ΔMASE contrib", color="#7B3294")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(g.group, rotation=15)
    axes[0, 0].axhline(0, color="grey", lw=0.8)
    axes[0, 0].set_title("a", loc="left", fontweight="bold")
    axes[0, 0].legend(fontsize=7); style_axes(axes[0, 0])
    # b: spectrum
    ax = axes[0, 1]
    for a, b, color, name in [
        ("granite_patchtst_fm", "ets", "#E66101", "Granite/ETS"),
        ("timesfm_3", "ets", "#7B3294", "TimesFM/ETS"),
        ("chronos_2", "ets", "#1F77B4", "Chronos/ETS"),
    ]:
        part = spec[(spec.split_name == "pandemic_stress") & (spec.model_a == a) & (spec.model_b == b)].sort_values("alpha")
        ax.plot(part.alpha, part.R_alpha, color=color, label=name)
        ax.fill_between(part.alpha, part.ci95_low, part.ci95_high, color=color, alpha=0.15)
        star = part.alpha_star.iloc[0]
        if np.isfinite(star):
            ax.axvline(star, color=color, ls=":", lw=1)
    ax.axhline(1.0, color="grey", ls="--")
    ax.set_xlabel("alpha"); ax.set_ylabel("R(alpha)")
    ax.set_title("b", loc="left", fontweight="bold"); ax.legend(fontsize=7); style_axes(ax)
    # c: reopen granite vs timesfm group
    g = groups[(groups.split_name == "outer_backtest") & (groups.model_a == "granite_patchtst_fm") & (groups.model_b == "timesfm_3")]
    g = g[g.group.isin(["Mainland", "HongKong", "Medium", "Small"])]
    x = np.arange(len(g)); width = 0.2
    for i, (name, color) in enumerate([("MAE", "#E66101"), ("MASE", "#7B3294"), ("MAPE", "#1F77B4"), ("RMSE", "#2CA02C")]):
        axes[1, 0].bar(x + (i - 1.5) * width, g[f"contrib_{name}"], width, label=name, color=color)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(g.group, rotation=15)
    axes[1, 0].axhline(0, color="grey", lw=0.8)
    axes[1, 0].set_title("c", loc="left", fontweight="bold"); axes[1, 0].legend(fontsize=7); style_axes(axes[1, 0])
    # d: extreme concentration top5/10 SSE
    e = ext[ext.split_name == "outer_backtest"]
    e = e[e.config_id.isin(["granite_patchtst_fm", "timesfm_3", "chronos_2", "ets", "LL01"])]
    x = np.arange(len(e))
    axes[1, 1].bar(x - 0.15, e.top5pct_share_SSE, 0.3, label="Top5% SSE", color="#7B3294")
    axes[1, 1].bar(x + 0.15, e.top10pct_share_SSE, 0.3, label="Top10% SSE", color="#E66101")
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(e.model, rotation=20, ha="right")
    axes[1, 1].set_title("d", loc="left", fontweight="bold"); axes[1, 1].legend(fontsize=7); style_axes(axes[1, 1])
    fig.tight_layout(); save_fig(fig, "figure6_metric_divergence_mechanism"); plt.close(fig)


def fig7(s5: pd.DataFrame):
    s5 = s5[s5.status == "ok"].copy()
    models = [m for m in ["granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "ets", "seasonal_naive", "svr", "lightgbm"] if m in set(s5.config_id)]
    order = ["Clean-180", "Clean-365", "Full-with-COVID"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for cid in models:
        part = s5[s5.config_id == cid].set_index("history_setting").reindex(order)
        color = COLORS.get(LABEL.get(cid, cid), "#333333")
        axes[0].plot(order, part.MAE, marker="o", label=LABEL.get(cid, cid), color=color)
        axes[0].plot(order, part.MAPE_pct, marker="x", ls="--", color=color, alpha=0.7)
        axes[1].plot(order, part.MASE_own_denom, marker="o", color=color, label=LABEL.get(cid, cid))
        axes[2].plot(order, part.MASE_fixed_ref_denom, marker="o", color=color, label=LABEL.get(cid, cid))
    for ax, letter, ylab in zip(axes, "abc", ["MAE (solid) / MAPE% (dashed)", "MASE own denom", "MASE fixed q_all"]):
        ax.set_title(letter, loc="left", fontweight="bold"); ax.set_ylabel(ylab); style_axes(ax)
        ax.tick_params(axis="x", rotation=15)
    axes[2].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout(); save_fig(fig, "figure7_history_length_and_covid_contamination"); plt.close(fig)


def fig8():
    s6 = pd.read_csv(TABLES / "table_S6_market_size_groups_4metrics.csv")
    s7 = pd.read_csv(TABLES / "table_S7_top10_markets_bootstrap_winner_stability.csv")
    s8 = pd.read_csv(TABLES / "table_S8_horizon_weekly_blocks_and_same_weekday.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # a heatmap R_MASE top10 winners annotation
    tops = s7[s7.metric == "MASE"].copy()
    markets = tops.market.tolist()
    # build matrix model x market of R vs ets from task — use winprob
    models = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "LL01", "svr", "ets"]
    mat = np.full((len(markets), len(models)), np.nan)
    for i, mkt in enumerate(markets):
        row = tops[tops.market == mkt].iloc[0]
        for j, cid in enumerate(models):
            mat[i, j] = row.get(f"winprob_{cid}", np.nan)
    im = axes[0, 0].imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set_yticks(range(len(markets))); axes[0, 0].set_yticklabels([m.split("_")[0] for m in markets], fontsize=7)
    axes[0, 0].set_xticks(range(len(models))); axes[0, 0].set_xticklabels([LABEL[c] for c in models], rotation=30, ha="right", fontsize=7)
    for i, mkt in enumerate(markets):
        row = tops[tops.market == mkt].iloc[0]
        j = models.index(row.bootstrap_winner) if row.bootstrap_winner in models else 0
        mark = "*" if row.status == "Clear Winner" else "~"
        axes[0, 0].text(j, i, mark, ha="center", va="center", color="black", fontsize=8)
    axes[0, 0].set_title("a", loc="left", fontweight="bold"); fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    # b forest by group for headline
    ax = axes[0, 1]
    y = 0
    yticks, ylabels = [], []
    for g in ["Large", "Medium", "Small"]:
        for cid in ["granite_patchtst_fm", "timesfm_3", "chronos_2", "LL01", "svr"]:
            row = s6[(s6.market_group == g) & (s6.config_id == cid)]
            if row.empty: continue
            r = float(row.R_MASE_vs_ETS.iloc[0])
            ax.plot(r, y, "o", color=COLORS.get(LABEL[cid], "#333"))
            yticks.append(y); ylabels.append(f"{g[:1]}-{LABEL[cid]}")
            y += 1
        y += 0.5
    ax.axvline(1, color="grey", ls="--"); ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("R_MASE vs ETS"); ax.set_title("b", loc="left", fontweight="bold"); style_axes(ax)
    # c week trajectory
    ax = axes[1, 0]
    for cid in ["granite_patchtst_fm", "timesfm_3", "chronos_2", "LL01", "ets"]:
        part = s8[(s8.config_id == cid) & (s8.week_block.isin(["Week1","Week2","Week3","Week4"]))].set_index("week_block").loc[["Week1","Week2","Week3","Week4"]]
        ax.plot(["W1","W2","W3","W4"], part.MASE, marker="o", color=COLORS.get(LABEL[cid], "#333"), label=LABEL[cid])
    ax.set_title("c", loc="left", fontweight="bold"); ax.legend(fontsize=6); style_axes(ax); ax.set_ylabel("MASE")
    # d degradation
    ax = axes[1, 1]
    deg = s8[s8.week_block == "SameWeekday_W4_over_W1"]
    deg = deg[deg.config_id.isin(["granite_patchtst_fm", "timesfm_3", "chronos_2", "LL01", "svr", "ets", "seasonal_naive"])]
    ax.barh(deg.model, deg.MASE, color=[COLORS.get(m, "#777") for m in deg.model])
    ax.axvline(1, color="grey", ls="--"); ax.set_xlabel("Week4/Week1 MASE ratio"); ax.set_title("d", loc="left", fontweight="bold"); style_axes(ax)
    fig.tight_layout(); save_fig(fig, "figure8_market_heterogeneity_and_horizon"); plt.close(fig)


def fig9():
    s9 = pd.read_csv(TABLES / "table_S9_high_demand_holiday_underprediction_risk.csv")
    s10 = pd.read_csv(TABLES / "table_S10_prediction_interval_80_coverage_and_wis.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    models = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "ets", "LL01"]
    # a underprediction rates
    ax = axes[0, 0]
    width = 0.35
    for i, rule in enumerate(["exante_official_holiday", "ordinary_non_holiday"]):
        part = s9[(s9.high_demand_rule == rule) & (s9.config_id.isin(models))]
        x = np.arange(len(part))
        ax.bar(x + i * width, part.underprediction_rate, width, label=rule)
    ax.set_xticks(np.arange(len(models))); ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right", fontsize=7)
    # rebuild properly
    ax.clear()
    x = np.arange(len(models))
    hol = [float(s9[(s9.config_id==m)&(s9.high_demand_rule=="exante_official_holiday")].underprediction_rate.iloc[0]) for m in models]
    ord_ = [float(s9[(s9.config_id==m)&(s9.high_demand_rule=="ordinary_non_holiday")].underprediction_rate.iloc[0]) for m in models]
    sev = [float(s9[(s9.config_id==m)&(s9.high_demand_rule=="exante_official_holiday")].severe_under_rate_20pct.iloc[0]) for m in models]
    ax.bar(x - 0.2, hol, 0.2, label="Holiday under", color="#E66101")
    ax.bar(x, ord_, 0.2, label="Ordinary under", color="#1F77B4")
    ax.bar(x + 0.2, sev, 0.2, label="Holiday severe under", color="#7B3294")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right", fontsize=7)
    ax.set_title("a", loc="left", fontweight="bold"); ax.legend(fontsize=6); style_axes(ax)
    # b shortfall
    ax = axes[0, 1]
    people = [float(s9[(s9.config_id==m)&(s9.high_demand_rule=="exante_official_holiday")].cond_shortfall_people.iloc[0]) for m in models]
    pct = [float(s9[(s9.config_id==m)&(s9.high_demand_rule=="exante_official_holiday")].cond_shortfall_pct.iloc[0]) for m in models]
    ax.bar(x - 0.15, people, 0.3, label="People", color="#E66101")
    ax2 = ax.twinx(); ax2.bar(x + 0.15, pct, 0.3, label="Pct", color="#7B3294")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right", fontsize=7)
    ax.set_title("b", loc="left", fontweight="bold"); style_axes(ax)
    # c coverage
    ax = axes[1, 0]
    sub = s10[(s10.split_name == "outer_backtest") & (s10.model.isin(FOUNDATION5 := ["granite_patchtst_fm","timesfm_3","chronos_2","moirai","lag_llama"]))]
    for model in FOUNDATION5:
        part = sub[sub.model == model]
        order = ["overall", "non_holiday", "official_holiday", "Week1", "Week4"]
        vals = [float(part[part.subset == s].coverage80.iloc[0]) if (part.subset == s).any() else np.nan for s in order]
        ax.plot(order, vals, marker="o", label=LABEL[model], color=COLORS.get(LABEL[model], None))
    ax.axhline(0.8, color="grey", ls="--"); ax.set_ylim(0.5, 1.0)
    ax.set_title("c", loc="left", fontweight="bold"); ax.legend(fontsize=6); style_axes(ax)
    ax.tick_params(axis="x", rotation=20)
    # d WIS stacked
    ax = axes[1, 1]
    part = s10[(s10.split_name == "outer_backtest") & (s10.subset == "overall")]
    y = np.arange(len(part))
    ax.barh(y, part.WIS_dispersion, label="Dispersion", color="#9ECAE1")
    ax.barh(y, part.WIS_overprediction_penalty, left=part.WIS_dispersion, label="Over", color="#FDBB84")
    left = part.WIS_dispersion.to_numpy() + part.WIS_overprediction_penalty.to_numpy()
    ax.barh(y, part.WIS_underprediction_penalty, left=left, label="Under", color="#E66101")
    ax.set_yticks(y); ax.set_yticklabels(part.display_name); ax.legend(fontsize=6)
    ax.set_title("d", loc="left", fontweight="bold"); style_axes(ax)
    fig.tight_layout(); save_fig(fig, "figure9_holiday_risk_and_interval_calibration"); plt.close(fig)


def fig10():
    s13 = pd.read_csv(TABLES / "table_S13_rq2_boundary_summary_matrix.csv")
    metrics = ["MASE", "MAPE", "MAE", "RMSE"]
    models = [c for c in HEADLINE if c in set(s13.config_id)]
    dims = sorted(s13.dimension.unique())
    # build grid dims x (models*metrics)
    fig, ax = plt.subplots(figsize=(14, 7))
    cols = []
    data = []
    for metric in metrics:
        for cid in models:
            cols.append(f"{LABEL[cid]}\n{metric}")
    for dim in dims:
        row = []
        for metric in metrics:
            for cid in models:
                part = s13[(s13.dimension == dim) & (s13.config_id == cid) & (s13.metric == metric)]
                row.append(float(part.R_vs_ETS.iloc[0]) if len(part) else np.nan)
        data.append(row)
    data = np.asarray(data)
    vmax = np.nanmax(np.abs(data - 1))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=1 - vmax, vmax=1 + vmax)
    ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims, fontsize=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6, rotation=90)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.02); ax.set_title("RQ2 advantage boundary (R vs ETS)")
    fig.tight_layout(); save_fig(fig, "figure10_rq2_advantage_boundary_heatmap"); plt.close(fig)


def run():
    ensure_dirs()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 9})
    s1 = pd.read_csv(TABLES / "table_S1_regime_comparison.csv")
    fig5(s1); log("fig5")
    fig6(); log("fig6")
    s5 = pd.read_csv(TABLES / "table_S5_history_length_and_denominator_control.csv")
    fig7(s5); log("fig7")
    fig8(); log("fig8")
    fig9(); log("fig9")
    fig10(); log("fig10")


if __name__ == "__main__":
    run()
