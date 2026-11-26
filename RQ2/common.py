"""Shared constants and statistics for RQ2 (section 4.2)."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
OUT = REPO / "RQ2_outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"
RAW = OUT / "raw"
RQ1 = REPO / "RQ1_outputs"

SEED, B, BLOCK_REOPEN, BLOCK_STRESS = 20261116, 4000, 3, 1
E_START, E_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-07")

CORE = [
    ("granite_patchtst_fm", "Granite", "Foundation"),
    ("timesfm_3", "TimesFM 3.0", "Foundation"),
    ("chronos_2", "Chronos-2", "Foundation"),
    ("moirai", "Moirai 2.0", "Foundation"),
    ("lag_llama", "Lag-Llama", "Foundation"),
    ("ets", "ETS", "Statistical"),
    ("sarima", "SARIMA", "Statistical"),
    ("seasonal_naive", "Seasonal Naive", "Statistical"),
    ("svr", "SVR", "Machine learning"),
    ("lightgbm", "LightGBM (Local)", "Machine learning"),
    ("GL03", "LightGBM (Global)", "Machine learning"),
    ("LL01", "LightGBM+Calendar (Local)", "Machine learning"),
    ("transformer", "Small Transformer", "Deep learning"),
    ("NL01", "Global NLinear", "Deep learning"),
]
CORE_IDS = [c[0] for c in CORE]
LABEL = {c[0]: c[1] for c in CORE}
FAMILY = {c[0]: c[2] for c in CORE}
FOUNDATION5 = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama"]
HEADLINE = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "LL01", "svr"]

COLORS = {
    "Granite": "#E66101", "TimesFM 3.0": "#7B3294", "Chronos-2": "#1F77B4",
    "Moirai 2.0": "#17BECF", "Lag-Llama": "#6BAED6", "ETS": "#636363",
    "LightGBM+Calendar (Local)": "#DF65B0", "SVR": "#8C564B",
    "Global NLinear": "#2CA02C", "Seasonal Naive": "#D62728",
    "SARIMA": "#9E9E9E", "LightGBM (Local)": "#F4A582",
    "LightGBM (Global)": "#B2ABD2", "Small Transformer": "#A6D854",
}

FROZEN_BASELINE = {
    "seasonal_naive": "SN01", "ets": "ETS03", "sarima": "SAR03",
    "svr": "SVR02", "lightgbm": "LGB01", "transformer": "TR03",
}

# Volume ranking from shared normalized_daily (Macau, all years).
LARGE = ["CN_Chinese_mainland", "HK_Hong_Kong_SAR"]
MEDIUM = [
    "TW_Taiwan_region", "KR_Republic_of_Korea", "PH_Philippines", "JP_Japan",
    "MY_Malaysia", "ID_Indonesia", "US_United_States_of_America", "TH_Thailand",
]
SMALL = [
    "SG_Singapore", "IN_India", "AU_Australia", "GB_United_Kingdom", "CA_Canada",
    "RU_Russian_Federation", "DE_Germany", "FR_France", "IT_Italy", "BR_Brazil",
    "PT_Portugal", "NZ_New_Zealand", "NL_Netherlands", "ES_Spain", "CH_Switzerland",
    "ZA_South_Africa", "VN_Viet_Nam",
]
MARKET_GROUP = {m: "Large" for m in LARGE}
MARKET_GROUP.update({m: "Medium" for m in MEDIUM})
MARKET_GROUP.update({m: "Small" for m in SMALL})


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, LOGS, RAW):
        path.mkdir(parents=True, exist_ok=True)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def dm_hln(loss_a, loss_b, h: int = 1, lag: int | None = None):
    """HLN-DM; d = model - benchmark; negative => model better. lag overrides Bartlett HAC order."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    t = len(d)
    if t < 3 or np.allclose(d, d[0]):
        return np.nan, np.nan
    mean = d.mean()
    L = h if lag is None else lag
    gamma = [np.sum((d[k:] - mean) * (d[: t - k] - mean)) / t for k in range(L + 1)]
    var = gamma[0] + 2 * sum(gamma[1:])
    var = var / t
    if var <= 0:
        return np.nan, np.nan
    # Harvey–Leybourne–Newbold small-sample factor uses forecast horizon h (here 1).
    h_star = 1
    factor = np.sqrt((t + 1 - 2 * h_star + h_star * (h_star - 1) / t) / t)
    stat = mean / np.sqrt(var) * factor
    return float(stat), float(2 * stats.t.sf(abs(stat), df=t - 1))


def block_indices(t: int, block: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    if block <= 0:
        raise ValueError("block must be positive")
    if block >= t:
        # independent resampling of whole series length with replacement of units
        return rng.integers(0, t, size=(resamples, t))
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, t - block + 1, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)
    return idx[:, :t]


def ci_from_idx(values, idx) -> tuple[float, float]:
    means = np.asarray(values, float)[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def origin_means(task: pd.DataFrame, metric: str) -> pd.Series:
    """Equal-market mean at each origin, sorted by origin_date."""
    part = task.copy()
    part["origin_date"] = pd.to_datetime(part["origin_date"])
    return part.groupby("origin_date", sort=True)[metric].mean()


def market_then_equal(task: pd.DataFrame, metric: str) -> float:
    return float(task.groupby("market")[metric].mean().mean())


def ratio_of_means(task_a: pd.DataFrame, task_b: pd.DataFrame, metric: str) -> float:
    return market_then_equal(task_a, metric) / market_then_equal(task_b, metric)


def log(msg: str) -> None:
    ensure_dirs()
    line = msg if msg.endswith("\n") else msg + "\n"
    print(msg, flush=True)
    with (LOGS / "rq2_run.log").open("a", encoding="utf-8") as stream:
        stream.write(line)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
