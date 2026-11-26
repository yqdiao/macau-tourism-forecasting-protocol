"""RQ3 shared constants — aligned with RQ1/RQ2 sealed protocol."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
OUT = REPO / "RQ3_outputs"
TABLES, FIGURES, LOGS, CACHE = OUT / "tables", OUT / "figures", OUT / "logs", OUT / "cache"
RQ1, RQ2 = REPO / "RQ1_outputs", REPO / "RQ2_outputs"

SEED, B, BLOCK_REOPEN, BLOCK_STRESS = 20261116, 4000, 3, 1
LAMBDA0, ALPHA0, EPS, DELTA0, ETA0, K0 = 0.85, 0.20, 1e-4, 0.02, 0.30, 4
DEFAULT_SUBSET = "C2"

POOL = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "NL01", "GL03"]
FM = ["granite_patchtst_fm", "timesfm_3", "chronos_2"]
LW = ["NL01", "GL03"]
LABEL = {
    "granite_patchtst_fm": "Granite",
    "timesfm_3": "TimesFM 3.0",
    "chronos_2": "Chronos-2",
    "NL01": "Global NLinear",
    "GL03": "LightGBM (Global)",
    "LL01": "LightGBM+Calendar (Local)",
    "ets": "ETS",
}
DEVICE = {
    "granite_patchtst_fm": "gpu",
    "timesfm_3": "gpu",
    "chronos_2": "gpu",
    "NL01": "cpu",
    "GL03": "cpu",
    "ets": "cpu",
    "LL01": "cpu",
}

# Combinations
COMBOS = {
    "C1": ["granite_patchtst_fm", "timesfm_3", "NL01"],
    "C2": ["granite_patchtst_fm", "timesfm_3", "GL03"],
    "C3": ["granite_patchtst_fm", "chronos_2", "NL01"],
    "C4": ["granite_patchtst_fm", "chronos_2", "GL03"],
    "C5": ["timesfm_3", "chronos_2", "NL01"],
    "C6": ["timesfm_3", "chronos_2", "GL03"],
    "C7": ["granite_patchtst_fm", "NL01", "GL03"],
    "C8": ["timesfm_3", "NL01", "GL03"],
    "C9": ["chronos_2", "NL01", "GL03"],
    "B_3FM": ["granite_patchtst_fm", "timesfm_3", "chronos_2"],
    "B_5Pool": POOL.copy(),
}
STRUCTURE = {
    **{f"C{i}": "2FM+1LW" for i in range(1, 7)},
    **{f"C{i}": "1FM+2LW" for i in range(7, 10)},
    "B_3FM": "3FM",
    "B_5Pool": "5Pool",
}

LARGE = ["CN_Chinese_mainland", "HK_Hong_Kong_SAR"]
MEDIUM = [
    "TW_Taiwan_region", "KR_Republic_of_Korea", "PH_Philippines", "JP_Japan",
    "MY_Malaysia", "ID_Indonesia", "US_United_States_of_America", "TH_Thailand",
]
MARKET_GROUP = {m: "Large" for m in LARGE}
MARKET_GROUP.update({m: "Medium" for m in MEDIUM})

# Price scenarios ρ (GPU$/CPU$)
RHO = {"S1": 13.25, "S2": 8.25, "S3": 27.25}
# LLM judgment cost placeholder (GPU-sec per call) when real timing unavailable
LLM_SEC_PER_JUDGMENT = 0.85

COLORS = {
    "Granite": "#E66101", "TimesFM 3.0": "#7B3294", "Chronos-2": "#1F77B4",
    "Global NLinear": "#2CA02C", "LightGBM (Global)": "#B2ABD2",
    "ETS": "#636363", "3FM-Equal": "#8C564B", "G1_Stat_Stat": "#222222",
}


def ensure_dirs():
    for p in (TABLES, FIGURES, LOGS, CACHE):
        p.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ensure_dirs()
    print(msg, flush=True)
    with (LOGS / "rq3_run.log").open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


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


def dm_hln(loss_a, loss_b, lag: int = 2):
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    t = len(d)
    if t < 3 or np.allclose(d, d[0]):
        return np.nan, np.nan
    mean = d.mean()
    gamma = [np.sum((d[k:] - mean) * (d[: t - k] - mean)) / t for k in range(lag + 1)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        return np.nan, np.nan
    dm = mean / np.sqrt(var / t)
    # HLN small-sample correction
    hln = dm * np.sqrt((t + 1 - 2 * 1 + 1 / t) / t)
    p = 2 * stats.t.sf(abs(hln), df=max(t - 1, 1))
    return float(hln), float(p)


def block_indices(t: int, block: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    if block >= t:
        return rng.integers(0, t, size=(resamples, t))
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, t - block + 1, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)
    return idx[:, :t]


def market_then_equal(task: pd.DataFrame, metric: str) -> float:
    return float(task.groupby("market")[metric].mean().mean())


def origin_means(task: pd.DataFrame, metric: str) -> pd.Series:
    part = task.copy()
    part["origin_date"] = pd.to_datetime(part["origin_date"])
    return part.groupby("origin_date", sort=True)[metric].mean()


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")


def combo_cost_rank(combo_id: str) -> float:
    """Lower = cheaper under S1 conditional cost (GPU weighted by ρ)."""
    t4 = pd.read_csv(RQ1 / "tables/table4_timing_and_price_ratio.csv").set_index("model")
    rho = RHO["S1"]
    members = COMBOS[combo_id]
    total = 0.0
    for m in members:
        sec = float(t4.loc[m, "seconds_per_vector"])
        if DEVICE[m] == "gpu":
            total += sec * rho
        else:
            total += sec
    return total
