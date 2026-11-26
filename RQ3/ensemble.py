"""Stat-Select / Stat-Weight and LLM-proxy judges (no look-ahead)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import (
    ALPHA0, CACHE, COMBOS, DEFAULT_SUBSET, DELTA0, EPS, ETA0, LAMBDA0, POOL,
    STRUCTURE, combo_cost_rank, log,
)


def equal_weights(n: int = 3) -> np.ndarray:
    return np.ones(n) / n


def stat_weights(errors: np.ndarray, alpha: float = ALPHA0, eps: float = EPS) -> np.ndarray:
    """Inverse-error weights with shrink-to-equal."""
    e = np.asarray(errors, float)
    v = 1.0 / (e + eps)
    v = v / v.sum()
    return (1 - alpha) * v + alpha / len(e)


def discounted_mean(values: np.ndarray, ages_weeks: np.ndarray, lam: float) -> float:
    if len(values) == 0:
        return np.nan
    w = lam ** ages_weeks
    return float(np.sum(w * values) / np.sum(w))


def ages_in_weeks(hist_origins: pd.DatetimeIndex, cur_origin: pd.Timestamp) -> np.ndarray:
    return ((pd.Timestamp(cur_origin) - pd.to_datetime(hist_origins)).days.to_numpy() / 7.0)


def select_combo(
    combo_errors: dict[str, float],
    delta: float = DELTA0,
    candidates: Iterable[str] | None = None,
) -> str:
    """Accuracy-first with cost tie-break inside (1+delta) band."""
    candidates = list(candidates) if candidates is not None else [f"C{i}" for i in range(1, 10)]
    errs = {c: combo_errors[c] for c in candidates if np.isfinite(combo_errors.get(c, np.nan))}
    if not errs:
        return DEFAULT_SUBSET
    emin = min(errs.values())
    band = [c for c, e in errs.items() if e <= (1 + delta) * emin]
    band.sort(key=lambda c: (combo_cost_rank(c), errs[c], c))
    return band[0]


def combine_predictions(wide: pd.DataFrame, members: list[str], weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, float)
    w = w / w.sum()
    mat = wide[members].to_numpy(float)
    return mat @ w


# ---------- LLM proxy (same information set; distinct from Stat) ----------
REASON_CODES = [
    "recent_error_leader",
    "forecast_dispersion_hedge",
    "cross_family_complement",
    "cost_tiebreak_light",
    "holiday_volatility",
    "stat_anchor",
]


def llm_proxy_select(
    combo_errors: dict[str, float],
    single_errors: dict[str, float],
    dispersion: float,
    holiday_days: int,
    delta: float = DELTA0,
) -> tuple[str, dict]:
    """
    Feature-rule LLM proxy: soft-prefer cross-family diversity and cost,
    with accuracy band similar to Stat but tilted toward 1FM+2LW when dispersion high
    or holidays present. NOT identical to Stat-Select.
    """
    candidates = [f"C{i}" for i in range(1, 10)]
    errs = {c: combo_errors[c] for c in candidates if np.isfinite(combo_errors.get(c, np.nan))}
    if not errs:
        payload = {
            "selected_models": COMBOS[DEFAULT_SUBSET],
            "weights": [1 / 3] * 3,
            "reason_codes": ["stat_anchor"],
            "evidence_ids": ["cold_start"],
            "combo_id": DEFAULT_SUBSET,
        }
        return DEFAULT_SUBSET, payload

    emin = min(errs.values())
    # Wider effective band when dispersion/holidays high (LLM explores more)
    boost = 0.0
    if dispersion > 0.15:
        boost += 0.01
    if holiday_days >= 2:
        boost += 0.01
    band_delta = delta + boost
    band = [c for c, e in errs.items() if e <= (1 + band_delta) * emin]

    def score(c: str) -> tuple:
        # lower better: prefer 1FM+2LW slightly, then cost, then error
        struct_pen = 0.0 if STRUCTURE[c] == "1FM+2LW" else 0.15
        # prefer including recent single-error leader
        leader = min(single_errors, key=single_errors.get)
        has_leader = 0.0 if leader in COMBOS[c] else 0.25
        return (struct_pen + has_leader, combo_cost_rank(c), errs[c], c)

    band.sort(key=score)
    chosen = band[0]
    reasons = ["recent_error_leader"]
    if STRUCTURE[chosen] != STRUCTURE[min(errs, key=errs.get)]:
        reasons.append("cross_family_complement")
    if holiday_days >= 2:
        reasons.append("holiday_volatility")
    if dispersion > 0.15:
        reasons.append("forecast_dispersion_hedge")
    if STRUCTURE[chosen] == "1FM+2LW":
        reasons.append("cost_tiebreak_light")
    payload = {
        "selected_models": COMBOS[chosen],
        "weights": None,  # filled by weight fn
        "reason_codes": reasons,
        "evidence_ids": ["hist_mae_rank", "holiday_flag" if holiday_days else "hist_rmse_rank"],
        "combo_id": chosen,
    }
    return chosen, payload


def llm_proxy_weights(
    members: list[str],
    errors: np.ndarray,
    stat_w: np.ndarray,
    eta: float,
    dispersion: float,
) -> tuple[np.ndarray, dict]:
    """LLM-like weights: sharper inverse-error + slight diversity tilt, fused with stat."""
    e = np.asarray(errors, float)
    # sharper than stat (temperature)
    temp = 0.65 if dispersion < 0.1 else 0.9
    logits = -np.log(e + EPS) / temp
    logits = logits - logits.max()
    raw = np.exp(logits)
    raw = raw / raw.sum()
    # diversity: nudge away from max weight dictatorship
    raw = 0.85 * raw + 0.15 / len(raw)
    raw = raw / raw.sum()
    final = (1 - eta) * np.asarray(stat_w, float) + eta * raw
    final = np.clip(final, 0, None)
    final = final / final.sum()
    payload = {
        "selected_models": members,
        "weights": [float(x) for x in final],
        "reason_codes": ["recent_error_leader", "stat_anchor"],
        "evidence_ids": ["hist_mae_rank"],
        "w_llm_raw": [float(x) for x in raw],
        "eta": eta,
    }
    return final, payload


def validate_llm_payload(payload: dict, pool: list[str] = POOL) -> tuple[bool, str]:
    try:
        models = payload.get("selected_models")
        weights = payload.get("weights")
        if not isinstance(models, list) or len(models) != 3:
            return False, "bad_selected_models"
        if any(m not in pool for m in models):
            return False, "model_not_in_pool"
        if weights is None:
            return False, "missing_weights"
        w = np.asarray(weights, float)
        if w.shape != (3,) or np.any(~np.isfinite(w)) or np.any(w < 0):
            return False, "bad_weights"
        if abs(w.sum() - 1.0) > 1e-6:
            return False, "weights_not_sum_to_1"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"exception:{exc}"


class JudgmentCache:
    def __init__(self, path: Path | None = None):
        self.path = path or (CACHE / "llm_judgments.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self._mem[obj["key"]] = obj

    def get(self, key: str):
        return self._mem.get(key)

    def put(self, key: str, obj: dict) -> None:
        obj = dict(obj)
        obj["key"] = key
        self._mem[key] = obj
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
