"""Causal dynamic replay for fixed and adaptive ensembles."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from common import (
    ALPHA0, COMBOS, DEFAULT_SUBSET, DELTA0, EPS, ETA0, K0, LAMBDA0, POOL, STRUCTURE, log,
)
from data import completed_history_mask, origin_schedule, score_task_from_days
from ensemble import (
    JudgmentCache, discounted_mean, ages_in_weeks, combine_predictions, equal_weights,
    llm_proxy_select, llm_proxy_weights, select_combo, stat_weights, validate_llm_payload,
)


def _task_mae_lookup(task: pd.DataFrame) -> dict:
    """(config, market, origin) -> mae"""
    d = {}
    for r in task.itertuples(index=False):
        d[(r.config_id, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)
    return d


def precompute_equal_combo_tasks(wide: pd.DataFrame, combo_ids: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for cid in combo_ids:
        members = COMBOS[cid]
        w = equal_weights(len(members))
        df = wide.copy()
        df["yhat"] = combine_predictions(df, members, w)
        out[cid] = score_task_from_days(df, "yhat")
        out[cid]["config_id"] = cid
    return out


def error_corr_and_dispersion(wide: pd.DataFrame, members: list[str]) -> tuple[float, float]:
    """Mean pairwise error correlation + forecast CV dispersion on same days."""
    y = wide.actual.to_numpy(float)
    errs = np.column_stack([wide[m].to_numpy(float) - y for m in members])
    # pairwise corr of errors
    corrs = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = errs[:, i], errs[:, j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            corrs.append(np.corrcoef(a, b)[0, 1])
    preds = wide[members].to_numpy(float)
    row_std = preds.std(axis=1)
    row_mean = np.abs(preds.mean(axis=1)) + 1e-6
    disp = float(np.mean(row_std / row_mean))
    return float(np.nanmean(corrs)) if corrs else np.nan, disp


def replay_fixed_equal(wide: pd.DataFrame, combo_id: str) -> pd.DataFrame:
    members = COMBOS[combo_id]
    df = wide.copy()
    df["yhat"] = combine_predictions(df, members, equal_weights(len(members)))
    task = score_task_from_days(df, "yhat")
    task["config_id"] = combo_id
    task["scheme"] = f"{combo_id}_Equal"
    return task


def replay_fixed_stat_weight(
    wide: pd.DataFrame,
    single_task: pd.DataFrame,
    combo_id: str,
    split: str,
    lam: float = LAMBDA0,
    alpha: float = ALPHA0,
    driver: str = "mae",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed subset, dynamic Stat-Weight with completed-history rule."""
    members = COMBOS[combo_id]
    # build schedule from wide
    sched = (
        wide.groupby(["fold_id", "origin_date"], sort=False)
        .agg(target_end=("target_date", "max"))
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    sched["origin_idx"] = np.arange(len(sched))
    origin_to_idx = {pd.Timestamp(o): i for i, o in enumerate(sched.origin_date)}

    # single model task metric lookup
    st = single_task[single_task.config_id.isin(members)].copy()
    metric_col = "mae" if driver == "mae" else "mase"
    lookup = {}
    for r in st.itertuples(index=False):
        lookup[(r.config_id, r.market, pd.Timestamp(r.origin_date))] = float(getattr(r, metric_col))

    weight_rows = []
    parts = []
    for market, mwide in wide.groupby("market", sort=False):
        for _, row in sched.iterrows():
            od = pd.Timestamp(row.origin_date)
            idx = int(row.origin_idx)
            sub = mwide[mwide.origin_date == od]
            if sub.empty:
                continue
            # completed history
            hist = sched[(sched.origin_idx < idx) & (sched.target_end <= od)]
            if len(hist) == 0:
                w = equal_weights(len(members))
                cold = True
            else:
                errs = []
                for m in members:
                    vals, ages = [], []
                    for _, h in hist.iterrows():
                        key = (m, market, pd.Timestamp(h.origin_date))
                        if key in lookup:
                            vals.append(lookup[key])
                            ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                    errs.append(discounted_mean(np.array(vals), np.array(ages), lam) if vals else np.nan)
                if not np.all(np.isfinite(errs)):
                    w = equal_weights(len(members))
                    cold = True
                else:
                    w = stat_weights(np.array(errs), alpha=alpha)
                    cold = False
            yhat = combine_predictions(sub, members, w)
            chunk = sub[["market", "fold_id", "origin_date", "target_date", "horizon_day", "task_id", "actual", "q_all"]].copy()
            chunk["yhat"] = yhat
            parts.append(chunk)
            weight_rows.append({
                "scheme": f"{combo_id}_StatWeight",
                "combo_id": combo_id,
                "market": market,
                "origin_date": od,
                "cold_start": cold,
                **{f"w_{m}": float(ww) for m, ww in zip(members, w)},
                "members": "|".join(members),
            })
    day = pd.concat(parts, ignore_index=True)
    task = score_task_from_days(day, "yhat")
    task["config_id"] = f"{combo_id}_StatWeight"
    task["scheme"] = f"{combo_id}_StatWeight"
    return task, pd.DataFrame(weight_rows)


def _holiday_count(sub: pd.DataFrame, holiday_dates: set) -> int:
    dates = pd.to_datetime(sub.target_date).dt.date
    return int(sum(d in holiday_dates for d in dates.unique()))


def replay_dynamic(
    wide: pd.DataFrame,
    single_task: pd.DataFrame,
    equal_combo_tasks: dict[str, pd.DataFrame],
    split: str,
    select_mode: str,   # 'stat' | 'llm'
    weight_mode: str,   # 'stat' | 'llm'
    eta: float = ETA0,
    lam: float = LAMBDA0,
    alpha: float = ALPHA0,
    delta: float = DELTA0,
    driver: str = "mae",
    holiday_dates: set | None = None,
    scheme_name: str = "G1",
    mode: str = "mode1",  # mode1|mode2 informational for logging
    refresh_k: int = K0,
    cache: JudgmentCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full causal replay for one 2x2 cell."""
    holiday_dates = holiday_dates or set()
    cache = cache or JudgmentCache()
    sched = (
        wide.groupby(["fold_id", "origin_date"], sort=False)
        .agg(target_end=("target_date", "max"))
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    sched["origin_idx"] = np.arange(len(sched))

    metric_col = "mae" if driver == "mae" else "mase"
    single_lookup = {}
    for r in single_task.itertuples(index=False):
        single_lookup[(r.config_id, r.market, pd.Timestamp(r.origin_date))] = float(getattr(r, metric_col))

    combo_lookup = {}
    for cid, tdf in equal_combo_tasks.items():
        for r in tdf.itertuples(index=False):
            combo_lookup[(cid, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)

    fallback_counts = defaultdict(int)
    decision_rows = []
    parts = []
    models_run_counter = defaultdict(float)  # for mode2 cost accounting later

    for market, mwide in wide.groupby("market", sort=False):
        for _, row in sched.iterrows():
            od = pd.Timestamp(row.origin_date)
            idx = int(row.origin_idx)
            sub = mwide[mwide.origin_date == od]
            if sub.empty:
                continue
            hist = sched[(sched.origin_idx < idx) & (sched.target_end <= od)]
            cold = len(hist) == 0

            # historical discounted errors for combos & singles
            combo_errors = {}
            for cid in [f"C{i}" for i in range(1, 10)]:
                vals, ages = [], []
                for _, h in hist.iterrows():
                    key = (cid, market, pd.Timestamp(h.origin_date))
                    if key in combo_lookup:
                        vals.append(combo_lookup[key])
                        ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                combo_errors[cid] = discounted_mean(np.array(vals), np.array(ages), lam) if vals else np.nan

            single_errors = {}
            for m in POOL:
                vals, ages = [], []
                for _, h in hist.iterrows():
                    key = (m, market, pd.Timestamp(h.origin_date))
                    if key in single_lookup:
                        vals.append(single_lookup[key])
                        ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                single_errors[m] = discounted_mean(np.array(vals), np.array(ages), lam) if vals else np.nan

            # dispersion of pool forecasts at this origin (Mode1 info; also used by proxy)
            pool_preds = sub[POOL].to_numpy(float)
            dispersion = float(pool_preds.std(axis=1).mean() / (np.abs(pool_preds.mean()) + 1e-6)) if len(sub) else 0.0
            hol_n = _holiday_count(sub, holiday_dates)

            # --- select ---
            used_fallback = False
            fallback_reason = ""
            if cold:
                combo_id = DEFAULT_SUBSET
                select_backend = "cold_default_C2"
            elif select_mode == "stat":
                combo_id = select_combo(combo_errors, delta=delta)
                select_backend = "stat_select"
            else:
                combo_id, sel_payload = llm_proxy_select(combo_errors, single_errors, dispersion, hol_n, delta=delta)
                select_backend = "llm_proxy_select"
                # cache judgment
                key = f"{scheme_name}|{split}|{market}|{od.date()}|select"
                cache.put(key, {"backend": select_backend, "payload": {k: v for k, v in sel_payload.items() if k != "weights"},
                                "split": split, "market": market, "origin_date": str(od.date())})

            members = COMBOS[combo_id]

            # --- weights ---
            if cold:
                w = equal_weights(3)
                weight_backend = "cold_equal"
            else:
                errs = []
                for m in members:
                    errs.append(single_errors.get(m, np.nan))
                if not np.all(np.isfinite(errs)):
                    w = equal_weights(3)
                    weight_backend = "fallback_equal_nan_hist"
                    used_fallback = True
                    fallback_reason = "nan_hist_errors"
                    fallback_counts[fallback_reason] += 1
                else:
                    w_stat = stat_weights(np.array(errs), alpha=alpha)
                    if weight_mode == "stat":
                        w = w_stat
                        weight_backend = "stat_weight"
                    else:
                        w, w_payload = llm_proxy_weights(members, np.array(errs), w_stat, eta=eta, dispersion=dispersion)
                        ok, reason = validate_llm_payload({**w_payload, "selected_models": members, "weights": list(w)})
                        if not ok:
                            w = w_stat
                            weight_backend = f"fallback_stat:{reason}"
                            used_fallback = True
                            fallback_reason = reason
                            fallback_counts[reason] += 1
                        else:
                            weight_backend = "llm_proxy_weight"
                            key = f"{scheme_name}|{split}|{market}|{od.date()}|weight"
                            cache.put(key, {"backend": weight_backend, "payload": w_payload,
                                            "split": split, "market": market, "origin_date": str(od.date()), "eta": eta})

            # Mode2 refresh bookkeeping: which models are "run"
            refresh = (idx % refresh_k == 0)
            if mode == "mode1":
                run_models = list(POOL)
            else:
                run_models = list(members) if not refresh else list(POOL)
            for m in run_models:
                models_run_counter[m] += 1

            yhat = combine_predictions(sub, members, w)
            chunk = sub[["market", "fold_id", "origin_date", "target_date", "horizon_day", "task_id", "actual", "q_all"]].copy()
            chunk["yhat"] = yhat
            parts.append(chunk)
            decision_rows.append({
                "scheme": scheme_name,
                "split_name": split,
                "market": market,
                "origin_date": od,
                "origin_idx": idx,
                "combo_id": combo_id,
                "structure": STRUCTURE[combo_id],
                "select_backend": select_backend,
                "weight_backend": weight_backend,
                "cold_start": cold,
                "fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "dispersion": dispersion,
                "holiday_days": hol_n,
                "mode": mode,
                "refresh": refresh,
                **{f"w_{m}": float(ww) for m, ww in zip(members, w)},
                "member_0": members[0], "member_1": members[1], "member_2": members[2],
                "w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]),
            })

    day = pd.concat(parts, ignore_index=True)
    task = score_task_from_days(day, "yhat")
    task["config_id"] = scheme_name
    task["scheme"] = scheme_name
    meta = {
        "fallback_counts": dict(fallback_counts),
        "models_run_counter": dict(models_run_counter),
        "n_decisions": len(decision_rows),
    }
    return task, pd.DataFrame(decision_rows), meta
