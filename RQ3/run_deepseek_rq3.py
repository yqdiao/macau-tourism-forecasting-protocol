#!/usr/bin/env python3
"""Collect DeepSeek judgments and recompute G2–G4 + figures for RQ3."""
from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# clear proxies that break HTTPS
for k in list(os.environ):
    if "proxy" in k.lower():
        os.environ.pop(k, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ALPHA0, CACHE, COMBOS, DELTA0, ETA0, FIGURES, K0, LABEL, LAMBDA0, LOGS, OUT, POOL,
    STRUCTURE, TABLES, ensure_dirs, log, save_fig, style_axes,
)
from data import build_prediction_cube, load_day_panel, load_holiday_flags, load_task_panel
from ensemble import (
    JudgmentCache, discounted_mean, equal_weights, select_combo, stat_weights, combine_predictions,
)
from llm_deepseek import (
    build_select_weight_prompt, build_weight_only_prompt, chat_json, load_rq3_env,
    validate_select_payload, validate_weight_payload,
)
from metrics import bootstrap_levels, pairwise_dm, ratio_ci_vs, summarize_task
from replay import precompute_equal_combo_tasks, replay_dynamic, replay_fixed_equal, replay_fixed_stat_weight
from data import score_task_from_days


SPLITS = ["outer_backtest", "pandemic_stress"]
SPLIT_LABEL = {"outer_backtest": "reopening", "pandemic_stress": "pandemic_stress"}
JUDGE_PATH = CACHE / "llm_judgments_deepseek.jsonl"
MAX_WORKERS = int(os.environ.get("RQ3_LLM_WORKERS", "6"))


def demand_features(days: pd.DataFrame, split: str, market: str, origin: pd.Timestamp) -> dict:
    """Past 28 calendar days before origin from actuals of sealed panel (ets rows)."""
    d = days[(days.split_name == split) & (days.config_id == "ets") & (days.market == market)].copy()
    # use actuals with target_date in (origin-28, origin]
    d["target_date"] = pd.to_datetime(d["target_date"])
    lo = origin - pd.Timedelta(days=28)
    hist = d[(d.target_date > lo) & (d.target_date <= origin)]
    # may be sparse; fallback to last 28 rows by target for this market across tasks ending before origin
    if len(hist) < 7:
        hist = d[d.target_date <= origin].sort_values("target_date").tail(28)
    y = hist.actual.to_numpy(float) if len(hist) else np.array([])
    if len(y) == 0:
        return {"demand_mean": None, "demand_std": None, "zero_share": None, "peak_ratio": None}
    return {
        "demand_mean": float(np.mean(y)),
        "demand_std": float(np.std(y)),
        "zero_share": float(np.mean(y <= 0)),
        "peak_ratio": float(np.max(y) / (np.mean(y) + 1e-6)),
    }


def build_feature_jobs(task_all, days, holiday_dates):
    jobs = []
    for split in SPLITS:
        wide = build_prediction_cube(days, split, POOL)
        single = task_all[task_all.split_name == split]
        eq = precompute_equal_combo_tasks(wide, [f"C{i}" for i in range(1, 10)])
        sched = (
            wide.groupby(["fold_id", "origin_date"], sort=False)
            .agg(target_end=("target_date", "max"))
            .reset_index()
            .sort_values("origin_date")
            .reset_index(drop=True)
        )
        sched["origin_idx"] = np.arange(len(sched))

        single_lookup, combo_lookup = {}, {}
        for r in single.itertuples(index=False):
            single_lookup[(r.config_id, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)
        for cid, tdf in eq.items():
            for r in tdf.itertuples(index=False):
                combo_lookup[(cid, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)

        for market, mwide in wide.groupby("market", sort=False):
            for _, row in sched.iterrows():
                od = pd.Timestamp(row.origin_date)
                idx = int(row.origin_idx)
                sub = mwide[mwide.origin_date == od]
                if sub.empty:
                    continue
                hist = sched[(sched.origin_idx < idx) & (sched.target_end <= od)]
                cold = len(hist) == 0
                if cold:
                    continue  # no LLM on cold start

                combo_errors, single_errors = {}, {}
                for cid in [f"C{i}" for i in range(1, 10)]:
                    vals, ages = [], []
                    for _, h in hist.iterrows():
                        key = (cid, market, pd.Timestamp(h.origin_date))
                        if key in combo_lookup:
                            vals.append(combo_lookup[key])
                            ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                    combo_errors[cid] = discounted_mean(np.array(vals), np.array(ages), LAMBDA0) if vals else np.nan
                for m in POOL:
                    vals, ages = [], []
                    for _, h in hist.iterrows():
                        key = (m, market, pd.Timestamp(h.origin_date))
                        if key in single_lookup:
                            vals.append(single_lookup[key])
                            ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                    single_errors[m] = discounted_mean(np.array(vals), np.array(ages), LAMBDA0) if vals else np.nan

                pool_preds = sub[POOL].to_numpy(float)
                dispersion = float(pool_preds.std(axis=1).mean() / (np.abs(pool_preds.mean()) + 1e-6))
                dates = pd.to_datetime(sub.target_date).dt.date
                hol_n = int(sum(d in holiday_dates for d in dates.unique()))
                stat_combo = select_combo(combo_errors, delta=DELTA0)
                members = COMBOS[stat_combo]
                errs = np.array([single_errors[m] for m in members], float)
                if np.all(np.isfinite(errs)):
                    sw = stat_weights(errs, alpha=ALPHA0)
                else:
                    sw = equal_weights(3)
                feat = {
                    "split_name": split,
                    "market": market,
                    "origin_date": str(od.date()),
                    "holiday_days": hol_n,
                    "dispersion": dispersion,
                    "single_mae": {k: (None if not np.isfinite(v) else float(v)) for k, v in single_errors.items()},
                    "combo_mae": {k: (None if not np.isfinite(v) else float(v)) for k, v in combo_errors.items()},
                    "stat_combo": stat_combo,
                    "stat_members": members,
                    "stat_weights": [float(x) for x in sw],
                    "stat_weights_for_members": [float(x) for x in sw],
                    **demand_features(days, split, market, od),
                }
                jobs.append(feat)
    return jobs


def collect_judgments(jobs, env, resume=True):
    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and JUDGE_PATH.exists():
        with JUDGE_PATH.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                done.add(obj["key"])
    log(f"DeepSeek jobs={len(jobs)} already_cached_keys={len(done)}")

    def one(feat, kind: str):
        if kind == "select_weight":
            key = f"deepseek|select_weight|{feat['split_name']}|{feat['market']}|{feat['origin_date']}"
            if key in done:
                return None
            prompt = build_select_weight_prompt(feat)
            raw, meta = chat_json(env, prompt)
            ok, reason, norm = validate_select_payload(raw)
            rec = {
                "key": key, "kind": kind, "backend": "deepseek_api",
                "model": env["DEEPSEEK_MODEL"], "ok": ok, "reason": reason,
                "feat": {k: feat[k] for k in ["split_name", "market", "origin_date", "stat_combo", "holiday_days"]},
                "payload": norm if ok else raw, "meta": {k: meta[k] for k in meta if k != "raw_content"},
                "raw_content": meta.get("raw_content", "")[:2000],
            }
            return rec
        if kind == "weight_only":
            members = feat["stat_members"]
            key = (
                f"deepseek|weight_only|{feat['split_name']}|{feat['market']}|{feat['origin_date']}|"
                + "|".join(members)
            )
            if key in done:
                return None
            prompt = build_weight_only_prompt(feat, members)
            raw, meta = chat_json(env, prompt)
            ok, reason, norm = validate_weight_payload(raw, members)
            rec = {
                "key": key, "kind": kind, "backend": "deepseek_api",
                "model": env["DEEPSEEK_MODEL"], "ok": ok, "reason": reason,
                "feat": {k: feat[k] for k in ["split_name", "market", "origin_date", "stat_combo", "holiday_days"]},
                "members": members,
                "payload": norm if ok else raw, "meta": {k: meta[k] for k in meta if k != "raw_content"},
                "raw_content": meta.get("raw_content", "")[:2000],
            }
            return rec
        return None

    pending = []
    for feat in jobs:
        for kind in ("select_weight", "weight_only"):
            if kind == "select_weight":
                key = f"deepseek|select_weight|{feat['split_name']}|{feat['market']}|{feat['origin_date']}"
            else:
                key = (
                    f"deepseek|weight_only|{feat['split_name']}|{feat['market']}|{feat['origin_date']}|"
                    + "|".join(feat["stat_members"])
                )
            if key not in done:
                pending.append((feat, kind))

    log(f"pending API calls={len(pending)} workers={MAX_WORKERS}")
    n_ok = n_fail = 0
    with JUDGE_PATH.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(one, feat, kind): (feat, kind) for feat, kind in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
                if rec is None:
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if rec["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as exc:  # noqa: BLE001
                feat, kind = futs[fut]
                n_fail += 1
                err = {
                    "key": f"deepseek|{kind}|ERROR|{feat['split_name']}|{feat['market']}|{feat['origin_date']}",
                    "kind": kind, "backend": "deepseek_api", "ok": False,
                    "reason": f"api_exception:{exc}", "feat": feat,
                }
                out.write(json.dumps(err, ensure_ascii=False) + "\n")
                out.flush()
            if i % 50 == 0 or i == len(futs):
                log(f"API progress {i}/{len(futs)} ok={n_ok} fail={n_fail}")
    log(f"collect done ok={n_ok} fail={n_fail}")


def load_judge_maps():
    sel, wonly = {}, {}
    if not JUDGE_PATH.exists():
        return sel, wonly
    with JUDGE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not obj.get("ok"):
                continue
            feat = obj.get("feat") or {}
            key = (feat.get("split_name"), feat.get("market"), feat.get("origin_date"))
            if obj.get("kind") == "select_weight":
                sel[key] = obj["payload"]
            elif obj.get("kind") == "weight_only":
                wonly[(key[0], key[1], key[2], tuple(obj.get("members") or obj["payload"]["selected_models"]))] = obj["payload"]
    return sel, wonly


def replay_with_deepseek(wide, single, eq_tasks, split, select_mode, weight_mode, eta, holiday_dates,
                         scheme_name, mode, sel_map, wonly_map, refresh_k=K0):
    """Causal replay using DeepSeek cache; fallback to G1/stat on miss/invalid."""
    from common import STRUCTURE
    sched = (
        wide.groupby(["fold_id", "origin_date"], sort=False)
        .agg(target_end=("target_date", "max"))
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    sched["origin_idx"] = np.arange(len(sched))
    single_lookup, combo_lookup = {}, {}
    for r in single.itertuples(index=False):
        single_lookup[(r.config_id, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)
    for cid, tdf in eq_tasks.items():
        for r in tdf.itertuples(index=False):
            combo_lookup[(cid, r.market, pd.Timestamp(r.origin_date))] = float(r.mae)

    parts, decisions = [], []
    fb = {"select_miss": 0, "weight_miss": 0, "ok": 0}
    models_run = {m: 0.0 for m in POOL}

    for market, mwide in wide.groupby("market", sort=False):
        for _, row in sched.iterrows():
            od = pd.Timestamp(row.origin_date)
            idx = int(row.origin_idx)
            sub = mwide[mwide.origin_date == od]
            if sub.empty:
                continue
            hist = sched[(sched.origin_idx < idx) & (sched.target_end <= od)]
            cold = len(hist) == 0
            combo_errors, single_errors = {}, {}
            for cid in [f"C{i}" for i in range(1, 10)]:
                vals, ages = [], []
                for _, h in hist.iterrows():
                    key = (cid, market, pd.Timestamp(h.origin_date))
                    if key in combo_lookup:
                        vals.append(combo_lookup[key]); ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                combo_errors[cid] = discounted_mean(np.array(vals), np.array(ages), LAMBDA0) if vals else np.nan
            for m in POOL:
                vals, ages = [], []
                for _, h in hist.iterrows():
                    key = (m, market, pd.Timestamp(h.origin_date))
                    if key in single_lookup:
                        vals.append(single_lookup[key]); ages.append((od - pd.Timestamp(h.origin_date)).days / 7.0)
                single_errors[m] = discounted_mean(np.array(vals), np.array(ages), LAMBDA0) if vals else np.nan

            pool_preds = sub[POOL].to_numpy(float)
            dispersion = float(pool_preds.std(axis=1).mean() / (np.abs(pool_preds.mean()) + 1e-6))
            dates = pd.to_datetime(sub.target_date).dt.date
            hol_n = int(sum(d in holiday_dates for d in dates.unique()))
            fallback = False
            fallback_reason = ""

            if cold:
                combo_id = "C2"
                members = COMBOS[combo_id]
                w = equal_weights(3)
                select_backend = "cold_default_C2"
                weight_backend = "cold_equal"
            else:
                # select
                if select_mode == "stat":
                    combo_id = select_combo(combo_errors, delta=DELTA0)
                    select_backend = "stat_select"
                else:
                    hit = sel_map.get((split, market, str(od.date())))
                    if hit:
                        combo_id = hit["combo_id"]
                        select_backend = "deepseek_select"
                    else:
                        combo_id = select_combo(combo_errors, delta=DELTA0)
                        select_backend = "fallback_stat_select"
                        fallback = True
                        fallback_reason = "select_miss"
                        fb["select_miss"] += 1
                members = COMBOS[combo_id]
                errs = np.array([single_errors.get(m, np.nan) for m in members], float)
                if not np.all(np.isfinite(errs)):
                    w = equal_weights(3)
                    weight_backend = "fallback_equal"
                    fallback = True
                    fallback_reason = fallback_reason or "nan_hist"
                else:
                    w_stat = stat_weights(errs, alpha=ALPHA0)
                    if weight_mode == "stat":
                        w = w_stat
                        weight_backend = "stat_weight"
                    else:
                        # LLM weights: for G3 use weight_only on stat members; for G4 use select_weight payload weights
                        if select_mode == "llm":
                            hit = sel_map.get((split, market, str(od.date())))
                            if hit and hit.get("combo_id") == combo_id:
                                w_llm = np.array(hit["weights"], float)
                                w = (1 - eta) * w_stat + eta * w_llm
                                w = w / w.sum()
                                weight_backend = f"deepseek_weight_fused_eta{eta}"
                                fb["ok"] += 1
                            else:
                                w = w_stat
                                weight_backend = "fallback_stat_weight"
                                fallback = True
                                fallback_reason = fallback_reason or "weight_miss_g4"
                                fb["weight_miss"] += 1
                        else:
                            wh = wonly_map.get((split, market, str(od.date()), tuple(members)))
                            if wh:
                                w_llm = np.array(wh["weights"], float)
                                w = (1 - eta) * w_stat + eta * w_llm
                                w = w / w.sum()
                                weight_backend = f"deepseek_weight_only_fused_eta{eta}"
                                fb["ok"] += 1
                            else:
                                w = w_stat
                                weight_backend = "fallback_stat_weight"
                                fallback = True
                                fallback_reason = fallback_reason or "weight_miss_g3"
                                fb["weight_miss"] += 1

            refresh = (idx % refresh_k == 0)
            run_models = list(POOL) if (mode == "mode1" or refresh) else list(members)
            for m in run_models:
                models_run[m] += 1

            yhat = combine_predictions(sub, members, w)
            chunk = sub[["market", "fold_id", "origin_date", "target_date", "horizon_day", "task_id", "actual", "q_all"]].copy()
            chunk["yhat"] = yhat
            parts.append(chunk)
            decisions.append({
                "scheme": scheme_name, "split_name": split, "market": market, "origin_date": od,
                "origin_idx": idx, "combo_id": combo_id, "structure": STRUCTURE[combo_id],
                "select_backend": select_backend, "weight_backend": weight_backend,
                "cold_start": cold, "fallback": fallback, "fallback_reason": fallback_reason,
                "dispersion": dispersion, "holiday_days": hol_n, "mode": mode, "refresh": refresh,
                "member_0": members[0], "member_1": members[1], "member_2": members[2],
                "w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]), "eta": eta,
            })

    day = pd.concat(parts, ignore_index=True)
    task = score_task_from_days(day, "yhat")
    task["config_id"] = scheme_name
    task["scheme"] = scheme_name
    meta = {"fallback_counts": fb, "models_run_counter": models_run, "n_decisions": len(decisions)}
    return task, pd.DataFrame(decisions), meta


def recompute_tables(task_all, days, holiday_dates, sel_map, wonly_map):
    s15_rows, decisions_all, task_dyn, metas = [], [], {}, {}
    for split in SPLITS:
        log(f"recompute {split}")
        wide = build_prediction_cube(days, split, POOL)
        single = task_all[task_all.split_name == split]
        eq = precompute_equal_combo_tasks(wide, [f"C{i}" for i in range(1, 10)])
        gran = single[single.config_id == "granite_patchtst_fm"]
        ets = single[single.config_id == "ets"]
        b3 = replay_fixed_equal(wide, "B_3FM")
        g1, g1dec, g1meta = replay_dynamic(
            wide, single, eq, split, "stat", "stat", eta=ETA0,
            holiday_dates=holiday_dates, scheme_name="G1_Stat_Stat", mode="mode1",
        )
        task_dyn[(split, "G1_Stat_Stat")] = g1
        decisions_all.append(g1dec)
        metas[(split, "G1_Stat_Stat")] = g1meta

        specs = [
            ("G1_Stat_Stat_Mode2", "stat", "stat", ETA0, "mode2"),
            ("G2_LLM_Stat_eta0.3", "llm", "stat", 0.3, "mode1"),
            ("G2_LLM_Stat_Mode2", "llm", "stat", 0.3, "mode2"),
            ("G3_Stat_LLM_eta0.3", "stat", "llm", 0.3, "mode1"),
            ("G3_Stat_LLM_eta0.5", "stat", "llm", 0.5, "mode1"),
            ("G3_Stat_LLM_eta1.0", "stat", "llm", 1.0, "mode1"),
            ("G4_LLM_LLM_eta0.3", "llm", "llm", 0.3, "mode1"),
            ("G4_LLM_LLM_eta0.5", "llm", "llm", 0.5, "mode1"),
            ("G4_LLM_LLM_eta1.0", "llm", "llm", 1.0, "mode1"),
            ("G4_LLM_LLM_Mode2_eta0.3", "llm", "llm", 0.3, "mode2"),
        ]
        # store G1 first into s15
        for name, sel, wt, eta, mode in [("G1_Stat_Stat", "stat", "stat", ETA0, "mode1")] + specs:
            if name == "G1_Stat_Stat":
                t, dec, meta = g1, g1dec, g1meta
            elif sel == "stat" and wt == "stat":
                t, dec, meta = replay_dynamic(
                    wide, single, eq, split, "stat", "stat", eta=eta,
                    holiday_dates=holiday_dates, scheme_name=name, mode=mode, refresh_k=K0,
                )
            else:
                t, dec, meta = replay_with_deepseek(
                    wide, single, eq, split, sel, wt, eta, holiday_dates, name, mode, sel_map, wonly_map,
                )
            task_dyn[(split, name)] = t
            decisions_all.append(dec)
            metas[(split, name)] = meta
            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            row = {
                "split_name": split, "period": SPLIT_LABEL[split], "scheme_id": name,
                "select_mode": sel if name != "G1_Stat_Stat" else "stat",
                "weight_mode": wt if name != "G1_Stat_Stat" else "stat",
                "eta": eta, "mode": mode, "llm_backend": "deepseek_api" if "LLM" in name else "none",
                **s, **ci,
                "fallback_total": int(sum(v for k, v in meta["fallback_counts"].items() if k != "ok")) if isinstance(meta["fallback_counts"], dict) else 0,
                "fallback_json": json.dumps(meta["fallback_counts"]),
            }
            for ref_name, ref_t in [("ETS", ets), ("Granite", gran), ("G1", g1), ("B3FM", b3)]:
                for metric, col in [("MASE", "mase"), ("MAE", "mae")]:
                    r, lo, hi = ratio_ci_vs(t, ref_t, col, split)
                    row[f"R_{metric}_vs_{ref_name}"] = r
                    row[f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                    row[f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
                    dm, p, st = pairwise_dm(t, ref_t, col)
                    row[f"DM_{metric}_vs_{ref_name}"] = dm
                    row[f"p_{metric}_vs_{ref_name}"] = p
                    row[f"sig_{metric}_vs_{ref_name}"] = st
            s15_rows.append(row)
            log(f"{split} {name} MASE={s['MASE']:.4f} fallback={row['fallback_total']}")

        # static anchors
        for sid, t in [
            ("C2_Equal", replay_fixed_equal(wide, "C2")),
            ("B_3FM_Equal", b3),
            ("C2_StatWeight", replay_fixed_stat_weight(wide, single, "C2", split)[0]),
            ("C7_StatWeight", replay_fixed_stat_weight(wide, single, "C7", split)[0]),
        ]:
            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            row = {"split_name": split, "period": SPLIT_LABEL[split], "scheme_id": sid,
                   "select_mode": "fixed", "weight_mode": "equal" if "Equal" in sid else "stat",
                   "eta": np.nan, "mode": "n/a", "llm_backend": "none", **s, **ci}
            for ref_name, ref_t in [("ETS", ets), ("Granite", gran), ("G1", g1), ("B3FM", b3)]:
                r, lo, hi = ratio_ci_vs(t, ref_t, "mase", split)
                row[f"R_MASE_vs_{ref_name}"] = r
                dm, p, st = pairwise_dm(t, ref_t, "mase")
                row[f"DM_MASE_vs_{ref_name}"] = dm
                row[f"p_MASE_vs_{ref_name}"] = p
                row[f"sig_MASE_vs_{ref_name}"] = st
            s15_rows.append(row)

    s15 = pd.DataFrame(s15_rows)
    decisions = pd.concat(decisions_all, ignore_index=True)
    s15.to_csv(TABLES / "table_S15_dynamic_selection_and_weighting_ablation.csv", index=False)
    decisions.to_parquet(CACHE / "all_decisions_deepseek.parquet", index=False)
    with (CACHE / "metas_deepseek.json").open("w", encoding="utf-8") as f:
        json.dump({f"{a}|{b}": m for (a, b), m in metas.items()}, f, indent=2)
    return s15, decisions, task_dyn, metas


def redraw_figures(s14, s15, s16, s17, s18):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # a
    ax = axes[0, 0]
    part = s14[(s14.split_name == "outer_backtest") & (s14.kind.isin(["single", "combo_equal"]))].copy()
    order = POOL + ["B_3FM"] + [f"C{i}" for i in range(1, 10)]
    part = part[part.scheme_id.isin(order)].copy()
    part["ord"] = part.scheme_id.map({k: i for i, k in enumerate(order)})
    part = part.sort_values("ord")
    y = np.arange(len(part))
    ax.errorbar(part.MASE, y, xerr=[part.MASE - part.MASE_ci95_low, part.MASE_ci95_high - part.MASE],
                fmt="o", color="#222", ms=4, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(s, s) for s in part.scheme_id], fontsize=7)
    ax.set_xlabel("MASE"); ax.set_title("a", loc="left", fontweight="bold"); style_axes(ax)
    # b
    ax = axes[0, 1]
    g = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id.isin([
        "G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G3_Stat_LLM_eta0.3", "G4_LLM_LLM_eta0.3"]))].copy()
    # preserve order
    order = ["G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G3_Stat_LLM_eta0.3", "G4_LLM_LLM_eta0.3"]
    g["ord"] = g.scheme_id.map({k: i for i, k in enumerate(order)})
    g = g.sort_values("ord")
    x = np.arange(len(g))
    improve = (1 - g.R_MASE_vs_Granite) * 100
    ax.bar(x, improve, color=["#222222", "#1F77B4", "#7B3294", "#E66101"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stat+Stat", "DS-Sel+Stat", "Stat+DS-W", "DS+DS"], rotation=15, fontsize=7)
    ax.set_ylabel("% MASE improvement vs Granite")
    ax.set_title("b", loc="left", fontweight="bold"); style_axes(ax)
    # c holiday shares if available
    ax = axes[1, 0]
    hol = s16[(s16.get("stratum_type") == "holiday_window") if "stratum_type" in s16.columns else []] if "stratum_type" in s16.columns else pd.DataFrame()
    if "stratum_type" in s16.columns:
        hol = s16[(s16.stratum_type == "holiday_window") & (s16.split_name == "outer_backtest") &
                  (s16.scheme_id.isin(["G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G4_LLM_LLM_eta0.3"]))]
        schemes = ["G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G4_LLM_LLM_eta0.3"]
        x = np.arange(len(schemes)); w = 0.35
        h = [hol[(hol.scheme_id == s) & (hol.stratum_value == "has_official_holiday")].share_1FM2LW.mean() for s in schemes]
        o = [hol[(hol.scheme_id == s) & (hol.stratum_value == "ordinary_non_holiday")].share_1FM2LW.mean() for s in schemes]
        ax.bar(x - w / 2, h, w, label="holiday", color="#E66101")
        ax.bar(x + w / 2, o, w, label="ordinary", color="#222")
        ax.set_xticks(x); ax.set_xticklabels(["G1", "G2", "G4"])
        ax.set_ylabel("Share 1FM+2LW"); ax.legend(fontsize=7)
    ax.set_title("c", loc="left", fontweight="bold"); style_axes(ax)
    # d eta curves
    ax = axes[1, 1]
    for pref, color, lab in [("G3_Stat_LLM_eta", "#7B3294", "Stat+DS-Weight"), ("G4_LLM_LLM_eta", "#E66101", "DS+DS")]:
        sub = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id.str.startswith(pref))].copy()
        if sub.empty:
            continue
        sub["eta"] = sub.scheme_id.str.extract(r"eta([0-9.]+)").astype(float)
        sub = sub.sort_values("eta")
        ax.plot(sub.eta, sub.MASE, "o-", color=color, label=lab)
    ax.axhline(float(s15[(s15.split_name == "outer_backtest") & (s15.scheme_id == "G1_Stat_Stat")].MASE.iloc[0]),
               color="#222", ls="--", label="G1")
    ax.set_xlabel("η"); ax.set_ylabel("MASE"); ax.legend(fontsize=7)
    ax.set_title("d", loc="left", fontweight="bold"); style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "figure11_hybrid_structures_and_2x2_ablation")
    plt.close(fig)

    # figure 12 reuse cost table
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, sname in zip(axes[:2], ["S1", "S2"]):
        xs, ys, labs = [], [], []
        for _, r in s18.iterrows():
            col = f"cost_ratio_vs_ETS_{sname}"
            if col not in s18.columns:
                continue
            hit = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id == r.scheme_id)]
            if hit.empty and r.scheme_id == "Granite_single":
                ys.append(float(s14[(s14.split_name == "outer_backtest") & (s14.scheme_id == "granite_patchtst_fm")].MASE.iloc[0]))
            elif hit.empty:
                continue
            else:
                ys.append(float(hit.MASE.iloc[0]))
            xs.append(float(r[col])); labs.append(r.scheme_id)
        ax.scatter(xs, ys, c="#222")
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(str(lab)[:16], (x, y), fontsize=5, alpha=0.8)
        ax.set_xscale("log"); ax.set_xlabel(f"Cost ratio vs ETS ({sname})")
        ax.set_ylabel("MASE"); ax.set_title("a" if sname == "S1" else "b", loc="left", fontweight="bold")
        style_axes(ax)
    axes[2].set_title("c", loc="left", fontweight="bold"); style_axes(axes[2])
    if len(s17):
        grid = s17[s17.grid == "refresh_K"] if "grid" in s17.columns else s17
        if len(grid) and "K" in grid.columns:
            axes[2].plot(grid.K, grid.MASE, "s--", color="#222")
            axes[2].set_xlabel("K"); axes[2].set_ylabel("MASE")
    fig.tight_layout()
    save_fig(fig, "figure12_mode1_mode2_cost_accuracy_frontier")
    plt.close(fig)
    log("figures redrawn with DeepSeek results")


def main():
    ensure_dirs()
    log("=== DeepSeek RQ3 start ===")
    env = load_rq3_env()
    log(f"model={env['DEEPSEEK_MODEL']} base={env['DEEPSEEK_BASE_URL']}")
    days = load_day_panel()
    task = load_task_panel()
    holiday_dates = load_holiday_flags()

    log("=== build feature jobs ===")
    jobs = build_feature_jobs(task, days, holiday_dates)
    pd.DataFrame([{k: j[k] for k in ["split_name", "market", "origin_date", "stat_combo", "holiday_days"]} for j in jobs]).to_csv(
        CACHE / "deepseek_feature_job_index.csv", index=False
    )
    log(f"non-cold decision points={len(jobs)}")

    log("=== collect DeepSeek judgments ===")
    collect_judgments(jobs, env, resume=True)
    sel_map, wonly_map = load_judge_maps()
    log(f"cached select_weight={len(sel_map)} weight_only={len(wonly_map)}")

    log("=== recompute S15 with DeepSeek ===")
    s15, decisions, task_dyn, metas = recompute_tables(task, days, holiday_dates, sel_map, wonly_map)

    # update S18 llm_backend labels
    s18 = pd.read_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv")
    s18["llm_backend"] = s18.scheme_id.map(lambda s: "deepseek_api" if "LLM" in str(s) else (
        "qwen3_4b_direct_forecast_rq1" if "Qwen" in str(s) else "none"))
    for i, row in s18.iterrows():
        part = decisions[(decisions.scheme == row.scheme_id) & (decisions.split_name == "outer_backtest")]
        if len(part):
            s18.at[i, "fallback_rate"] = float(part.fallback.mean())
            s18.at[i, "json_fail_or_fallback_rate"] = float(part.fallback.mean())
            s18.at[i, "n_decisions"] = int(len(part))
            s18.at[i, "cold_start_rate"] = float(part.cold_start.mean())
    s18.to_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv", index=False)

    # rebuild S16 holiday quickly from new decisions
    from fill_gaps import build_s16_complete
    s16 = build_s16_complete(decisions, days)
    s16.to_csv(TABLES / "table_S16_subset_switching_and_market_heterogeneity.csv", index=False)

    s14 = pd.read_csv(TABLES / "table_S14_fixed_combinations_benchmark.csv")
    s17 = pd.read_csv(TABLES / "table_S17_hyperparameter_sensitivity_grid.csv")
    redraw_figures(s14, s15, s16, s17, s18)

    # summary append
    lines = ["\n\n---\n\n# DeepSeek API 补全记录\n\n"]
    lines.append(f"- backend: `deepseek_api` model=`{env['DEEPSEEK_MODEL']}`\n")
    lines.append(f"- judgments file: `{JUDGE_PATH}`\n")
    lines.append(f"- select_weight cached: {len(sel_map)}; weight_only cached: {len(wonly_map)}\n\n")
    peek = s15[(s15.split_name == "outer_backtest") & s15.scheme_id.str.startswith("G")][
        ["scheme_id", "MASE", "MAE", "R_MASE_vs_Granite", "DM_MASE_vs_Granite", "fallback_total", "llm_backend"]
    ]
    try:
        lines.append(peek.to_markdown(index=False))
    except Exception:
        lines.append("```\n" + peek.to_string(index=False) + "\n```")
    lines.append("\n")
    man = Path(LOGS / "missing_items_manifest_rq3.csv")
    if man.exists():
        df = pd.read_csv(man)
        df.loc[df.item == "real_Qwen3_4B_llm_select_weight", "item"] = "real_llm_select_weight"
        df.loc[df.item == "real_llm_select_weight", "status"] = "filled_deepseek_api"
        df.loc[df.item == "real_llm_select_weight", "note"] = (
            f"Used DeepSeek API ({env['DEEPSEEK_MODEL']}) instead of Qwen3-4B; judgments in llm_judgments_deepseek.jsonl"
        )
        # if row missing add
        if not (df.item == "real_llm_select_weight").any():
            df = pd.concat([df, pd.DataFrame([{
                "item": "real_llm_select_weight", "status": "filled_deepseek_api",
                "note": f"DeepSeek API {env['DEEPSEEK_MODEL']}"
            }])], ignore_index=True)
        df.to_csv(man, index=False)
    path = OUT / "4.3_RQ3_实验数据核对汇总稿.md"
    path.write_text(path.read_text(encoding="utf-8") + "".join(lines), encoding="utf-8")
    log("=== DeepSeek RQ3 DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
