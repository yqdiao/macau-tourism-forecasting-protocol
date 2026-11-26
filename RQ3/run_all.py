#!/usr/bin/env python3
"""RQ3 full pipeline: S14–S18, Figures 11–12, summary MD. Read-only vs RQ1/RQ2."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ALPHA0, CACHE, COMBOS, DELTA0, DEVICE, ETA0, FIGURES, FM, K0, LABEL, LAMBDA0, LLM_SEC_PER_JUDGMENT,
    LOGS, LW, OUT, POOL, RHO, RQ1, STRUCTURE, TABLES, ensure_dirs, log, save_fig, style_axes, stars,
)
from data import (
    build_prediction_cube, load_annual_volume, load_day_panel, load_holiday_flags,
    load_task_panel, load_timing, score_task_from_days,
)
from ensemble import JudgmentCache, equal_weights, combine_predictions
from metrics import bootstrap_levels, pairwise_dm, ratio_ci_vs, summarize_task
from replay import (
    error_corr_and_dispersion, precompute_equal_combo_tasks, replay_dynamic,
    replay_fixed_equal, replay_fixed_stat_weight,
)


SPLITS = ["outer_backtest", "pandemic_stress"]
SPLIT_LABEL = {"outer_backtest": "reopening", "pandemic_stress": "pandemic_stress"}


def verify_baselines(task: pd.DataFrame) -> None:
    t3 = pd.read_csv(RQ1 / "tables/table3_accuracy.csv").set_index("config_id")
    rows = []
    for m in POOL + ["ets"]:
        part = task[(task.config_id == m) & (task.split_name == "outer_backtest")]
        s = summarize_task(part)
        for metric, col in [("MASE", "MASE"), ("MAPE", "MAPE_pct"), ("MAE", "MAE"), ("RMSE", "RMSE")]:
            ref = float(t3.loc[m, col if metric != "MAPE" else "MAPE_pct"])
            got = s[metric]
            rows.append({"model": m, "metric": metric, "rq3": got, "table3": ref, "abs_diff": abs(got - ref)})
            assert abs(got - ref) < 1e-5, (m, metric, got, ref)
    pd.DataFrame(rows).to_csv(LOGS / "baseline_alignment_vs_table3.csv", index=False)
    log(f"baseline alignment OK; max abs diff={pd.DataFrame(rows).abs_diff.max():.3e}")


def build_s14(task_all: pd.DataFrame, days: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    rows = []
    refs = {}
    for split in SPLITS:
        refs[split] = {
            "ets": task_all[(task_all.config_id == "ets") & (task_all.split_name == split)],
            "granite": task_all[(task_all.config_id == "granite_patchtst_fm") & (task_all.split_name == split)],
        }
        wide = build_prediction_cube(days, split, POOL)
        # singles
        schemes = []
        for m in POOL:
            t = task_all[(task_all.config_id == m) & (task_all.split_name == split)].copy()
            t["scheme"] = m
            schemes.append(("single", m, LABEL[m], t, None, [m]))
        # equal combos
        for cid in list(COMBOS.keys()):
            t = replay_fixed_equal(wide, cid)
            schemes.append(("combo_equal", cid, cid, t, STRUCTURE[cid], COMBOS[cid]))
        # B_3FM stat weight
        t_sw, _ = replay_fixed_stat_weight(wide, task_all[task_all.split_name == split], "B_3FM", split)
        schemes.append(("combo_statweight", "B_3FM_StatWeight", "B_3FM_StatWeight", t_sw, "3FM", COMBOS["B_3FM"]))

        # structure group means (equal C1-C6 / C7-C9)
        group_tasks = {"2FM+1LW": [], "1FM+2LW": []}
        for kind, sid, name, t, struct, members in schemes:
            if struct in group_tasks:
                group_tasks[struct].append(t.assign(_sid=sid))

        for kind, sid, name, t, struct, members in schemes:
            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            row = {
                "split_name": split,
                "period": SPLIT_LABEL[split],
                "kind": kind,
                "scheme_id": sid,
                "display_name": name,
                "structure": struct if struct else ("single" if kind == "single" else ""),
                **s,
                **ci,
            }
            # ratios vs ETS / Granite / B_3FM equal
            for ref_name, ref_task, metcol in [
                ("ETS", refs[split]["ets"], "mase"),
                ("Granite", refs[split]["granite"], "mase"),
            ]:
                for metric, col in [("MASE", "mase"), ("MAPE", "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
                    r, lo, hi = ratio_ci_vs(t, ref_task, col, split)
                    row[f"R_{metric}_vs_{ref_name}"] = r
                    row[f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                    row[f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
                dm, p, st = pairwise_dm(t, ref_task, "mase")
                row[f"DM_MASE_vs_{ref_name}"] = dm
                row[f"p_MASE_vs_{ref_name}"] = p
                row[f"sig_MASE_vs_{ref_name}"] = st

            # complementarity on day level for combos
            if kind.startswith("combo") and len(members) >= 2:
                # use equal-weight members predictions from wide
                corr, disp = error_corr_and_dispersion(wide, members[:3] if len(members) >= 3 else members)
                row["error_corr_mean"] = corr
                row["forecast_dispersion"] = disp
            else:
                row["error_corr_mean"] = np.nan
                row["forecast_dispersion"] = np.nan
            rows.append(row)

        # group averages as rows
        for gname, tlist in group_tasks.items():
            if not tlist:
                continue
            # average metrics across schemes (not pooled tasks) — report mean of scheme metrics
            metrics = pd.DataFrame([summarize_task(t) for t in tlist])
            rows.append({
                "split_name": split,
                "period": SPLIT_LABEL[split],
                "kind": "structure_group_mean",
                "scheme_id": f"GROUP_{gname}",
                "display_name": f"Mean of {gname}",
                "structure": gname,
                "MASE": float(metrics.MASE.mean()),
                "MAPE": float(metrics.MAPE.mean()),
                "MAE": float(metrics.MAE.mean()),
                "RMSE": float(metrics.RMSE.mean()),
                "n_tasks": int(metrics.n_tasks.iloc[0]),
            })

        # attach B_3FM equal for later DM refs
        refs[split]["B_3FM"] = replay_fixed_equal(wide, "B_3FM")

    # second pass DM vs B_3FM
    out = pd.DataFrame(rows)
    extra = []
    for split in SPLITS:
        wide = build_prediction_cube(days, split, POOL)
        b3 = refs[split]["B_3FM"]
        for _, row in out[out.split_name == split].iterrows():
            if row.kind == "structure_group_mean":
                continue
            # rebuild task for this scheme is expensive; store DM only for key schemes already computed above vs granite/ets
            pass
    # Compute DM vs B_3FM for all non-group rows by replaying quickly from cache of tasks
    # We'll recompute in a dedicated loop storing tasks
    return out


def build_s14_v2(task_all: pd.DataFrame, days: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    task_store = {}
    for split in SPLITS:
        wide = build_prediction_cube(days, split, POOL)
        ets = task_all[(task_all.config_id == "ets") & (task_all.split_name == split)]
        gran = task_all[(task_all.config_id == "granite_patchtst_fm") & (task_all.split_name == split)]
        local_tasks = {}
        # singles
        for m in POOL:
            t = task_all[(task_all.config_id == m) & (task_all.split_name == split)].copy()
            t["scheme"] = m
            local_tasks[m] = t
        for cid in COMBOS:
            t = replay_fixed_equal(wide, cid)
            local_tasks[cid] = t
        t_sw, wdf = replay_fixed_stat_weight(wide, task_all[task_all.split_name == split], "B_3FM", split)
        local_tasks["B_3FM_StatWeight"] = t_sw
        wdf.to_csv(CACHE / f"weights_B3FM_stat_{split}.csv", index=False)

        b3 = local_tasks["B_3FM"]
        for sid, t in local_tasks.items():
            kind = "single" if sid in POOL else ("combo_statweight" if sid.endswith("StatWeight") else "combo_equal")
            struct = STRUCTURE.get(sid.replace("_StatWeight", ""), "single" if sid in POOL else "")
            members = COMBOS.get(sid.replace("_StatWeight", ""), [sid] if sid in POOL else [])
            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            row = {
                "split_name": split, "period": SPLIT_LABEL[split], "kind": kind,
                "scheme_id": sid, "display_name": LABEL.get(sid, sid), "structure": struct,
                **s, **ci,
            }
            for ref_name, ref_t in [("ETS", ets), ("Granite", gran), ("B3FM", b3)]:
                for metric, col in [("MASE", "mase"), ("MAPE", "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
                    r, lo, hi = ratio_ci_vs(t, ref_t, col, split)
                    row[f"R_{metric}_vs_{ref_name}"] = r
                    row[f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                    row[f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
                dm, p, st = pairwise_dm(t, ref_t, "mase")
                row[f"DM_MASE_vs_{ref_name}"] = dm
                row[f"p_MASE_vs_{ref_name}"] = p
                row[f"sig_MASE_vs_{ref_name}"] = st
            if len(members) >= 3:
                corr, disp = error_corr_and_dispersion(wide, members[:3])
                row["error_corr_mean"] = corr
                row["forecast_dispersion"] = disp
            else:
                row["error_corr_mean"] = np.nan
                row["forecast_dispersion"] = np.nan
            rows.append(row)
            task_store[(split, sid)] = t

        # group means
        for gname, ids in [("2FM+1LW", [f"C{i}" for i in range(1, 7)]), ("1FM+2LW", [f"C{i}" for i in range(7, 10)])]:
            mets = pd.DataFrame([summarize_task(local_tasks[i]) for i in ids])
            rows.append({
                "split_name": split, "period": SPLIT_LABEL[split], "kind": "structure_group_mean",
                "scheme_id": f"GROUP_{gname}", "display_name": f"Mean of {gname}", "structure": gname,
                "MASE": float(mets.MASE.mean()), "MAPE": float(mets.MAPE.mean()),
                "MAE": float(mets.MAE.mean()), "RMSE": float(mets.RMSE.mean()),
                "n_tasks": int(mets.n_tasks.iloc[0]),
            })
    return pd.DataFrame(rows), task_store


def build_dynamic_tables(task_all, days, holiday_dates):
    s15_rows, s16_rows = [], []
    decision_all = []
    task_dyn = {}
    cache = JudgmentCache()
    metas = {}

    for split in SPLITS:
        wide = build_prediction_cube(days, split, POOL)
        single = task_all[task_all.split_name == split]
        eq_tasks = precompute_equal_combo_tasks(wide, [f"C{i}" for i in range(1, 10)])
        # also keep equal static baselines
        static_c2 = replay_fixed_equal(wide, "C2")
        task_dyn[(split, "C2_Equal")] = static_c2
        b3 = replay_fixed_equal(wide, "B_3FM")
        task_dyn[(split, "B_3FM_Equal")] = b3
        gran = single[single.config_id == "granite_patchtst_fm"]

        # Fixed subset + stat weight for C2 and C7 as anchors
        for cid in ["C2", "C7"]:
            t, _ = replay_fixed_stat_weight(wide, single, cid, split)
            task_dyn[(split, f"{cid}_StatWeight")] = t

        specs = [
            ("G1_Stat_Stat", "stat", "stat", ETA0, "mode1"),
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
        g1_task = None
        for name, sel, wt, eta, mode in specs:
            log(f"replay {split} {name}")
            t, dec, meta = replay_dynamic(
                wide, single, eq_tasks, split,
                select_mode=sel, weight_mode=wt, eta=eta,
                holiday_dates=holiday_dates, scheme_name=name, mode=mode,
                cache=cache,
            )
            task_dyn[(split, name)] = t
            metas[(split, name)] = meta
            decision_all.append(dec)
            if name == "G1_Stat_Stat":
                g1_task = t

            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            row = {
                "split_name": split, "period": SPLIT_LABEL[split], "scheme_id": name,
                "select_mode": sel, "weight_mode": wt, "eta": eta, "mode": mode,
                **s, **ci,
                "fallback_total": int(sum(meta["fallback_counts"].values())),
                "fallback_json": json.dumps(meta["fallback_counts"]),
            }
            for ref_name, ref_t in [("Granite", gran), ("G1", g1_task if g1_task is not None else t), ("B3FM", b3)]:
                if ref_t is None:
                    continue
                dm, p, st = pairwise_dm(t, ref_t, "mase")
                row[f"DM_MASE_vs_{ref_name}"] = dm
                row[f"p_MASE_vs_{ref_name}"] = p
                row[f"sig_MASE_vs_{ref_name}"] = st
                r, lo, hi = ratio_ci_vs(t, ref_t, "mase", split)
                row[f"R_MASE_vs_{ref_name}"] = r
                row[f"R_MASE_vs_{ref_name}_ci95_low"] = lo
                row[f"R_MASE_vs_{ref_name}_ci95_high"] = hi
            s15_rows.append(row)

            # S16 behavior from decisions
            dec = dec.copy()
            dec["market_group"] = dec.market.map(lambda m: __import__("common", fromlist=["MARKET_GROUP"]).MARKET_GROUP.get(m, "Small"))
            for (mg,), part in dec.groupby(["market_group"]):
                s16_rows.append({
                    "split_name": split, "scheme_id": name, "market_group": mg,
                    "n": len(part),
                    "share_2FM1LW": float((part.structure == "2FM+1LW").mean()),
                    "share_1FM2LW": float((part.structure == "1FM+2LW").mean()),
                    "cold_start_share": float(part.cold_start.mean()),
                    "fallback_share": float(part.fallback.mean()),
                })
            # activation rates + mean weight when active
            for m in POOL:
                active = (dec.member_0 == m) | (dec.member_1 == m) | (dec.member_2 == m)
                act = float(active.mean())
                wvals = []
                for r in dec.itertuples(index=False):
                    for j, mem in enumerate([r.member_0, r.member_1, r.member_2]):
                        if mem == m:
                            wvals.append([r.w0, r.w1, r.w2][j])
                s16_rows.append({
                    "split_name": split, "scheme_id": name, "market_group": "ALL",
                    "model": m, "activation_rate": act,
                    "mean_weight_when_active": float(np.mean(wvals)) if wvals else 0.0,
                    "n": int(len(dec)),
                })

    s15 = pd.DataFrame(s15_rows)
    # add static anchors into s15
    for split in SPLITS:
        gran = task_all[(task_all.config_id == "granite_patchtst_fm") & (task_all.split_name == split)]
        for sid in ["C2_Equal", "B_3FM_Equal", "C2_StatWeight", "C7_StatWeight"]:
            if (split, sid) not in task_dyn:
                continue
            t = task_dyn[(split, sid)]
            s = summarize_task(t)
            ci = bootstrap_levels(t, split)
            s15 = pd.concat([s15, pd.DataFrame([{
                "split_name": split, "period": SPLIT_LABEL[split], "scheme_id": sid,
                "select_mode": "fixed", "weight_mode": "equal" if "Equal" in sid else "stat",
                "eta": np.nan, "mode": "n/a", **s, **ci,
            }])], ignore_index=True)

    decisions = pd.concat(decision_all, ignore_index=True)
    decisions.to_parquet(CACHE / "all_decisions.parquet", index=False)
    return s15, pd.DataFrame(s16_rows), task_dyn, metas, decisions


def build_s17(task_all, days, holiday_dates, task_dyn):
    rows = []
    split = "outer_backtest"
    wide = build_prediction_cube(days, split, POOL)
    single = task_all[task_all.split_name == split]
    eq_tasks = precompute_equal_combo_tasks(wide, [f"C{i}" for i in range(1, 10)])
    gran = single[single.config_id == "granite_patchtst_fm"]
    cache = JudgmentCache(CACHE / "llm_judgments_sensitivity.jsonl")

    for lam in [0.70, 0.85, 1.00]:
        for alpha in [0.00, 0.20, 0.50]:
            t, dec, meta = replay_dynamic(
                wide, single, eq_tasks, split, "stat", "stat",
                lam=lam, alpha=alpha, delta=DELTA0,
                holiday_dates=holiday_dates,
                scheme_name=f"G1_lam{lam}_a{alpha}", mode="mode1", cache=cache,
            )
            s = summarize_task(t)
            rows.append({"grid": "lambda_alpha", "lambda": lam, "alpha": alpha, "delta": DELTA0,
                         "driver": "mae", "K": K0, **s})

    for delta in [0.00, 0.02, 0.05]:
        t, _, _ = replay_dynamic(
            wide, single, eq_tasks, split, "stat", "stat",
            delta=delta, holiday_dates=holiday_dates,
            scheme_name=f"G1_delta{delta}", mode="mode1", cache=cache,
        )
        s = summarize_task(t)
        rows.append({"grid": "delta", "lambda": LAMBDA0, "alpha": ALPHA0, "delta": delta,
                     "driver": "mae", "K": K0, **s})

    for driver in ["mae", "mase"]:
        t, _, _ = replay_dynamic(
            wide, single, eq_tasks, split, "stat", "stat",
            driver=driver, holiday_dates=holiday_dates,
            scheme_name=f"G1_driver_{driver}", mode="mode1", cache=cache,
        )
        s = summarize_task(t)
        rows.append({"grid": "driver", "lambda": LAMBDA0, "alpha": ALPHA0, "delta": DELTA0,
                     "driver": driver, "K": K0, **s})

    for K in [2, 4, 8]:
        t, _, meta = replay_dynamic(
            wide, single, eq_tasks, split, "stat", "stat",
            holiday_dates=holiday_dates, scheme_name=f"G1_Mode2_K{K}",
            mode="mode2", refresh_k=K, cache=cache,
        )
        s = summarize_task(t)
        rows.append({"grid": "refresh_K", "lambda": LAMBDA0, "alpha": ALPHA0, "delta": DELTA0,
                     "driver": "mae", "K": K, "mode": "mode2", **s,
                     "models_run": json.dumps(meta["models_run_counter"])})
    return pd.DataFrame(rows)


def build_s18(task_dyn, metas, decisions, task_all):
    timing = load_timing().set_index("model")
    annual = load_annual_volume()
    rows = []

    def model_sec(m):
        return float(timing.loc[m, "seconds_per_vector"])

    def conditional_cost(sec_gpu, sec_cpu, rho):
        return sec_gpu * rho + sec_cpu

    # ETS reference per-vector seconds (CPU)
    ets_sec = model_sec("ets")

    # For each scheme estimate total compute on reopen evaluation set
    split = "outer_backtest"
    n_tasks = 891  # 27*33
    # Mode1: always all 5 models
    mode1_gpu = sum(model_sec(m) for m in FM) * n_tasks
    mode1_cpu = sum(model_sec(m) for m in LW) * n_tasks

    schemes_cost = [
        ("Granite_single", "mode1", {"granite_patchtst_fm": n_tasks}, 0),
        ("B_3FM_Equal", "mode1", {m: n_tasks for m in FM}, 0),
        ("B_5Pool_Equal", "mode1", {m: n_tasks for m in POOL}, 0),
        ("G1_Stat_Stat", "mode1", {m: n_tasks for m in POOL}, 0),
        ("G1_Stat_Stat_Mode2", "mode2", None, 0),
        ("G4_LLM_LLM_eta0.3", "mode1", {m: n_tasks for m in POOL}, n_tasks),
        ("G4_LLM_LLM_Mode2_eta0.3", "mode2", None, n_tasks),
        ("Qwen3-4B_direct", "direct_llm", {}, n_tasks),  # direct forecast baseline cost
    ]

    # pull mode2 counters from metas when available
    for scheme, mode, counter, n_llm in schemes_cost:
        if counter is None:
            meta = metas.get((split, scheme)) or metas.get((split, scheme.replace("Mode2", "Mode2")))
            # try exact keys
            key = (split, scheme)
            if key not in metas:
                # map names
                alt = scheme
                key = (split, alt)
            counter = metas.get(key, {}).get("models_run_counter", {m: n_tasks * 0.6 for m in POOL})
        gpu = sum(model_sec(m) * counter.get(m, 0) for m in FM)
        cpu = sum(model_sec(m) * counter.get(m, 0) for m in LW)
        gpu += n_llm * LLM_SEC_PER_JUDGMENT
        row = {
            "scheme_id": scheme, "mode": mode,
            "gpu_seconds_eval_reopen": gpu, "cpu_seconds_eval_reopen": cpu,
            "llm_judgments": n_llm,
        }
        for sname, rho in RHO.items():
            c = conditional_cost(gpu, cpu, rho)
            c_ets = conditional_cost(0.0, ets_sec * n_tasks, rho)  # ETS is CPU-only
            # Actually ETS cost = ets_sec * n_tasks (CPU). Conditional cost ratio uses ρ for GPU.
            # Match RQ1 definition: cost = gpu_sec*ρ + cpu_sec, ratio vs ETS cpu_sec
            row[f"cond_cost_{sname}"] = c
            row[f"cost_ratio_vs_ETS_{sname}"] = c / (ets_sec * n_tasks)
        # annual USD scaling using chronos/granite annual hours as templates — linear in seconds
        # annual_hours_inference_only for granite 27 ≈ from table; scale by relative seconds
        for N in [27, 1000, 10000]:
            # vectors/year approx from annual volume for granite: hours * 3600 / sec
            g_ann = annual[(annual.model == "granite_patchtst_fm") & (annual.series == N)]
            if len(g_ann):
                # scale total scheme seconds relative to granite single annual
                gran_sec_eval = model_sec("granite_patchtst_fm") * n_tasks
                scale = (gpu + cpu) / max(gran_sec_eval, 1e-12)
                # use cloud on-demand $ from annual cost table for granite * scale — approximate
                ac = pd.read_csv(RQ1 / "tables/table4_annual_cost.csv")
                gcost = ac[(ac.model == "granite_patchtst_fm") & (ac.series == N)]
                if len(gcost):
                    row[f"annual_usd_S1_N{N}"] = float(gcost.annual_usd_cloud_on_demand.iloc[0]) * scale
                else:
                    row[f"annual_usd_S1_N{N}"] = np.nan
            else:
                row[f"annual_usd_S1_N{N}"] = np.nan
        rows.append(row)

    # reliability from decisions
    rel = []
    for scheme in decisions.scheme.unique():
        part = decisions[(decisions.scheme == scheme) & (decisions.split_name == "outer_backtest")]
        rel.append({
            "scheme_id": scheme,
            "json_fail_or_fallback_rate": float(part.fallback.mean()) if len(part) else np.nan,
            "fallback_rate": float(part.fallback.mean()) if len(part) else np.nan,
            "cold_start_rate": float(part.cold_start.mean()) if len(part) else np.nan,
            "n_decisions": int(len(part)),
        })
    # Qwen direct extreme collapse from RQ1 appendix
    qpath = RQ1 / "raw/qwen_extreme_tasks.csv"
    q_collapse = np.nan
    if qpath.exists():
        qe = pd.read_csv(qpath)
        q_collapse = float(len(qe)) / 891.0 if len(qe) else np.nan
    rel.append({"scheme_id": "Qwen3-4B_direct", "extreme_collapse_rate": q_collapse,
                "json_fail_or_fallback_rate": np.nan, "fallback_rate": np.nan, "n_decisions": 891})
    # merge
    cost_df = pd.DataFrame(rows)
    rel_df = pd.DataFrame(rel)
    out = cost_df.merge(rel_df, on="scheme_id", how="left")
    return out


def plot_figures(s14, s15, s16, s17, decisions):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 9})
    # Figure 11 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # a: forest MASE reopen
    ax = axes[0, 0]
    part = s14[(s14.split_name == "outer_backtest") & (s14.kind.isin(["single", "combo_equal"]))].copy()
    keep = POOL + ["B_3FM"] + [f"C{i}" for i in range(1, 10)]
    part = part[part.scheme_id.isin(keep)].copy()
    # order
    order = POOL + ["B_3FM"] + [f"C{i}" for i in range(1, 10)]
    part["ord"] = part.scheme_id.map({k: i for i, k in enumerate(order)})
    part = part.sort_values("ord")
    y = np.arange(len(part))
    ax.errorbar(part.MASE, y, xerr=[part.MASE - part.MASE_ci95_low, part.MASE_ci95_high - part.MASE],
                fmt="o", color="#222", ms=4, capsize=2, label="MASE")
    ax2 = ax.twiny()
    ax2.errorbar(part.MAE, y + 0.15, xerr=[part.MAE - part.MAE_ci95_low, part.MAE_ci95_high - part.MAE],
                 fmt="D", color="#E66101", ms=3, capsize=2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(s, s) for s in part.scheme_id], fontsize=7)
    ax.set_xlabel("MASE")
    ax2.set_xlabel("MAE")
    ax.set_title("a", loc="left", fontweight="bold")
    style_axes(ax)

    # b: 2x2 ablation bars vs Granite
    ax = axes[0, 1]
    g = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id.isin([
        "G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G3_Stat_LLM_eta0.3", "G4_LLM_LLM_eta0.3"
    ]))]
    x = np.arange(len(g))
    improve = (1 - g.R_MASE_vs_Granite) * 100
    ax.bar(x, improve, color=["#222222", "#1F77B4", "#7B3294", "#E66101"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stat+Stat", "LLM-Sel+Stat", "Stat+LLM-W", "LLM+LLM"], rotation=15, fontsize=7)
    ax.set_ylabel("% MASE improvement vs Granite")
    ax.set_title("b", loc="left", fontweight="bold")
    style_axes(ax)

    # c: activation rates G1
    ax = axes[1, 0]
    act = s16[(s16.scheme_id == "G1_Stat_Stat") & (s16.market_group == "ALL") & (s16.get("model") == s16.get("model"))]
    act = s16[(s16.scheme_id.str.startswith("G1_Stat_Stat")) & (s16.split_name == "outer_backtest") & (s16.market_group == "ALL") & s16.model.notna()]
    if len(act):
        # one scheme exact
        act = act[act.scheme_id == "G1_Stat_Stat"]
        ax.bar([LABEL[m] for m in act.model], act.activation_rate, color="#E66101")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Activation rate")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    ax.set_title("c", loc="left", fontweight="bold")
    style_axes(ax)

    # d: eta response from s15 G3/G4
    ax = axes[1, 1]
    for pref, color, lab in [
        ("G3_Stat_LLM_eta", "#7B3294", "Stat+LLM-Weight"),
        ("G4_LLM_LLM_eta", "#E66101", "LLM+LLM"),
    ]:
        sub = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id.str.startswith(pref))]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["eta"] = sub.scheme_id.str.extract(r"eta([0-9.]+)").astype(float)
        sub = sub.sort_values("eta")
        ax.plot(sub.eta, sub.MASE, "o-", color=color, label=lab)
    # alpha sensitivity from s17
    grid = s17[(s17.grid == "lambda_alpha") & (s17["lambda"] == 0.85)]
    if len(grid):
        ax.plot(grid.alpha, grid.MASE, "s--", color="#222", label="G1 α (λ=0.85)")
    ax.set_xlabel("η or α")
    ax.set_ylabel("MASE (reopening)")
    ax.legend(fontsize=7)
    ax.set_title("d", loc="left", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "figure11_hybrid_structures_and_2x2_ablation")
    plt.close(fig)

    # Figure 12
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    s18 = pd.read_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv")
    for ax, sname in zip(axes[:2], ["S1", "S2"]):
        # merge MASE
        mase_map = {}
        for sid in s18.scheme_id:
            # find in s15 or s14
            hit = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id == sid)]
            if len(hit):
                mase_map[sid] = float(hit.MASE.iloc[0])
            else:
                hit = s14[(s14.split_name == "outer_backtest") & (s14.scheme_id == sid.replace("_Equal", "").replace("B_5Pool_equal", "B_5Pool").replace("B_3FM_Equal", "B_3FM").replace("Granite_single", "granite_patchtst_fm"))]
                if len(hit):
                    mase_map[sid] = float(hit.MASE.iloc[0])
        xs, ys, labs = [], [], []
        for _, r in s18.iterrows():
            if r.scheme_id not in mase_map and r.scheme_id != "Qwen3-4B_direct":
                continue
            xs.append(r[f"cost_ratio_vs_ETS_{sname}"])
            if r.scheme_id == "Qwen3-4B_direct":
                # from RQ1 qwen metrics
                q = pd.read_parquet(RQ1 / "raw/qwen3_4b_task_metrics.parquet")
                ys.append(float(q.groupby("market").mase.mean().mean()))
            else:
                ys.append(mase_map.get(r.scheme_id, np.nan))
            labs.append(r.scheme_id)
        ax.scatter(xs, ys, c="#222")
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab.replace("G", "").replace("_", "\n")[:18], (x, y), fontsize=5, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"Cost ratio vs ETS ({sname})")
        ax.set_ylabel("MASE")
        ax.set_title("a" if sname == "S1" else "b", loc="left", fontweight="bold")
        style_axes(ax)
    # c: K tradeoff
    ax = axes[2]
    grid = s17[s17.grid == "refresh_K"]
    if len(grid):
        # savings approx 1 - mean runs/5
        sav = []
        for _, r in grid.iterrows():
            ctr = json.loads(r.models_run) if isinstance(r.models_run, str) else {}
            total = sum(ctr.values()) if ctr else 0
            full = 5 * 891
            sav.append(1 - total / full if full else np.nan)
        ax.plot(grid.K, sav, "o-", color="#E66101", label="GPU/CPU run savings")
        ax2 = ax.twinx()
        ax2.plot(grid.K, grid.MASE, "s--", color="#222", label="MASE")
        ax.set_xlabel("Refresh K")
        ax.set_ylabel("Compute savings share")
        ax2.set_ylabel("MASE")
    ax.set_title("c", loc="left", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "figure12_mode1_mode2_cost_accuracy_frontier")
    plt.close(fig)
    log("figures written")


def write_summary(s14, s15, s16, s17, s18):
    lines = ["# RQ3 verification summary\n\n"]
    lines.append("Generated by `RQ3/run_all.py`. RQ1/RQ2 sealed outputs are read-only.\n\n")
    lines.append("## Checklist\n\n")
    base = pd.read_csv(LOGS / "baseline_alignment_vs_table3.csv")
    lines.append(f"- Single-model metrics vs Table 3 max abs diff = `{base.abs_diff.max():.3e}`\n")
    lines.append("- Replay history rule: `target_end(r) ≤ origin(t)` (28-day origin spacing ⇒ cold start only at the first origin)\n")
    lines.append("- Cold-start default subset: equal-weight `C2` (1/3 each)\n")
    lines.append("- Prefer DeepSeek caches (`cache/*_deepseek.*`). Early `llm_proxy` files are not for manuscript citation.\n")
    lines.append("- Weights non-negative and sum to one; failures fall back to G1\n\n")
    for name, df in [("S14", s14), ("S15", s15), ("S16", s16), ("S17", s17), ("S18", s18)]:
        lines.append(f"## Table {name}\n\n")
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("```\n" + df.to_string(index=False) + "\n```")
        lines.append("\n\n")
    text = "".join(lines)
    (OUT / "RQ3_verification_summary.md").write_text(text, encoding="utf-8")
    log("summary written")


def main():
    ensure_dirs()
    (LOGS / "rq3_run.log").write_text("", encoding="utf-8")
    log("=== RQ3 start ===")
    days = load_day_panel()
    task = load_task_panel()
    holiday_dates = load_holiday_flags()
    log(f"loaded days={len(days)} tasks={len(task)} holidays={len(holiday_dates)}")
    verify_baselines(task)

    log("=== S14 ===")
    s14, task_store = build_s14_v2(task, days)
    s14.to_csv(TABLES / "table_S14_fixed_combinations_benchmark.csv", index=False)

    log("=== S15/S16 dynamic ===")
    s15, s16, task_dyn, metas, decisions = build_dynamic_tables(task, days, holiday_dates)
    # DM vs G1 fill for rows where G1 exists
    s15.to_csv(TABLES / "table_S15_dynamic_selection_and_weighting_ablation.csv", index=False)
    s16.to_csv(TABLES / "table_S16_subset_switching_and_market_heterogeneity.csv", index=False)
    with (CACHE / "metas.json").open("w", encoding="utf-8") as f:
        json.dump({f"{a}|{b}": m for (a, b), m in metas.items()}, f, indent=2)

    log("=== S17 sensitivity ===")
    s17 = build_s17(task, days, holiday_dates, task_dyn)
    s17.to_csv(TABLES / "table_S17_hyperparameter_sensitivity_grid.csv", index=False)

    log("=== S18 costs ===")
    s18 = build_s18(task_dyn, metas, decisions, task)
    s18.to_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv", index=False)

    # LL01 sensitivity appendix (optional)
    log("=== LL01 sensitivity appendix ===")
    try:
        wide = build_prediction_cube(days, "outer_backtest", ["granite_patchtst_fm", "timesfm_3", "chronos_2", "NL01", "LL01"])
        # replace GL03 with LL01 in C2-like
        from common import COMBOS as _C
        # manual
        df = wide.copy()
        members = ["granite_patchtst_fm", "timesfm_3", "LL01"]
        df["yhat"] = combine_predictions(df, members, equal_weights(3))
        t = score_task_from_days(df, "yhat")
        s = summarize_task(t)
        pd.DataFrame([{"scheme": "C2_LL01_sens", **s}]).to_csv(TABLES / "table_S14_appendix_LL01_sensitivity.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        log(f"LL01 sensitivity skipped: {exc}")

    log("=== figures ===")
    plot_figures(s14, s15, s16, s17, decisions)
    write_summary(s14, s15, s16, s17, s18)
    log("=== RQ3 DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
