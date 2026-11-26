#!/usr/bin/env python3
"""Fill RQ3 remaining gaps on sealed predictions (read-only RQ1/RQ2)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CACHE, COMBOS, FIGURES, LABEL, LOGS, OUT, POOL, STRUCTURE, TABLES, ensure_dirs, log,
)
from data import build_prediction_cube, load_day_panel, load_holiday_flags, load_task_panel
from ensemble import JudgmentCache
from metrics import bootstrap_levels, pairwise_dm, ratio_ci_vs, summarize_task
from replay import precompute_equal_combo_tasks, replay_dynamic, replay_fixed_equal, replay_fixed_stat_weight


SPLITS = ["outer_backtest", "pandemic_stress"]
SPLIT_LABEL = {"outer_backtest": "reopening", "pandemic_stress": "pandemic_stress"}


def attach_ref_stats(rows_df: pd.DataFrame, task_map: dict, refs: dict, split: str, prefix_refs: list[str]) -> pd.DataFrame:
    """Add DM/R columns vs named refs for rows in this split."""
    out = rows_df.copy()
    part_idx = out.index[out.split_name == split]
    for ref_name in prefix_refs:
        ref_t = refs[ref_name]
        for i in part_idx:
            sid = out.at[i, "scheme_id"]
            key = (split, sid)
            if key not in task_map:
                continue
            t = task_map[key]
            for metric, col in [("MASE", "mase"), ("MAPE", "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
                r, lo, hi = ratio_ci_vs(t, ref_t, col, split)
                out.at[i, f"R_{metric}_vs_{ref_name}"] = r
                out.at[i, f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                out.at[i, f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
            dm, p, st = pairwise_dm(t, ref_t, "mase")
            out.at[i, f"DM_MASE_vs_{ref_name}"] = dm
            out.at[i, f"p_MASE_vs_{ref_name}"] = p
            out.at[i, f"sig_MASE_vs_{ref_name}"] = st
            dm2, p2, st2 = pairwise_dm(t, ref_t, "mae")
            out.at[i, f"DM_MAE_vs_{ref_name}"] = dm2
            out.at[i, f"p_MAE_vs_{ref_name}"] = p2
            out.at[i, f"sig_MAE_vs_{ref_name}"] = st2
    return out


def rebuild_task_maps(task_all: pd.DataFrame, days: pd.DataFrame, holiday_dates: set):
    """Rebuild sealed scheme tasks needed for DM attachments + S16."""
    task_map = {}
    decisions_list = []
    cache = JudgmentCache(CACHE / "llm_judgments_fill.jsonl")

    for split in SPLITS:
        log(f"[fill] rebuild tasks {split}")
        wide = build_prediction_cube(days, split, POOL)
        single = task_all[task_all.split_name == split]
        ets = single[single.config_id == "ets"]
        gran = single[single.config_id == "granite_patchtst_fm"]
        eq = precompute_equal_combo_tasks(wide, [f"C{i}" for i in range(1, 10)] + ["B_3FM", "B_5Pool"])

        # singles + equal combos
        for m in POOL:
            t = single[single.config_id == m].copy()
            t["scheme"] = m
            task_map[(split, m)] = t
        for cid, t in eq.items():
            task_map[(split, cid)] = t
        t_sw, _ = replay_fixed_stat_weight(wide, single, "B_3FM", split)
        task_map[(split, "B_3FM_StatWeight")] = t_sw
        for cid in ["C2", "C7"]:
            t, _ = replay_fixed_stat_weight(wide, single, cid, split)
            task_map[(split, f"{cid}_StatWeight")] = t
            task_map[(split, f"{cid}_Equal")] = eq[cid]
        task_map[(split, "B_3FM_Equal")] = eq["B_3FM"]

        # dynamic schemes (same set as main run)
        specs = [
            ("G1_Stat_Stat", "stat", "stat", 0.30, "mode1"),
            ("G1_Stat_Stat_Mode2", "stat", "stat", 0.30, "mode2"),
            ("G2_LLM_Stat_eta0.3", "llm", "stat", 0.30, "mode1"),
            ("G2_LLM_Stat_Mode2", "llm", "stat", 0.30, "mode2"),
            ("G3_Stat_LLM_eta0.3", "stat", "llm", 0.30, "mode1"),
            ("G3_Stat_LLM_eta0.5", "stat", "llm", 0.50, "mode1"),
            ("G3_Stat_LLM_eta1.0", "stat", "llm", 1.00, "mode1"),
            ("G4_LLM_LLM_eta0.3", "llm", "llm", 0.30, "mode1"),
            ("G4_LLM_LLM_eta0.5", "llm", "llm", 0.50, "mode1"),
            ("G4_LLM_LLM_eta1.0", "llm", "llm", 1.00, "mode1"),
            ("G4_LLM_LLM_Mode2_eta0.3", "llm", "llm", 0.30, "mode2"),
        ]
        for name, sel, wt, eta, mode in specs:
            t, dec, meta = replay_dynamic(
                wide, single, {k: eq[k] for k in [f"C{i}" for i in range(1, 10)]},
                split, sel, wt, eta=eta, holiday_dates=holiday_dates,
                scheme_name=name, mode=mode, cache=cache,
            )
            task_map[(split, name)] = t
            decisions_list.append(dec)
            (CACHE / "metas_fill.json").write_text("", encoding="utf-8")  # placeholder touch

        task_map[(split, "_ets")] = ets
        task_map[(split, "_granite")] = gran

    decisions = pd.concat(decisions_list, ignore_index=True)
    decisions.to_parquet(CACHE / "all_decisions_fill.parquet", index=False)
    return task_map, decisions


def enrich_s15(s15: pd.DataFrame, task_map: dict) -> pd.DataFrame:
    out = s15.copy()
    for split in SPLITS:
        refs = {
            "ETS": task_map[(split, "_ets")],
            "Granite": task_map[(split, "_granite")],
            "G1": task_map[(split, "G1_Stat_Stat")],
            "B3FM": task_map[(split, "B_3FM_Equal")],
        }
        # ensure task_map has each scheme
        for i in out.index[out.split_name == split]:
            sid = out.at[i, "scheme_id"]
            if (split, sid) not in task_map:
                continue
            t = task_map[(split, sid)]
            for ref_name, ref_t in refs.items():
                for metric, col in [("MASE", "mase"), ("MAPE", "mape_pct"), ("MAE", "mae"), ("RMSE", "rmse")]:
                    r, lo, hi = ratio_ci_vs(t, ref_t, col, split)
                    out.at[i, f"R_{metric}_vs_{ref_name}"] = r
                    out.at[i, f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                    out.at[i, f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
                for mcol, tag in [("mase", "MASE"), ("mae", "MAE")]:
                    dm, p, st = pairwise_dm(t, ref_t, mcol)
                    out.at[i, f"DM_{tag}_vs_{ref_name}"] = dm
                    out.at[i, f"p_{tag}_vs_{ref_name}"] = p
                    out.at[i, f"sig_{tag}_vs_{ref_name}"] = st
            # refresh point metrics from rebuilt tasks for consistency
            s = summarize_task(t)
            for k, v in s.items():
                out.at[i, k] = v
            ci = bootstrap_levels(t, split)
            for k, v in ci.items():
                out.at[i, k] = v
    return out


def enrich_s14(s14: pd.DataFrame, task_map: dict) -> pd.DataFrame:
    out = s14.copy()
    for split in SPLITS:
        g1 = task_map[(split, "G1_Stat_Stat")]
        ets = task_map[(split, "_ets")]
        gran = task_map[(split, "_granite")]
        b3 = task_map[(split, "B_3FM")]
        for i in out.index[out.split_name == split]:
            sid = out.at[i, "scheme_id"]
            # map group means skip
            if str(sid).startswith("GROUP_"):
                continue
            key = (split, sid)
            if key not in task_map:
                # StatWeight naming
                if sid.endswith("_StatWeight") and key not in task_map:
                    continue
                continue
            t = task_map[key]
            for ref_name, ref_t in [("G1", g1), ("ETS", ets), ("Granite", gran), ("B3FM", b3)]:
                for metric, col in [("MASE", "mase"), ("MAE", "mae")]:
                    r, lo, hi = ratio_ci_vs(t, ref_t, col, split)
                    out.at[i, f"R_{metric}_vs_{ref_name}"] = r
                    out.at[i, f"R_{metric}_vs_{ref_name}_ci95_low"] = lo
                    out.at[i, f"R_{metric}_vs_{ref_name}_ci95_high"] = hi
                    dm, p, st = pairwise_dm(t, ref_t, col)
                    out.at[i, f"DM_{metric}_vs_{ref_name}"] = dm
                    out.at[i, f"p_{metric}_vs_{ref_name}"] = p
                    out.at[i, f"sig_{metric}_vs_{ref_name}"] = st
    return out


def build_s16_complete(decisions: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Market-group × holiday-window × weekday strata + activation rates."""
    from common import MARKET_GROUP

    dec = decisions.copy()
    dec["origin_date"] = pd.to_datetime(dec["origin_date"])
    dec["market_group"] = dec.market.map(lambda m: MARKET_GROUP.get(m, "Small"))
    # holiday window stratum from stored holiday_days
    dec["holiday_window"] = np.where(dec.holiday_days > 0, "has_official_holiday", "ordinary_non_holiday")

    # weekday vs weekend dominance in the 28-day target window (from sealed day panel)
    base = days[(days.config_id == "ets")][["split_name", "market", "origin_date", "target_date"]].copy()
    base["origin_date"] = pd.to_datetime(base["origin_date"])
    base["target_date"] = pd.to_datetime(base["target_date"])
    base["is_weekend"] = base["target_date"].dt.dayofweek >= 5
    wk = (
        base.groupby(["split_name", "market", "origin_date"], sort=False)
        .agg(weekend_share=("is_weekend", "mean"))
        .reset_index()
    )
    wk["weekend_window"] = np.where(wk.weekend_share >= 0.5, "weekend_heavy", "weekday_heavy")
    dec = dec.merge(wk[["split_name", "market", "origin_date", "weekend_share", "weekend_window"]],
                    on=["split_name", "market", "origin_date"], how="left")

    rows = []
    focus = [s for s in dec.scheme.unique() if s.startswith("G1_") or s.startswith("G2_") or s.startswith("G4_")]
    # also include G3 for completeness
    focus = sorted(dec.scheme.unique())

    for scheme in focus:
        part0 = dec[dec.scheme == scheme]
        for split, part_s in part0.groupby("split_name"):
            for stratum_name, col, values in [
                ("market_group", "market_group", ["Large", "Medium", "Small"]),
                ("holiday_window", "holiday_window", ["has_official_holiday", "ordinary_non_holiday"]),
                ("weekend_window", "weekend_window", ["weekday_heavy", "weekend_heavy"]),
            ]:
                for val in values:
                    part = part_s[part_s[col] == val]
                    if part.empty:
                        continue
                    rows.append({
                        "split_name": split,
                        "period": SPLIT_LABEL[split],
                        "scheme_id": scheme,
                        "select_mode": "llm" if "LLM" in scheme and "Stat_LLM" not in scheme and not scheme.startswith("G3") else (
                            "llm" if scheme.startswith("G2") or scheme.startswith("G4") else "stat" if scheme.startswith("G1") or scheme.startswith("G3") else "other"
                        ),
                        "stratum_type": stratum_name,
                        "stratum_value": val,
                        "n_origins": int(len(part)),
                        "share_2FM1LW": float((part.structure == "2FM+1LW").mean()),
                        "share_1FM2LW": float((part.structure == "1FM+2LW").mean()),
                        "cold_start_share": float(part.cold_start.mean()),
                        "fallback_share": float(part.fallback.mean()),
                        "mean_holiday_days": float(part.holiday_days.mean()),
                    })
            # ALL + activation
            rows.append({
                "split_name": split,
                "period": SPLIT_LABEL[split],
                "scheme_id": scheme,
                "stratum_type": "ALL",
                "stratum_value": "ALL",
                "n_origins": int(len(part_s)),
                "share_2FM1LW": float((part_s.structure == "2FM+1LW").mean()),
                "share_1FM2LW": float((part_s.structure == "1FM+2LW").mean()),
                "cold_start_share": float(part_s.cold_start.mean()),
                "fallback_share": float(part_s.fallback.mean()),
                "mean_holiday_days": float(part_s.holiday_days.mean()),
            })
            for m in POOL:
                active = (part_s.member_0 == m) | (part_s.member_1 == m) | (part_s.member_2 == m)
                wvals = []
                for r in part_s.itertuples(index=False):
                    for j, mem in enumerate([r.member_0, r.member_1, r.member_2]):
                        if mem == m:
                            wvals.append([r.w0, r.w1, r.w2][j])
                rows.append({
                    "split_name": split,
                    "period": SPLIT_LABEL[split],
                    "scheme_id": scheme,
                    "stratum_type": "activation_ALL",
                    "stratum_value": m,
                    "model": m,
                    "activation_rate": float(active.mean()),
                    "mean_weight_when_active": float(np.mean(wvals)) if wvals else 0.0,
                    "n_origins": int(len(part_s)),
                })
            # activation by holiday window
            for val in ["has_official_holiday", "ordinary_non_holiday"]:
                part = part_s[part_s.holiday_window == val]
                if part.empty:
                    continue
                for m in POOL:
                    active = (part.member_0 == m) | (part.member_1 == m) | (part.member_2 == m)
                    wvals = []
                    for r in part.itertuples(index=False):
                        for j, mem in enumerate([r.member_0, r.member_1, r.member_2]):
                            if mem == m:
                                wvals.append([r.w0, r.w1, r.w2][j])
                    rows.append({
                        "split_name": split,
                        "period": SPLIT_LABEL[split],
                        "scheme_id": scheme,
                        "stratum_type": "activation_by_holiday_window",
                        "stratum_value": val,
                        "model": m,
                        "activation_rate": float(active.mean()),
                        "mean_weight_when_active": float(np.mean(wvals)) if wvals else 0.0,
                        "n_origins": int(len(part)),
                    })
    return pd.DataFrame(rows)


def write_manifest():
    rows = [
        {
            "item": "warm_start_inner_selection_prior",
            "status": "not_run",
            "note": "RQ1/RQ2 sealed task_metrics only contain outer folds (O*); no per-market inner-selection completed-task MAE library for M5 pool was found under track1/results or RQ*_outputs. Main replay keeps cold-start = equal-weight C2.",
        },
        {
            "item": "real_Qwen3_4B_llm_select_weight",
            "status": "not_run_gpu_unavailable",
            "note": "nvidia driver unavailable on host; G2–G4 use information-set-matched llm_proxy with JSON cache. Swap-in path: RQ3_outputs/cache/llm_judgments.jsonl",
        },
        {
            "item": "S16_holiday_weekday_strata",
            "status": "filled",
            "note": "Added holiday_window and weekend_window strata plus activation_by_holiday_window",
        },
        {
            "item": "S15_DM_vs_ETS",
            "status": "filled",
            "note": "Added R/DM for MASE&MAE vs ETS/Granite/G1/B3FM",
        },
        {
            "item": "S14_DM_vs_G1",
            "status": "filled",
            "note": "Added R/DM vs G1_Stat_Stat for fixed schemes",
        },
    ]
    pd.DataFrame(rows).to_csv(LOGS / "missing_items_manifest_rq3.csv", index=False)
    return rows


def enrich_s18(s18: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    out = s18.copy()
    out["llm_backend"] = np.where(
        out.scheme_id.astype(str).str.contains("LLM|Qwen"),
        np.where(out.scheme_id.eq("Qwen3-4B_direct"), "qwen3_4b_direct_forecast_rq1", "llm_proxy_info_set_matched"),
        "none",
    )
    out["llm_backend_note"] = out["llm_backend"].map({
        "none": "statistical only",
        "llm_proxy_info_set_matched": "GPU unavailable; proxy uses same H_it features as specified LLM input set",
        "qwen3_4b_direct_forecast_rq1": "RQ1 sealed direct forecast baseline (not constrained ensemble judge)",
    })
    # refresh reliability from decisions
    for i, row in out.iterrows():
        sid = row.scheme_id
        part = decisions[(decisions.scheme == sid) & (decisions.split_name == "outer_backtest")]
        if len(part):
            out.at[i, "json_fail_or_fallback_rate"] = float(part.fallback.mean())
            out.at[i, "fallback_rate"] = float(part.fallback.mean())
            out.at[i, "cold_start_rate"] = float(part.cold_start.mean())
            out.at[i, "n_decisions"] = int(len(part))
            out.at[i, "weight_sum_max_abs_deviation"] = float((part.w0 + part.w1 + part.w2 - 1).abs().max())
    return out


def update_summary(manifest_rows):
    path = OUT / "4.3_RQ3_实验数据核对汇总稿.md"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    block = ["\n\n---\n\n# 缺口补全记录（fill_gaps）\n\n"]
    block.append("补全脚本：`RQ3/fill_gaps.py`。\n\n")
    block.append("| item | status | note |\n|------|--------|------|\n")
    for r in manifest_rows:
        block.append(f"| {r['item']} | {r['status']} | {r['note']} |\n")
    block.append("\n更新表：S14（+vs G1）、S15（+vs ETS）、S16（假日/周末分层）、S18（llm_backend）。\n")
    # append key S16 holiday peek
    s16 = pd.read_csv(TABLES / "table_S16_subset_switching_and_market_heterogeneity.csv")
    peek = s16[(s16.stratum_type == "holiday_window") & (s16.scheme_id.isin(["G1_Stat_Stat", "G2_LLM_Stat_eta0.3", "G4_LLM_LLM_eta0.3"])) & (s16.split_name == "outer_backtest")]
    if len(peek):
        block.append("\n## S16 假日窗口选模比例（重开期）\n\n")
        try:
            block.append(peek[["scheme_id", "stratum_value", "n_origins", "share_2FM1LW", "share_1FM2LW"]].to_markdown(index=False))
        except Exception:
            block.append("```\n" + peek[["scheme_id", "stratum_value", "n_origins", "share_2FM1LW", "share_1FM2LW"]].to_string(index=False) + "\n```")
        block.append("\n")
    s15 = pd.read_csv(TABLES / "table_S15_dynamic_selection_and_weighting_ablation.csv")
    cols = [c for c in ["scheme_id", "MASE", "R_MASE_vs_ETS", "DM_MASE_vs_ETS", "p_MASE_vs_ETS", "R_MASE_vs_G1", "DM_MASE_vs_G1"] if c in s15.columns]
    peek15 = s15[(s15.split_name == "outer_backtest") & (s15.scheme_id.str.startswith("G"))][cols]
    block.append("\n## S15 相对 ETS / G1（重开期动态方案）\n\n")
    try:
        block.append(peek15.to_markdown(index=False))
    except Exception:
        block.append("```\n" + peek15.to_string(index=False) + "\n```")
    block.append("\n")
    path.write_text(old + "".join(block), encoding="utf-8")


def main():
    ensure_dirs()
    log("=== fill_gaps start ===")
    days = load_day_panel()
    task = load_task_panel()
    holiday_dates = load_holiday_flags()
    s14 = pd.read_csv(TABLES / "table_S14_fixed_combinations_benchmark.csv")
    s15 = pd.read_csv(TABLES / "table_S15_dynamic_selection_and_weighting_ablation.csv")
    s18 = pd.read_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv")

    task_map, decisions = rebuild_task_maps(task, days, holiday_dates)
    # persist task summaries
    rows = []
    for (split, sid), t in task_map.items():
        if sid.startswith("_"):
            continue
        s = summarize_task(t)
        rows.append({"split_name": split, "scheme_id": sid, **s})
    pd.DataFrame(rows).to_csv(CACHE / "scheme_task_summaries_fill.csv", index=False)

    log("=== enrich S15 ===")
    s15n = enrich_s15(s15, task_map)
    s15n.to_csv(TABLES / "table_S15_dynamic_selection_and_weighting_ablation.csv", index=False)

    log("=== enrich S14 ===")
    s14n = enrich_s14(s14, task_map)
    s14n.to_csv(TABLES / "table_S14_fixed_combinations_benchmark.csv", index=False)

    log("=== rebuild S16 ===")
    # prefer fill decisions (complete); fallback merge with old
    s16n = build_s16_complete(decisions, days)
    s16n.to_csv(TABLES / "table_S16_subset_switching_and_market_heterogeneity.csv", index=False)
    # also keep a wide holiday pivot helper
    hol = s16n[s16n.stratum_type == "holiday_window"].copy()
    hol.to_csv(TABLES / "table_S16_holiday_window_selection_shares.csv", index=False)

    log("=== enrich S18 ===")
    s18n = enrich_s18(s18, decisions)
    s18n.to_csv(TABLES / "table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv", index=False)

    manifest = write_manifest()
    update_summary(manifest)

    # rewrite audit
    audit = []
    audit.append("# RQ3 缺口补全后核对\n\n")
    audit.append(f"- S15 has DM_MASE_vs_ETS: {'DM_MASE_vs_ETS' in s15n.columns}\n")
    audit.append(f"- S14 has DM_MASE_vs_G1: {'DM_MASE_vs_G1' in s14n.columns}\n")
    audit.append(f"- S16 stratum types: {sorted(s16n.stratum_type.dropna().unique())}\n")
    audit.append(f"- S16 rows: {len(s16n)}\n")
    audit.append(f"- S18 llm_backend col: {'llm_backend' in s18n.columns}\n")
    (LOGS / "rq3_fill_gaps_audit.md").write_text("".join(audit), encoding="utf-8")
    log("=== fill_gaps DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
