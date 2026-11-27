#!/usr/bin/env python3
"""Set B preselection audit for fixed equal-weight combinations.

Read-only vs existing Track1 / RQ sealed artifacts. Writes only under
outputs/set_b_preselection_audit/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIGHT_B = REPO / "track1/results/light_models/recovery_tuning"
FOUND_B = OUT / "extracted_foundation_b/foundation_b_default"
TARGETS = REPO / "shared/data/targets.parquet"
ARCHIVED_LIGHT = REPO / "track1/evaluation/light_models_b/candidate_summary.csv"
S14 = REPO / "RQ3_outputs/tables/table_S14_fixed_combinations_benchmark.csv"
S15 = REPO / "RQ3_outputs/tables/table_S15_dynamic_selection_and_weighting_ablation.csv"

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
}
SINGLES = ["granite_patchtst_fm", "timesfm_3", "chronos_2", "NL01", "GL03"]
LABEL = {
    "granite_patchtst_fm": "Granite",
    "timesfm_3": "TimesFM 3.0",
    "chronos_2": "Chronos-2",
    "NL01": "Global NLinear",
    "GL03": "LightGBM (Global)",
}


def load_light(model_dir: Path, candidate: str, seed: int | None = 42) -> pd.DataFrame:
    if seed is None:
        files = sorted(model_dir.rglob("predictions.parquet"))
    else:
        files = sorted((model_dir / f"seed{seed}").glob("*/predictions.parquet"))
        if not files:
            files = sorted(model_dir.rglob("predictions.parquet"))
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame = frame[frame.destination == "Macau"].copy()
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["config_id"] = candidate
    out["origin_date"] = pd.to_datetime(out["origin_date"])
    out["target_date"] = pd.to_datetime(out["target_date"])
    return out


def load_foundation(model_id: str) -> pd.DataFrame:
    files = sorted(FOUND_B.glob(f"*/{model_id}/predictions.parquet"))
    if not files:
        raise FileNotFoundError(f"no foundation B predictions for {model_id} under {FOUND_B}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if "destination" in frame.columns:
            frame = frame[frame.destination == "Macau"].copy()
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["config_id"] = model_id
    out["origin_date"] = pd.to_datetime(out["origin_date"])
    if "target_date" in out.columns:
        out["target_date"] = pd.to_datetime(out["target_date"])
    pred_col = "prediction" if "prediction" in out.columns else "prediction_raw"
    out["prediction"] = out[pred_col].astype(float).clip(lower=0.0)
    return out


def attach_targets(pred: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["task_id", "horizon_day"]
    merged = pred.copy()
    if "actual" not in merged.columns:
        cols = ["task_id", "horizon_day", "actual"]
        if "q_all" in targets.columns:
            cols.append("q_all")
        merged = merged.merge(targets[cols], on=keys, how="left", validate="many_to_one")
    if merged.actual.isna().any():
        raise RuntimeError(f"unmatched predictions: {int(merged.actual.isna().sum())}")
    if "mase7_scale" not in merged.columns:
        if "q_all" in merged.columns:
            merged["mase7_scale"] = merged["q_all"].astype(float)
        else:
            raise RuntimeError("no MASE scale column")
    err = (merged.prediction.astype(float) - merged.actual.astype(float)).abs()
    merged["ae"] = err
    merged["se"] = (merged.prediction.astype(float) - merged.actual.astype(float)) ** 2
    merged["ape"] = np.where(merged.actual.astype(float) > 0, err / merged.actual.astype(float), np.nan)
    task = (
        merged.groupby(["config_id", "task_id", "market", "fold_id", "origin_date"], sort=False)
        .agg(
            mae=("ae", "mean"),
            rmse=("se", lambda s: float(np.sqrt(s.mean()))),
            mape=("ape", "mean"),
            scale=("mase7_scale", "first"),
            n=("ae", "size"),
        )
        .reset_index()
    )
    task["mase"] = np.where(task.scale > 0, task.mae / task.scale, np.nan)
    task["mape_pct"] = 100.0 * task.mape
    return merged, task


def market_then_equal(task: pd.DataFrame, metric: str) -> float:
    part = task[np.isfinite(task[metric])]
    return float(part.groupby("market")[metric].mean().mean())


def summarize(task: pd.DataFrame) -> dict:
    return {
        "MASE": market_then_equal(task, "mase"),
        "MAE": market_then_equal(task, "mae"),
        "RMSE": market_then_equal(task, "rmse"),
        "MAPE": market_then_equal(task, "mape_pct"),
        "n_tasks": int(task.task_id.nunique()),
        "n_markets": int(task.market.nunique()),
        "n_origins": int(task.origin_date.nunique()),
    }


def equal_combo(day_map: dict[str, pd.DataFrame], members: list[str]) -> pd.DataFrame:
    base = None
    for m in members:
        part = day_map[m][["task_id", "market", "fold_id", "origin_date", "horizon_day",
                           "target_date", "actual", "mase7_scale", "prediction"]].copy()
        part = part.rename(columns={"prediction": m})
        base = part if base is None else base.merge(
            part[["task_id", "horizon_day", m]], on=["task_id", "horizon_day"], how="inner"
        )
    base["prediction"] = base[members].mean(axis=1)
    base["config_id"] = "+".join(members)
    return base


def verify_source(nl01: pd.DataFrame, gl03: pd.DataFrame) -> dict:
    report = {
        "source": "recovered_from_server",
        "nl01_path": str(LIGHT_B / "global_linear/NL01"),
        "gl03_path": str(LIGHT_B / "global_lightgbm/GL03"),
        "nl01_rows": int(len(nl01)),
        "gl03_rows": int(len(gl03)),
        "nl01_tasks": int(nl01.task_id.nunique()),
        "gl03_tasks": int(gl03.task_id.nunique()),
        "nl01_target_min": str(nl01.target_date.min().date()),
        "nl01_target_max": str(nl01.target_date.max().date()),
        "gl03_target_min": str(gl03.target_date.min().date()),
        "gl03_target_max": str(gl03.target_date.max().date()),
        "excluded": [
            "LGB03/local LightGBM (not GL03)",
            "R3 variants",
            "Set O / Set A light_models splits",
            "GL-noCal / GL-local ablations",
        ],
        "phase2_rerun": False,
    }
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(TARGETS)
    targets = targets[(targets.split_name == "recovery_tuning") & (targets.destination == "Macau")].copy()
    targets["origin_date"] = pd.to_datetime(targets["origin_date"])
    targets["target_date"] = pd.to_datetime(targets["target_date"])

    # Prefer seed42 for combination consistency with outer RQ panels.
    nl01 = load_light(LIGHT_B / "global_linear/NL01", "NL01", seed=42)
    gl03 = load_light(LIGHT_B / "global_lightgbm/GL03", "GL03", seed=42)
    source = verify_source(nl01, gl03)
    (OUT / "source_report.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    # Export standardized daily tables (9072 rows each)
    for name, frame in [("nl01_set_b_daily.parquet", nl01), ("gl03_set_b_daily.parquet", gl03)]:
        keep = [
            "task_id", "split_name", "fold_id", "destination", "market", "origin_date",
            "history_length", "target_date", "horizon_day", "model", "candidate_id", "seed",
            "prediction_raw", "prediction", "mase7_scale",
        ]
        frame[keep].to_parquet(OUT / name, index=False)

    # Reproduce archived single-model B scores (NL01 seed-average for archive match)
    nl01_all = load_light(LIGHT_B / "global_linear/NL01", "NL01", seed=None)
    nl01_all = nl01_all[nl01_all.candidate_id == "NL01"].copy()
    # average predictions across seeds at day level then score once
    key = ["task_id", "market", "fold_id", "origin_date", "horizon_day", "target_date", "mase7_scale"]
    nl01_avg = (
        nl01_all.groupby(key, sort=False, as_index=False)
        .agg(prediction=("prediction", "mean"), prediction_raw=("prediction_raw", "mean"), seed=("seed", "nunique"))
    )
    nl01_avg["config_id"] = "NL01"
    nl01_avg["split_name"] = "recovery_tuning"

    _, task_nl01_s42 = attach_targets(nl01, targets)
    _, task_gl03 = attach_targets(gl03, targets)
    _, task_nl01_avg = attach_targets(nl01_avg, targets)
    sum_nl01_s42 = summarize(task_nl01_s42)
    sum_gl03 = summarize(task_gl03)
    sum_nl01_avg = summarize(task_nl01_avg)

    archived = pd.read_csv(ARCHIVED_LIGHT)
    arch_nl = archived[(archived.candidate_id == "NL01")].iloc[0]
    arch_gl = archived[(archived.candidate_id == "GL03")].iloc[0]
    repro = pd.DataFrame([
        {
            "model": "NL01",
            "scoring": "seed_average_3",
            "recomputed_MASE": sum_nl01_avg["MASE"],
            "archived_MASE": float(arch_nl.mean_MASE7),
            "delta_MASE": sum_nl01_avg["MASE"] - float(arch_nl.mean_MASE7),
            "recomputed_MAE": sum_nl01_avg["MAE"],
            "note": "matches light_models_b candidate_summary aggregation",
        },
        {
            "model": "NL01",
            "scoring": "seed42_only",
            "recomputed_MASE": sum_nl01_s42["MASE"],
            "archived_MASE": float(arch_nl.mean_MASE7),
            "delta_MASE": sum_nl01_s42["MASE"] - float(arch_nl.mean_MASE7),
            "recomputed_MAE": sum_nl01_s42["MAE"],
            "note": "seed used for equal-weight combinations below",
        },
        {
            "model": "GL03",
            "scoring": "seed42",
            "recomputed_MASE": sum_gl03["MASE"],
            "archived_MASE": float(arch_gl.mean_MASE7),
            "delta_MASE": sum_gl03["MASE"] - float(arch_gl.mean_MASE7),
            "recomputed_MAE": sum_gl03["MAE"],
            "note": "single seed in archive",
        },
    ])
    repro.to_csv(OUT / "set_b_single_model_repro_check.csv", index=False)

    # Load foundation B
    day_map = {}
    task_map = {}
    for mid in ["granite_patchtst_fm", "timesfm_3", "chronos_2"]:
        pred = load_foundation(mid)
        # align scale from light or targets: foundation preds may already have mase7_scale
        if "mase7_scale" not in pred.columns:
            scale = targets[["task_id", "horizon_day", "q_all"]].drop_duplicates(["task_id", "horizon_day"])
            # use first-horizon scale per task from light NL01
            scales = nl01[["task_id", "mase7_scale"]].drop_duplicates("task_id")
            pred = pred.merge(scales, on="task_id", how="left")
        day, task = attach_targets(pred, targets)
        day_map[mid] = day
        task_map[mid] = task
    day_map["NL01"] = attach_targets(nl01, targets)[0]
    day_map["GL03"] = attach_targets(gl03, targets)[0]
    task_map["NL01"] = task_nl01_s42
    task_map["GL03"] = task_gl03

    rows = []
    for sid in SINGLES:
        s = summarize(task_map[sid])
        rows.append({"scheme_id": sid, "display_name": LABEL[sid], "kind": "single", "members": sid, **s})
    for cid, members in COMBOS.items():
        combo_day = equal_combo(day_map, members)
        _, combo_task = attach_targets(combo_day, targets)
        s = summarize(combo_task)
        rows.append({
            "scheme_id": cid,
            "display_name": cid if cid.startswith("C") or cid == "B_3FM" else cid,
            "kind": "combo_equal",
            "members": ",".join(members),
            **s,
        })
        combo_task.to_parquet(OUT / f"task_metrics_{cid}.parquet", index=False)

    set_b = pd.DataFrame(rows)
    set_b.to_csv(OUT / "set_b_fixed_combinations.csv", index=False)

    pool_c = set_b[set_b.scheme_id.isin([f"C{i}" for i in range(1, 10)])].copy()
    all_fixed = set_b[set_b.scheme_id.isin([f"C{i}" for i in range(1, 10)] + ["B_3FM"])].copy()
    best = {
        "best_C_by_MASE": pool_c.loc[pool_c.MASE.idxmin(), ["scheme_id", "members", "MASE", "MAE", "RMSE"]].to_dict(),
        "best_C_by_MAE": pool_c.loc[pool_c.MAE.idxmin(), ["scheme_id", "members", "MASE", "MAE", "RMSE"]].to_dict(),
        "best_all_fixed_by_MASE": all_fixed.loc[all_fixed.MASE.idxmin(), ["scheme_id", "members", "MASE", "MAE", "RMSE"]].to_dict(),
        "best_all_fixed_by_MAE": all_fixed.loc[all_fixed.MAE.idxmin(), ["scheme_id", "members", "MASE", "MAE", "RMSE"]].to_dict(),
    }
    (OUT / "set_b_preselection_winners.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # Map winners to sealed Set O (S14) and dynamic G1/G2 (S15)
    s14 = pd.read_csv(S14)
    s15 = pd.read_csv(S15)
    outer14 = s14[s14.split_name == "outer_backtest"].copy()
    outer15 = s15[s15.split_name == "outer_backtest"].copy()

    def s14_row(scheme: str) -> dict:
        hit = outer14[outer14.scheme_id == scheme]
        if hit.empty:
            # B_3FM appears as B_3FM in combo_equal
            hit = outer14[outer14.scheme_id.isin([scheme, "B_3FM_equal", "B_3FM Equal"])]
        if hit.empty:
            return {"scheme_id": scheme, "MASE": np.nan, "MAE": np.nan, "RMSE": np.nan}
        r = hit.iloc[0]
        return {"scheme_id": scheme, "MASE": float(r.MASE), "MAE": float(r.MAE), "RMSE": float(r.RMSE), "n_tasks": int(r.n_tasks)}

    g1 = outer15[outer15.scheme_id == "G1_Stat_Stat"].iloc[0]
    g2 = outer15[outer15.scheme_id == "G2_LLM_Stat_eta0.3"].iloc[0]

    winners_needed = {
        "best_C_MASE": best["best_C_by_MASE"]["scheme_id"],
        "best_C_MAE": best["best_C_by_MAE"]["scheme_id"],
        "best_all_MASE": best["best_all_fixed_by_MASE"]["scheme_id"],
        "best_all_MAE": best["best_all_fixed_by_MAE"]["scheme_id"],
        "B_3FM": "B_3FM",
        "C2": "C2",
    }
    compare_rows = []
    for tag, sid in winners_needed.items():
        row = s14_row(sid)
        row["role"] = tag
        row["period"] = "Set_O_outer_backtest"
        row["source"] = "RQ3_outputs/table_S14"
        compare_rows.append(row)
    compare_rows.append({
        "role": "G1_Stat_Stat", "scheme_id": "G1_Stat_Stat", "period": "Set_O_outer_backtest",
        "MASE": float(g1.MASE), "MAE": float(g1.MAE), "RMSE": float(g1.RMSE),
        "source": "RQ3_outputs/table_S15",
    })
    compare_rows.append({
        "role": "G2_LLM_Stat_eta0.3", "scheme_id": "G2_LLM_Stat_eta0.3", "period": "Set_O_outer_backtest",
        "MASE": float(g2.MASE), "MAE": float(g2.MAE), "RMSE": float(g2.RMSE),
        "source": "RQ3_outputs/table_S15",
    })
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT / "set_o_mapped_comparison.csv", index=False)

    # Human-readable summary
    lines = []
    lines.append("# Set B preselection audit\n\n")
    lines.append("## Source\n\n")
    lines.append("- NL01 and GL03 Set B daily forecasts: **recovered on this server** "
                 "(no Phase-2 rerun).\n")
    lines.append(f"- NL01: `{source['nl01_path']}` — {source['nl01_rows']} rows, "
                 f"{source['nl01_tasks']} tasks, targets {source['nl01_target_min']}–{source['nl01_target_max']}.\n")
    lines.append(f"- GL03: `{source['gl03_path']}` — {source['gl03_rows']} rows, "
                 f"{source['gl03_tasks']} tasks, targets {source['gl03_target_min']}–{source['gl03_target_max']}.\n")
    lines.append("- Foundation Set B: extracted read-only from "
                 "`track1/artifacts/predictions/foundation_b_default/{granite,timesfm_3,chronos_2}.tar.gz`.\n")
    lines.append("- Combinations use NL01 **seed42** (same as RQ outer panels).\n\n")
    lines.append("## Single-model repro vs archived Set B scores\n\n")
    lines.append("```\n" + repro.to_string(index=False) + "\n```\n")
    lines.append("\n## Set B fixed combinations\n\n")
    show = set_b[["scheme_id", "kind", "members", "MASE", "MAE", "RMSE", "n_tasks"]].copy()
    lines.append("```\n" + show.to_string(index=False) + "\n```\n")
    lines.append("\n## Preselection winners on Set B\n\n")
    lines.append("```json\n" + json.dumps(best, indent=2) + "\n```\n\n")
    lines.append("## Mapped Set O performance (sealed S14/S15; not rerun)\n\n")
    lines.append("```\n" + compare.to_string(index=False) + "\n```\n")
    (OUT / "README.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT)
    print(show.to_string(index=False))
    print("winners", json.dumps(best, indent=2))
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
