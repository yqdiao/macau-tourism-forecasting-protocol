#!/usr/bin/env python3
"""Continue V1-V3, freeze, then score O and A. Selection uses B only."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, RUN_ID, SEEDS_LINEAR, SEED_TREE, code_hash, utc_now, write_json
from data import load_calendar, load_origins
from linear import fit_linear
from pipeline import _store, specs
from score import block_bootstrap, load_targets, macro_mase, scale_frame, task_metrics
from trees import fit_trees

B_FOLDS = [f"B{i:02d}" for i in range(1, 13)]
O_FOLDS = [f"O{i:02d}" for i in range(1, 34)]
A_FOLDS = [f"A{i:02d}" for i in range(1, 14)]
TREE_POOL = ["R3V1_L", "R3V2_T1", "R3V2_T2", "R3V2_T3", "R3V2_T4", "R3V2_T5"]
LINEAR_POOL = ["R3V1_NR", "R3V2_N2", "R3V2_N3"]


def _split(fold: str) -> str:
    if fold.startswith("B"):
        return "recovery_tuning"
    if fold.startswith("O"):
        return "outer_backtest"
    return "pandemic_stress"


def _path(model_id: str, seed: int, split: str, fold: str) -> Path:
    return OUT / "predictions" / model_id / f"seed{seed}" / split / f"{fold}.parquet"


def _seeds(spec: dict) -> tuple[int, ...]:
    return SEEDS_LINEAR if spec["family"] == "linear" else (SEED_TREE,)


def run_models(bundles, jobs: list[tuple[str, dict, list[str]]]) -> None:
    calendar = load_calendar()
    for model_id, spec, folds in jobs:
        for fold in folds:
            bundle = bundles[(_split(fold), fold)]
            for seed in _seeds(spec):
                path = _path(model_id, seed, bundle.split, fold)
                if path.exists():
                    print(json.dumps({"event": "skip", "model_id": model_id, "fold": fold, "seed": seed}), flush=True)
                    continue
                t0 = time.perf_counter()
                last_error = None
                for attempt in (1, 2):
                    try:
                        if spec["family"] == "linear":
                            result = fit_linear(bundle, spec, calendar, seed=seed)
                        elif spec["family"] == "tree":
                            result = fit_trees(bundle, spec, calendar, pooled=spec["pooled"], seed=seed)
                        else:
                            raise RuntimeError(spec["family"])
                        _store(bundle, model_id, seed, result, False)
                        print(json.dumps({"event": "done", "model_id": model_id, "fold": fold, "seed": seed,
                                          "seconds": round(time.perf_counter() - t0, 3), "attempt": attempt}), flush=True)
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        (OUT / "failures.csv").parent.mkdir(parents=True, exist_ok=True)
                        with (OUT / "failures.csv").open("a", encoding="utf-8") as stream:
                            stream.write(f"{model_id},{seed},{fold},attempt{attempt},{type(exc).__name__},{exc}\n")
                if last_error is not None:
                    print(json.dumps({"event": "failed", "model_id": model_id, "fold": fold, "seed": seed,
                                      "error": str(last_error), "trace": traceback.format_exc()[-1500:]}), flush=True)


def _read_split(model_ids: list[str], split: str) -> pd.DataFrame:
    frames = []
    for model_id in model_ids:
        root = OUT / "predictions" / model_id
        if not root.exists():
            continue
        for path in root.glob(f"seed*/{split}/*.parquet"):
            if path.name.endswith("_smoke.parquet"):
                continue
            frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError(f"no predictions for {model_ids} {split}")
    return pd.concat(frames, ignore_index=True)


def score_models(bundles, model_ids: list[str], split: str) -> pd.DataFrame:
    predictions = _read_split(model_ids, split)
    scales = scale_frame({k: v for k, v in bundles.items() if k[0] == split})
    return task_metrics(predictions, load_targets(), scales)


def leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, part in metrics.groupby("model_id"):
        rows.append({"model_id": model_id, "MASE7_common": macro_mase(part),
                     "MAE": float(part.groupby(["market", "task_id"])["mae"].mean().mean()),
                     "native_rate": float(part.groupby(["market", "task_id"])["native"].mean().mean()),
                     "tasks": int(part.groupby(["market", "task_id"]).ngroups)})
    board = pd.DataFrame(rows).sort_values(["MASE7_common", "model_id"])
    return board


def choose(board: pd.DataFrame, pool: list[str]) -> str:
    part = board[board["model_id"].isin(pool)].copy()
    part = part.sort_values(["MASE7_common", "model_id"])
    if abs(part.iloc[0]["MASE7_common"] - part.iloc[1]["MASE7_common"]) <= 1e-6 if len(part) > 1 else False:
        part = part.sort_values("model_id")
    return str(part.iloc[0]["model_id"])


def alias_copy(source: str, dest: str, split: str) -> None:
    for path in (OUT / "predictions" / source).glob(f"seed*/{split}/*.parquet"):
        frame = pd.read_parquet(path)
        frame["model_id"] = dest
        seed = path.parts[-3]
        fold = path.stem
        target = OUT / "predictions" / dest / seed / split / f"{fold}.parquet"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)


def mix_predictions(local_id: str, global_id: str, split: str) -> None:
    local = _read_split([local_id], split)
    glob = _read_split([global_id], split)
    keys = ["task_id", "horizon_day", "seed"]
    # stored seeds differ; mix the single tree seed by broadcasting seed 42
    local = local[local["seed"] == SEED_TREE]
    glob = glob[glob["seed"] == SEED_TREE]
    merged = local.merge(glob, on=["task_id", "horizon_day"], suffixes=("_l", "_g"))
    for lam, name in ((0.0, "R3V3_MIX_0"), (0.5, "R3V3_MIX_05"), (1.0, "R3V3_MIX_1")):
        out = local.copy()
        aligned = merged.sort_values(["task_id", "horizon_day"])
        pred = lam * aligned["prediction_l"].to_numpy() + (1 - lam) * aligned["prediction_g"].to_numpy()
        raw = lam * aligned["prediction_raw_l"].to_numpy() + (1 - lam) * aligned["prediction_raw_g"].to_numpy()
        out = out.sort_values(["task_id", "horizon_day"])
        if len(out) != len(aligned):
            raise RuntimeError("mix row mismatch")
        out["prediction"] = pred
        out["prediction_raw"] = raw
        out["model_id"] = name
        out["status"] = np.where((aligned["status_l"].to_numpy() == "native") & (aligned["status_g"].to_numpy() == "native"), "native", "fallback:member")
        dest = OUT / "predictions" / name / "seed42" / split
        for fold, part in out.groupby("fold"):
            path = dest / f"{fold}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)


def main() -> None:
    bundles = load_origins()
    catalog = specs()
    run_models(bundles, [(name, catalog[name], B_FOLDS) for name in
                         ["R3V1_NP", "R3V1_NR", "R3V1_G", "R3V1_L", "R3V2_T1", "R3V2_T2", "R3V2_T3",
                          "R3V2_T4", "R3V2_T5", "R3V2_N2", "R3V2_N3"]])
    b_metrics = score_models(bundles, list(catalog), "recovery_tuning")
    board = leaderboard(b_metrics)
    board.to_csv(OUT / "candidate_scores_B.csv", index=False)
    tree_best = choose(board, TREE_POOL)
    linear_best = choose(board, LINEAR_POOL)
    write_json(OUT / "selection_V2_B.json", {"tree_best": tree_best, "linear_best": linear_best,
                                            "at_utc": utc_now(), "code_hash": code_hash()})
    parent = catalog[tree_best]
    v3 = {
        "R3V3_L_INF": {**parent, "pooled": False, "half_life": "inf"},
        "R3V3_G_INF": {**parent, "pooled": True, "half_life": "inf"},
        "R3V3_L_365": {**parent, "pooled": False, "half_life": 365},
        "R3V3_G_365": {**parent, "pooled": True, "half_life": 365},
        "R3V3_L_180": {**parent, "pooled": False, "half_life": 180},
        "R3V3_G_180": {**parent, "pooled": True, "half_life": 180},
    }
    alias_copy(tree_best, "R3V3_L_INF", "recovery_tuning")
    if parent["target"] == "log" and not parent["l1"] and not parent["annual"] and not parent["phase"]:
        alias_copy("R3V1_G", "R3V3_G_INF", "recovery_tuning")
    train_v3 = [(name, spec, B_FOLDS) for name, spec in v3.items()
                if not (OUT / "predictions" / name / "seed42" / "recovery_tuning").exists()]
    # existence of directory is not enough; run_models skips per file
    run_models(bundles, [(n, s, B_FOLDS) for n, s in v3.items() if n not in ("R3V3_L_INF",) and not (
        n == "R3V3_G_INF" and parent["target"] == "log" and not parent["l1"] and not parent["annual"] and not parent["phase"])])
    member_ids = list(v3)
    member_metrics = score_models(bundles, member_ids, "recovery_tuning")
    member_board = leaderboard(member_metrics)
    local_best = choose(member_board, ["R3V3_L_INF", "R3V3_L_365", "R3V3_L_180"])
    global_best = choose(member_board, ["R3V3_G_INF", "R3V3_G_365", "R3V3_G_180"])
    mix_predictions(local_best, global_best, "recovery_tuning")
    mix_metrics = score_models(bundles, ["R3V3_MIX_0", "R3V3_MIX_05", "R3V3_MIX_1"], "recovery_tuning")
    mix_board = leaderboard(mix_metrics)
    mix_best = choose(mix_board, ["R3V3_MIX_0", "R3V3_MIX_05", "R3V3_MIX_1"])
    write_json(OUT / "selection_V3_B.json", {
        "local_best": local_best, "global_best": global_best, "mix_best": mix_best,
        "tree_best": tree_best, "linear_best": linear_best, "at_utc": utc_now(),
    })
    outer_models = {
        "R3V1_NP": catalog["R3V1_NP"], "R3V1_NR": catalog["R3V1_NR"],
        "R3V1_G": catalog["R3V1_G"], "R3V1_L": catalog["R3V1_L"],
        tree_best: catalog[tree_best], linear_best: catalog[linear_best],
        local_best: v3[local_best], global_best: v3[global_best],
    }
    frozen = {"run_id": RUN_ID, "at_utc": utc_now(), "code_hash": code_hash(),
              "tree_best": tree_best, "linear_best": linear_best, "local_best": local_best,
              "global_best": global_best, "mix_best": mix_best, "outer_models": list(outer_models),
              "exploratory": True}
    write_json(OUT / "frozen_R3.json", frozen)
    alias_ids = {"R3V3_L_INF"}
    if parent["target"] == "log" and not parent["l1"] and not parent["annual"] and not parent["phase"]:
        alias_ids.add("R3V3_G_INF")
    run_models(bundles, [(name, spec, O_FOLDS + A_FOLDS) for name, spec in outer_models.items() if name not in alias_ids])
    for split in ("outer_backtest", "pandemic_stress"):
        alias_copy(tree_best, "R3V3_L_INF", split)
        if "R3V3_G_INF" in alias_ids:
            alias_copy("R3V1_G", "R3V3_G_INF", split)
        mix_predictions(local_best, global_best, split)
    o_ids = list(outer_models) + ["R3V3_MIX_0", "R3V3_MIX_05", "R3V3_MIX_1"]
    o_metrics = score_models(bundles, o_ids, "outer_backtest")
    a_metrics = score_models(bundles, o_ids, "pandemic_stress")
    leaderboard(o_metrics).to_csv(OUT / "candidate_scores_O.csv", index=False)
    leaderboard(a_metrics).to_csv(OUT / "candidate_scores_A.csv", index=False)
    o_metrics.to_parquet(OUT / "task_metrics_O_R3.parquet", index=False)
    a_metrics.to_parquet(OUT / "task_metrics_A_R3.parquet", index=False)
    b_metrics.to_parquet(OUT / "task_metrics_B_R3.parquet", index=False)
    write_json(OUT / "V1_COMPLETE.json", {"status": "completed", "at_utc": utc_now()})
    o_board = leaderboard(o_metrics)
    a_board = leaderboard(a_metrics)
    lines = [
        "# R3 completion",
        "",
        f"Run `{RUN_ID}` finished at {utc_now()}. This round is exploratory: the old outer results were already known, and the old NL01 champion is unchanged.",
        "",
        "## B development MASE",
        "",
        board.to_csv(index=False),
        "",
        "## Outer MASE",
        "",
        o_board.to_csv(index=False),
        "",
        "## Stress MASE",
        "",
        a_board.to_csv(index=False),
        "",
        f"B selections: tree `{tree_best}`, linear `{linear_best}`, local `{local_best}`, global `{global_best}`, mix `{mix_best}`.",
        "",
        "Published full-history reference summaries, not paired intervals: Granite 1.094, TimesFM 1.117, Chronos-2 1.152, ETS 1.639, seasonal naive 1.883. Those models were not refit inside R3.",
        "",
        "Original outer MASE reproduced before training: NL01 1.627363, GL03 1.337990, LL01 1.260159.",
    ]
    (OUT / "completion_R3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FOLLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
