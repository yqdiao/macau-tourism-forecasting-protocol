"""R3 stage runner. Training stays on CPU and never reads outer targets."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BASELINE_ROOT, CALENDAR, CODE_DIR, ORIGINAL_CODE, OUT, REPO, RUN_ID, SEEDS_LINEAR, SEED_TREE,
    TARGETS, TASK_INPUTS, TASK_MANIFEST, code_hash, limit_cpu, sha256_file, sha256_text, utc_now, write_json,
)
from data import load_calendar, load_origins, seasonal_naive
from linear import fit_linear
from score import load_targets, macro_mase, scale_frame, task_metrics
from trees import fit_trees

EXPECTED = {
    "NL01": 1.6273629280046054,
    "GL03": 1.3379897245507033,
    "LL01": 1.2601594411786645,
}
B_FOLDS = [f"B{i:02d}" for i in range(1, 13)]
O_FOLDS = [f"O{i:02d}" for i in range(1, 34)]
PROBE_FOLDS = ["B01", "B06", "B12"]


def specs() -> dict:
    tree = lambda pooled, target, l1, annual=False, phase=False, half="inf": {
        "family": "tree", "pooled": pooled, "target": target, "l1": l1, "annual": annual,
        "phase": phase, "half_life": half,
    }
    linear = lambda kind, refit: {"family": "linear", "kind": kind, "refit": refit}
    return {
        "R3V1_NP": linear("nlinear", False),
        "R3V1_NR": linear("nlinear", True),
        "R3V1_G": tree(True, "log", False),
        "R3V1_L": tree(False, "log", False),
        "R3V2_T1": tree(False, "log", True),
        "R3V2_T2": tree(False, "mean28", True),
        "R3V2_T3": tree(False, "week4", True),
        "R3V2_T4": tree(False, "week4", True, annual=True),
        "R3V2_T5": tree(False, "week4", True, annual=True, phase=True),
        "R3V2_N2": linear("scaled", True),
        "R3V2_N3": linear("scaled_calendar", True),
        "BASE_WEEK4_MEDIAN": {"family": "baseline", "kind": "week4"},
        "BASE_SEASONAL_NAIVE": {"family": "baseline", "kind": "naive"},
    }


def _seeds(model_id: str) -> tuple[int, ...]:
    spec = specs()[model_id]
    if spec["family"] == "linear":
        return SEEDS_LINEAR
    return (SEED_TREE,)


def _pred_path(model_id: str, seed: int, split: str, fold: str) -> Path:
    return OUT / "predictions" / model_id / f"seed{seed}" / split / f"{fold}.parquet"


def discover() -> dict:
    import subprocess
    limit_cpu()
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = value.strip()
    available_kib = int(mem["MemAvailable"].split()[0])
    body = {
        "run_id": RUN_ID, "created_at_utc": utc_now(), "repo_root": str(REPO),
        "data_paths": {"task_inputs": str(TASK_INPUTS), "targets_scorer_only": str(TARGETS),
                       "task_manifest": str(TASK_MANIFEST)},
        "calendar_path": str(CALENDAR),
        "baseline_prediction_paths": str(BASELINE_ROOT),
        "original_code": str(ORIGINAL_CODE),
        "python_executable": sys.executable,
        "versions": _versions(),
        "git_commit": commit,
        "dirty_diff_sha256": sha256_text(dirty),
        "input_hashes": {
            "task_inputs": sha256_file(TASK_INPUTS), "targets": sha256_file(TARGETS),
            "calendar": sha256_file(CALENDAR), "original_light_models": sha256_file(ORIGINAL_CODE),
        },
        "expected_original_light_models_sha256": "999320d2df0ac76c00e72179b22ef766c3d8ca90b7084d36410749d54b47f533",
        "expected_calendar_sha256": "9060f62e542a884cf6f56d2feda8e527bb0b37eaebdedfbb5441a4357cf6cadc",
        "cpu_count": os.cpu_count(), "mem_available_kib": available_kib,
        "rss_budget_bytes": min(16 * 1024**3, int(available_kib * 1024 * 0.25)),
        "r3_code_hash": code_hash(),
    }
    body["original_code_matches_reference"] = body["input_hashes"]["original_light_models"] == body["expected_original_light_models_sha256"]
    body["calendar_matches_reference"] = body["input_hashes"]["calendar"] == body["expected_calendar_sha256"]
    write_json(OUT / "discovery.json", body)
    return body


def _versions() -> dict:
    import lightgbm
    import numpy
    import pandas
    import torch
    return {"lightgbm": lightgbm.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
            "torch": torch.__version__}


def reproduce_baselines(bundles) -> pd.DataFrame:
    scales = scale_frame(bundles)
    targets = load_targets()
    rows = []
    mapping = {
        "NL01": BASELINE_ROOT / "outer_backtest/global_linear/NL01",
        "GL03": BASELINE_ROOT / "outer_backtest/global_lightgbm/GL03",
        "LL01": BASELINE_ROOT / "outer_backtest/local_lightgbm/LL01",
    }
    records = []
    for name, root in mapping.items():
        for path in sorted(root.glob("seed*/O*_*/predictions.parquet")):
            frame = pd.read_parquet(path, columns=["task_id", "horizon_day", "prediction", "market"])
            seed = int(path.parts[-3].replace("seed", ""))
            frame["model_id"] = name
            frame["seed"] = seed
            frame["status"] = "native"
            rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    outer = {k: v for k, v in bundles.items() if k[0] == "outer_backtest"}
    metrics = task_metrics(predictions, targets, scale_frame(outer))
    for name, expected in EXPECTED.items():
        got = macro_mase(metrics[metrics["model_id"] == name])
        records.append({"model": name, "split": "outer_all", "MASE7_common": got, "expected": expected,
                        "abs_diff": abs(got - expected), "pass": abs(got - expected) <= 1e-6})
    report = pd.DataFrame(records)
    report.to_csv(OUT / "baseline_reproduction.csv", index=False)
    scales.to_parquet(OUT / "scale_reference.parquet", index=False)
    if not bool(report["pass"].all()):
        raise RuntimeError(report.to_string(index=False))
    return report


def _fit_one(bundle, model_id: str, seed: int, calendar, smoke: bool = False) -> dict:
    limit_cpu()
    spec = specs()[model_id]
    if spec["family"] == "baseline":
        preds, status = {}, {}
        from features import week4_baseline
        for i, market in enumerate(bundle.markets):
            y = bundle.values[i]
            raw = seasonal_naive(y) if spec["kind"] == "naive" else week4_baseline(y, np.array([len(y) - 1]))
            preds[market] = raw
            status[market] = "native"
        return {"predictions": preds, "status": status, "fit_seconds": 0.0, "inference_seconds": 0.0,
                "select_seconds": 0.0, "refit_seconds": 0.0, "best_epoch": None, "epochs_run": None}
    if spec["family"] == "linear":
        return fit_linear(bundle, spec, calendar, seed=seed, smoke=smoke)
    return fit_trees(bundle, spec, calendar, pooled=spec["pooled"], seed=seed,
                     n_estimators=10 if smoke else None)


def _store(bundle, model_id: str, seed: int, result: dict, smoke: bool) -> None:
    rows = []
    for i, (market, task_id) in enumerate(zip(bundle.markets, bundle.task_ids)):
        raw = np.asarray(result["predictions"][market], dtype=float)
        status = result["status"][market]
        for h in range(1, 29):
            target_date = bundle.origin.fromordinal(bundle.origin.toordinal() + h)
            value = float(raw[h - 1])
            rows.append({
                "run_id": RUN_ID, "model_id": model_id, "split": bundle.split, "task_id": task_id,
                "market": market, "origin": bundle.origin.isoformat(), "fold": bundle.fold,
                "horizon_day": h, "target_date": target_date.isoformat(), "seed": seed,
                "prediction_raw": value, "prediction": value if np.isfinite(value) else np.nan,
                "status": status, "smoke_only": smoke,
            })
    frame = pd.DataFrame(rows)
    frame["prediction"] = frame["prediction"].clip(lower=0)
    path = _pred_path(model_id, seed, bundle.split, bundle.fold)
    if smoke:
        path = path.with_name(path.stem + "_smoke.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manifest = {k: result.get(k) for k in ("fit_seconds", "select_seconds", "refit_seconds", "inference_seconds",
                                           "best_epoch", "epochs_run", "rows", "n_params")}
    manifest.update({"model_id": model_id, "seed": seed, "fold": bundle.fold, "split": bundle.split,
                     "at_utc": utc_now(), "code_hash": code_hash(), "smoke_only": smoke,
                     "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    (path.with_suffix(".manifest.json")).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    if result.get("epoch_history"):
        hist = pd.DataFrame(result["epoch_history"])
        hist["model_id"] = model_id
        hist["seed"] = seed
        hist["fold"] = bundle.fold
        hist_path = OUT / "epoch_history.csv"
        hist.to_csv(hist_path, mode="a", header=not hist_path.exists(), index=False)


def run_jobs(bundles, model_ids, folds, smoke: bool = False) -> list[dict]:
    calendar = load_calendar()
    done = []
    for fold in folds:
        split = "recovery_tuning" if fold.startswith("B") else "outer_backtest" if fold.startswith("O") else "pandemic_stress"
        bundle = bundles[(split, fold)]
        for model_id in model_ids:
            for seed in _seeds(model_id):
                path = _pred_path(model_id, seed, split, fold)
                if smoke:
                    path = path.with_name(path.stem + "_smoke.parquet")
                if path.exists() and not smoke:
                    done.append({"model_id": model_id, "seed": seed, "fold": fold, "event": "skip"})
                    continue
                t0 = time.perf_counter()
                try:
                    result = _fit_one(bundle, model_id, seed, calendar, smoke=smoke)
                    _store(bundle, model_id, seed, result, smoke)
                    done.append({"model_id": model_id, "seed": seed, "fold": fold, "event": "done",
                                 "seconds": round(time.perf_counter() - t0, 3),
                                 "fit_seconds": result.get("fit_seconds")})
                except Exception as exc:  # noqa: BLE001
                    done.append({"model_id": model_id, "seed": seed, "fold": fold, "event": "failed",
                                 "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]})
                    (OUT / "failures.csv").parent.mkdir(parents=True, exist_ok=True)
                    with (OUT / "failures.csv").open("a", encoding="utf-8") as stream:
                        stream.write(f"{model_id},{seed},{fold},{type(exc).__name__},{exc}\n")
                print(json.dumps(done[-1]), flush=True)
    return done


def score_stored(bundles, model_ids, split: str) -> pd.DataFrame:
    scales = scale_frame({k: v for k, v in bundles.items() if k[0] == split})
    targets = load_targets()
    frames = []
    for model_id in model_ids:
        for seed in _seeds(model_id):
            for path in (OUT / "predictions" / model_id / f"seed{seed}" / split).glob("*.parquet"):
                if path.name.endswith("_smoke.parquet"):
                    continue
                frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError(f"no predictions for {split}")
    metrics = task_metrics(pd.concat(frames, ignore_index=True), targets, scales)
    return metrics
