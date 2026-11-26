"""Apply the frozen Track 2 configuration. Does not refit weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, HORIZON, MEMBERS, RESULTS, TRACK1, load_predictions, sha256_file

FROZEN = json.loads((CONFIG / "track2_ensemble_config.yaml").read_text(encoding="utf-8"))
FALLBACK = json.loads((CONFIG / "track2_fallback_policy.yaml").read_text(encoding="utf-8"))
MANIFEST = json.loads((CONFIG / "track2_freeze_manifest.json").read_text(encoding="utf-8"))


def _check_freeze() -> None:
    if MANIFEST.get("outer_used_for_selection"):
        raise SystemExit("freeze manifest says outer data were used for selection")
    if sha256_file(CONFIG / "track2_ensemble_config.yaml") != MANIFEST["ensemble_config_sha256"]:
        raise SystemExit("ensemble config hash mismatch")
    if sha256_file(CONFIG / "track2_fallback_policy.yaml") != MANIFEST["fallback_policy_sha256"]:
        raise SystemExit("fallback policy hash mismatch")


def _wide(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    index = ["task_id", "fold_id", "destination", "market", "origin_date", "target_date", "horizon_day", "history_length", "mase7_scale"]
    extra = [column for column in ("high_demand_threshold", "rmsse7_scale") if column in frame.columns]
    pieces = []
    for model in models:
        part = frame[frame["model"] == model][index + extra + ["prediction", "q0.1", "q0.5", "q0.9"]].copy()
        part = part.rename(columns={
            "prediction": f"prediction__{model}",
            "q0.1": f"q0.1__{model}",
            "q0.5": f"q0.5__{model}",
            "q0.9": f"q0.9__{model}",
        })
        pieces.append(part)
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=index, how="inner", suffixes=("", "_dup"))
        drop = [column for column in merged.columns if column.endswith("_dup")]
        merged = merged.drop(columns=drop)
    return merged


def _weights_matrix(horizons: np.ndarray) -> np.ndarray:
    if FROZEN["per_horizon"]:
        table = FROZEN["horizon_weights"]
        return np.vstack([[table[str(h)][model] for model in MEMBERS] for h in horizons])
    weights = np.array([FROZEN["weights"][model] for model in MEMBERS], dtype=float)
    return np.repeat(weights.reshape(1, -1), len(horizons), axis=0)


def _combine(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    wide = _wide(frame, models)
    matrix = np.column_stack([wide[f"prediction__{model}"].to_numpy(float) for model in models])
    if models == MEMBERS:
        weights = _weights_matrix(wide["horizon_day"].to_numpy(int))
        point = np.maximum(0.0, np.sum(matrix * weights, axis=1))
        low = np.zeros(len(wide))
        mid = np.zeros(len(wide))
        high = np.zeros(len(wide))
        for offset, model in enumerate(models):
            low += weights[:, offset] * wide[f"q0.1__{model}"].to_numpy(float)
            mid += weights[:, offset] * wide[f"q0.5__{model}"].to_numpy(float)
            high += weights[:, offset] * wide[f"q0.9__{model}"].to_numpy(float)
        ordered = np.sort(np.column_stack([low, mid, high]), axis=1)
        if FROZEN["probability"]["method"] == "split_conformal_by_horizon":
            half = np.array(FROZEN["probability"]["conformal_abs_residual_q80"], dtype=float)
            half = half[wide["horizon_day"].to_numpy(int) - 1]
            ordered[:, 0] = np.maximum(0.0, point - half)
            ordered[:, 1] = point
            ordered[:, 2] = point + half
            ordered = np.sort(ordered, axis=1)
        wide["q0.1"] = ordered[:, 0]
        wide["q0.5"] = ordered[:, 1]
        wide["q0.9"] = ordered[:, 2]
    else:
        point = np.maximum(0.0, matrix.mean(axis=1))
    wide["prediction_raw"] = point
    wide["prediction"] = point
    return wide


def _write(frame: pd.DataFrame, experiment: str, directory: Path) -> None:
    out = frame.copy()
    out["model"] = experiment
    out["candidate_id"] = experiment
    out["seed"] = 0
    out["split_name"] = out.get("split_name", "outer_backtest")
    columns = [
        "task_id", "fold_id", "destination", "market", "origin_date", "target_date",
        "horizon_day", "history_length", "prediction_raw", "prediction", "model",
        "candidate_id", "seed", "mase7_scale",
    ]
    for optional in ("high_demand_threshold", "rmsse7_scale", "q0.1", "q0.5", "q0.9", "split_name"):
        if optional in out.columns and optional not in columns:
            columns.append(optional)
    directory.mkdir(parents=True, exist_ok=True)
    saved = out[columns].copy()
    for column in ("origin_date", "target_date"):
        if column in saved.columns:
            saved[column] = pd.to_datetime(saved[column])
    saved.to_parquet(directory / "predictions.parquet", index=False)
    (directory / "run_manifest.json").write_text(json.dumps({
        "experiment": experiment,
        "ensemble_config_sha256": MANIFEST["ensemble_config_sha256"],
        "actuals_visible_to_models": False,
        "rows": int(len(out)),
        "tasks": int(out["task_id"].nunique()),
    }, indent=2) + "\n", encoding="utf-8")


def _seasonal(destination: str) -> pd.DataFrame:
    root = TRACK1 / "outer_server_all" / "predictions" / "seasonal_naive" / "SN01" / "42"
    frame = pd.concat([pd.read_parquet(path) for path in root.glob("*.parquet")], ignore_index=True)
    return frame[frame["destination"] == destination]


def _passes_quality(part: pd.DataFrame, recent: float) -> bool:
    values = part.sort_values("horizon_day")
    if len(values) != HORIZON or not np.isfinite(values["prediction"]).all():
        return False
    if values["horizon_day"].tolist() != list(range(1, HORIZON + 1)):
        return False
    if recent < FALLBACK["recent28_skip_below"]:
        return True
    ratio = float(values["prediction"].mean() / recent)
    return FALLBACK["r_low"] <= ratio <= FALLBACK["r_high"]


def apply_source(source: pd.DataFrame, destination: str, out_root: Path, seasonal: pd.DataFrame | None, recent_map: dict | None = None) -> None:
    macau = source[source["destination"] == destination].copy()
    if "split_name" not in macau.columns:
        macau["split_name"] = "outer_backtest"
    m1 = macau[macau["model"] == MEMBERS[0]]
    pair = macau[macau["model"].isin(MEMBERS[:2])]
    trio = macau[macau["model"].isin(MEMBERS)]
    _write(m1.assign(prediction_raw=m1["prediction"]), "T2-01", out_root / "T2-01")
    equal2 = _combine(pair, MEMBERS[:2])
    equal2["split_name"] = m1["split_name"].iloc[0]
    _write(equal2, "T2-02", out_root / "T2-02")
    equal3 = _combine(trio, MEMBERS)
    # _combine of three members currently applies frozen weights, not equal weights.
    # Rebuild the equal-weight forecast explicitly.
    wide = _wide(trio, MEMBERS)
    equal_point = np.maximum(0.0, np.mean(np.column_stack([wide[f"prediction__{model}"] for model in MEMBERS]), axis=1))
    wide["prediction_raw"] = equal_point
    wide["prediction"] = equal_point
    wide["split_name"] = m1["split_name"].iloc[0]
    _write(wide, "T2-03", out_root / "T2-03")
    weighted = _combine(trio, MEMBERS)
    weighted["split_name"] = m1["split_name"].iloc[0]
    _write(weighted, "T2-04", out_root / "T2-04")
    # Residual module is frozen off, so T2-05 matches T2-04.
    _write(weighted, "T2-05", out_root / "T2-05")
    _write(weighted, "T2-06", out_root / "T2-06")
    if seasonal is not None:
        _write_c2(m1, seasonal, out_root / "T2-C2", recent_map or {})


def _write_c2(m1: pd.DataFrame, seasonal: pd.DataFrame, directory: Path, recent_map: dict) -> None:
    pieces = []
    for task_id, part in m1.groupby("task_id", sort=False):
        recent = float(recent_map.get(task_id, np.nan)) if recent_map is not None else np.nan
        if _passes_quality(part, recent):
            pieces.append(part)
        else:
            replacement = seasonal[seasonal["task_id"] == task_id]
            if replacement.empty:
                replacement = seasonal[(seasonal["market"] == part["market"].iloc[0]) & (seasonal["fold_id"] == part["fold_id"].iloc[0])]
            pieces.append(replacement)
    combined = pd.concat(pieces, ignore_index=True)
    _write(combined, "T2-C2", directory)


def _baseline_equal(destination: str) -> pd.DataFrame:
    root = TRACK1 / "outer_server_all" / "predictions"
    members = {"ets": "ETS03", "ridge": "RID02", "elasticnet": "EN03"}
    frames = []
    for model, candidate in members.items():
        folder = root / model / candidate / "42"
        frame = pd.concat((pd.read_parquet(path) for path in folder.glob("*.parquet")), ignore_index=True)
        frame = frame[frame["destination"] == destination]
        frames.append(frame[["task_id", "horizon_day", "prediction"]].rename(columns={"prediction": model}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["task_id", "horizon_day"], how="inner")
    merged["prediction"] = merged[list(members)].mean(axis=1).clip(lower=0)
    merged["prediction_raw"] = merged["prediction"]
    meta = pd.concat((pd.read_parquet(path) for path in (root / "ets" / "ETS03" / "42").glob("*.parquet")), ignore_index=True)
    meta = meta[meta["destination"] == destination]
    keep = ["task_id", "fold_id", "destination", "market", "origin_date", "target_date", "horizon_day", "history_length", "mase7_scale", "split_name"]
    return meta[keep].merge(merged[["task_id", "horizon_day", "prediction", "prediction_raw"]], on=["task_id", "horizon_day"], how="inner")


def main() -> None:
    _check_freeze()
    from common import history_scales
    outer = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS)
    recent = history_scales("outer_backtest", "all", "Macau")
    recent_hk = history_scales("outer_backtest", "all", "HongKong")
    recent_map = dict(zip(pd.concat([recent, recent_hk])["task_id"], pd.concat([recent, recent_hk])["recent28_mean"]))
    seasonal = pd.concat([
        pd.read_parquet(path) for path in (TRACK1 / "outer_server_all" / "predictions" / "seasonal_naive" / "SN01" / "42").glob("*.parquet")
    ], ignore_index=True)
    for destination, folder in (("Macau", RESULTS / "outer_macau"), ("HongKong", RESULTS / "extension_hong_kong")):
        source = outer.copy()
        source.attrs["recent_map"] = recent_map
        # attrs on a filtered frame are kept if we filter after setting on the subset
        subset = source[source["destination"] == destination].copy()
        subset.attrs["recent_map"] = recent_map
        apply_source(subset, destination, folder, seasonal[seasonal["destination"] == destination], recent_map)
        if destination == "Macau":
            c1 = _baseline_equal(destination)
            _write(c1, "T2-C1", folder / "T2-C1")
    for history, base in (("180", TRACK1 / "sensitivity" / "outer_backtest" / "180"), ("365", TRACK1 / "sensitivity" / "outer_backtest" / "365")):
        source = load_predictions(base, MEMBERS, destination="Macau")
        apply_source(source, "Macau", RESULTS / f"sensitivity_{history}", seasonal=None, recent_map=None)
    pandemic = load_predictions(TRACK1 / "sensitivity" / "pandemic_stress" / "all", MEMBERS, destination="Macau")
    apply_source(pandemic, "Macau", RESULTS / "pandemic_stress", seasonal=None, recent_map=None)
    print("applied frozen config")


if __name__ == "__main__":
    main()
