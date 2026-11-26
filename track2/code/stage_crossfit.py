"""B01–B12 cross-fit. This is the only stage allowed to choose Track 2 meta-parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    CONTEXTS,
    EPS,
    HORIZON,
    IMPROVE,
    LAMBDAS,
    MAX_FRAC,
    MEMBERS,
    MIN_PRIOR,
    RESULTS,
    TIE,
    TOP7,
    TRACK1,
    apply_weights,
    fit_constrained_least_squares,
    fit_inverse_mase,
    history_scales,
    load_b_targets,
    load_predictions,
    macro_mase,
    task_scores_from_long,
    write_json,
)

B_ROOT = TRACK1 / "foundation_b_default"
FOLD_RANK = {f"B{i:02d}": i for i in range(1, 13)}


def _panel() -> pd.DataFrame:
    predictions = load_predictions(B_ROOT, MEMBERS, destination="Macau")
    targets = load_b_targets()
    keep = [
        "task_id", "target_date", "horizon_day", "actual",
    ]
    merged = predictions.merge(targets[keep], on=["task_id", "target_date", "horizon_day"], how="left", validate="many_to_one")
    if merged["actual"].isna().any():
        raise RuntimeError("B predictions did not match B actuals")
    scales = history_scales("recovery_tuning", "all", "Macau")
    merged = merged.merge(scales, on="task_id", how="left", validate="many_to_one")
    if merged["s7_recomputed"].isna().any():
        raise RuntimeError("missing recomputed s7")
    # Scoring uses the recomputed scale. The file scale is only a check.
    merged["scale_gap"] = (merged["mase7_scale"] - merged["s7_recomputed"]).abs()
    gap = float(merged["scale_gap"].max())
    merged["mase7_scale"] = merged["s7_recomputed"]
    value_columns = ["prediction", "q0.1", "q0.5", "q0.9"]
    index = ["fold_id", "market", "task_id", "horizon_day", "origin_date", "target_date", "actual", "mase7_scale", "recent28_mean"]
    wide = merged.pivot_table(index=index, columns="model", values=value_columns, aggfunc="first")
    wide.columns = [f"{value}__{model}" for value, model in wide.columns]
    frame = wide.reset_index()
    threshold = merged.groupby("task_id")["high_demand_threshold"].first()
    frame["high_demand_threshold"] = frame["task_id"].map(threshold)
    folds = sorted(frame["fold_id"].unique(), key=FOLD_RANK.get)
    if folds != [f"B{i:02d}" for i in range(1, 13)]:
        raise RuntimeError(f"unexpected B folds: {folds}")
    if frame[[f"prediction__{model}" for model in MEMBERS]].isna().any().any():
        raise RuntimeError("member predictions did not align on the B panel")
    frame = frame.sort_values(["fold_id", "market", "horizon_day"]).reset_index(drop=True)
    frame["destination"] = "Macau"
    frame.attrs["scale_gap"] = gap
    return frame


def _matrix(frame: pd.DataFrame, models: list[str]) -> np.ndarray:
    return np.column_stack([frame[f"prediction__{model}"].to_numpy(float) for model in models])


def _member_macros(frame: pd.DataFrame, models: list[str]) -> dict[str, float]:
    scores = {}
    for model in models:
        column = f"prediction__{model}"
        tasks = task_scores_from_long(frame.assign(prediction=frame[column]), "prediction")
        scores[model] = macro_mase(tasks)
    return scores


def _predict_equal(matrix: np.ndarray) -> np.ndarray:
    return apply_weights(matrix, np.full(matrix.shape[1], 1.0 / matrix.shape[1]))


def _weights_for(method: str, train: pd.DataFrame) -> tuple[np.ndarray, str]:
    if method == "T2-E2" or train["fold_id"].nunique() < MIN_PRIOR:
        return np.full(3, 1.0 / 3.0), "equal"
    if method == "T2-E3":
        scores = _member_macros(train, MEMBERS)
        weights = fit_inverse_mase(scores)
        return np.array([weights[model] for model in MEMBERS]), "ok"
    design = _matrix(train, MEMBERS)
    target = train["actual"].to_numpy(float)
    return fit_constrained_least_squares(design, target)


def _horizon_weights(method: str, train: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return (28, 3) weights."""
    rows = []
    status = "ok"
    for horizon in range(1, HORIZON + 1):
        part = train[train["horizon_day"] == horizon]
        if method == "T2-E3":
            scores = {}
            for model in MEMBERS:
                tasks = task_scores_from_long(part.assign(prediction=part[f"prediction__{model}"]), "prediction")
                scores[model] = macro_mase(tasks)
            weights = fit_inverse_mase(scores)
            rows.append([weights[model] for model in MEMBERS])
        else:
            weights, this_status = fit_constrained_least_squares(_matrix(part, MEMBERS), part["actual"].to_numpy(float))
            if this_status != "ok":
                status = "equal"
            rows.append(weights)
    return np.vstack(rows), status


def _apply_fold_weights(part: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    matrix = _matrix(part, MEMBERS)
    if weights.ndim == 1:
        return apply_weights(matrix, weights)
    horizons = part["horizon_day"].to_numpy(int)
    out = np.empty(len(part), dtype=float)
    for horizon in range(1, HORIZON + 1):
        mask = horizons == horizon
        out[mask] = apply_weights(matrix[mask], weights[horizon - 1])
    return out


def _score_prediction(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = frame.copy()
    scored["prediction"] = prediction
    return macro_mase(task_scores_from_long(scored, "prediction"))


def _crossfit(frame: pd.DataFrame, method: str, per_horizon: bool) -> tuple[np.ndarray, list[dict]]:
    folds = [f"B{i:02d}" for i in range(1, 13)]
    prediction = np.zeros(len(frame), dtype=float)
    logs = []
    for fold in folds:
        train = frame[frame["fold_id"].map(FOLD_RANK) < FOLD_RANK[fold]]
        part = frame[frame["fold_id"] == fold]
        use_horizon = per_horizon and train["fold_id"].nunique() >= MIN_PRIOR and method in {"T2-E3", "T2-E4"}
        if use_horizon:
            weights, status = _horizon_weights(method, train)
        else:
            weights, status = _weights_for(method, train)
        values = _apply_fold_weights(part, weights)
        prediction[part.index.to_numpy()] = values
        record = {"fold_id": fold, "method": method, "per_horizon": use_horizon, "status": status, "n_train_folds": int(train["fold_id"].nunique())}
        flat = np.asarray(weights, dtype=float)
        if flat.ndim == 1:
            record.update({model: float(weight) for model, weight in zip(MEMBERS, flat)})
        logs.append(record)
    return prediction, logs


def _shrink(base: np.ndarray, correction: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = lam * correction
    limit = MAX_FRAC * np.abs(base)
    shrunk = np.clip(raw, -limit, limit)
    final = np.maximum(0.0, base + shrunk)
    return raw, shrunk, final


def _residual_forecast(prior: np.ndarray, context: int, kind: str) -> np.ndarray | None:
    """prior is (n_origins, 28) in time order. Returns a 28-vector or None if context is short."""
    flat = prior.reshape(-1)
    if len(flat) < context:
        return None
    window = flat[-context:]
    if kind == "horizon_mean":
        origins = window.reshape(-1, HORIZON)
        return origins.mean(axis=0)
    last = window[-7:]
    return np.resize(last, HORIZON)


def _residual_grid(frame: pd.DataFrame, ensemble: np.ndarray) -> tuple[pd.DataFrame, dict]:
    work = frame.copy()
    work["ensemble"] = ensemble
    work["residual"] = work["actual"] - work["ensemble"]
    folds = [f"B{i:02d}" for i in range(1, 13)]
    markets = sorted(work["market"].unique())
    rows = []
    cache: dict[tuple, np.ndarray] = {}
    for kind in ("horizon_mean", "seasonal7"):
        for context in CONTEXTS:
            for lam in LAMBDAS:
                corrected = work["ensemble"].to_numpy(float).copy()
                raw_sum = np.zeros(len(work))
                shrunk_sum = np.zeros(len(work))
                for market in markets:
                    market_frame = work[work["market"] == market]
                    residual_by_fold = {}
                    for fold in folds:
                        part = market_frame[market_frame["fold_id"] == fold].sort_values("horizon_day")
                        residual_by_fold[fold] = part["residual"].to_numpy(float)
                    for index, fold in enumerate(folds):
                        part = market_frame[market_frame["fold_id"] == fold].sort_values("horizon_day")
                        if index < MIN_PRIOR or lam == 0:
                            continue
                        prior = np.vstack([residual_by_fold[name] for name in folds[:index]])
                        forecast = _residual_forecast(prior, context, kind)
                        if forecast is None:
                            continue
                        raw, shrunk, final = _shrink(part["ensemble"].to_numpy(float), forecast, lam)
                        corrected[part.index.to_numpy()] = final
                        raw_sum[part.index.to_numpy()] = raw
                        shrunk_sum[part.index.to_numpy()] = shrunk
                score = _score_prediction(work, corrected)
                rows.append({"kind": kind, "context": context, "lambda": lam, "mase": score})
                cache[(kind, context, lam)] = corrected
    table = pd.DataFrame(rows).sort_values("mase")
    base = float(table.loc[table["lambda"] == 0, "mase"].iloc[0])
    best = table.iloc[0]
    adopted = bool(best["lambda"] > 0 and best["mase"] <= base - IMPROVE)
    choice = {
        "probe_best_kind": None if not adopted else str(best["kind"]),
        "probe_best_context": None if not adopted else int(best["context"]),
        "probe_best_lambda": 0.0 if not adopted else float(best["lambda"]),
        "probe_best_mase": float(best["mase"]),
        "lambda0_mase": base,
        "adopted": False,
        "freeze_reason": "no_stable_improvement" if not adopted else "probe_improved_but_member_checkpoint_not_run",
    }
    # Member checkpoints are the only eligible correctors. No GPU is available,
    # so a univariate probe is reported and is not written into the frozen config.
    if adopted:
        choice["freeze_reason"] = "no_gpu_member_corrector_not_fitted"
        choice["probe_best_kind"] = str(best["kind"])
        choice["probe_best_context"] = int(best["context"])
        choice["probe_best_lambda"] = float(best["lambda"])
    return table, choice


def _pinball(actual: np.ndarray, quantile: np.ndarray, level: float) -> float:
    gap = actual - quantile
    return float(np.mean(np.where(gap >= 0, level * gap, (level - 1) * gap)))


def _interval_score(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    width = high - low
    alpha = 0.2
    below = (low - actual) * (actual < low)
    above = (actual - high) * (actual > high)
    return float(np.mean(width + (2 / alpha) * below + (2 / alpha) * above))


def _probability(frame: pd.DataFrame, logs: list[dict]) -> dict:
    lows, mids, highs = [], [], []
    fold_weight = {row["fold_id"]: np.array([row[model] for model in MEMBERS], dtype=float) for row in logs if "granite_patchtst_fm" in row}
    for fold, part in frame.groupby("fold_id", sort=False):
        weights = fold_weight[fold]
        low = sum(weights[i] * part[f"q0.1__{model}"].to_numpy(float) for i, model in enumerate(MEMBERS))
        mid = sum(weights[i] * part[f"q0.5__{model}"].to_numpy(float) for i, model in enumerate(MEMBERS))
        high = sum(weights[i] * part[f"q0.9__{model}"].to_numpy(float) for i, model in enumerate(MEMBERS))
        ordered = np.sort(np.column_stack([low, mid, high]), axis=1)
        lows.append(pd.Series(ordered[:, 0], index=part.index))
        mids.append(pd.Series(ordered[:, 1], index=part.index))
        highs.append(pd.Series(ordered[:, 2], index=part.index))
    low = pd.concat(lows).sort_index().to_numpy(float)
    high = pd.concat(highs).sort_index().to_numpy(float)
    actual = frame["actual"].to_numpy(float)
    coverage = float(np.mean((actual >= low) & (actual <= high)))
    report = {
        "pinball_q10": _pinball(actual, low, 0.1),
        "pinball_q50": _pinball(actual, pd.concat(mids).sort_index().to_numpy(float), 0.5),
        "pinball_q90": _pinball(actual, high, 0.9),
        "wis80": _interval_score(actual, low, high),
        "coverage80": coverage,
        "mean_width": float(np.mean(high - low)),
        "method": "weighted_quantiles_only",
    }
    if abs(coverage - 0.8) > 0.10:
        report["method"] = "split_conformal_by_horizon"
        report["coverage_before_conformal"] = coverage
    return report


def _conformal_quantiles(train: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    residual = np.abs(train["actual"].to_numpy(float) - point)
    horizons = train["horizon_day"].to_numpy(int)
    quantiles = np.zeros(HORIZON)
    for horizon in range(1, HORIZON + 1):
        sample = residual[horizons == horizon]
        level = min(1.0, np.ceil((len(sample) + 1) * 0.8) / len(sample))
        quantiles[horizon - 1] = float(np.quantile(sample, level))
    return quantiles


def _quality(frame: pd.DataFrame) -> dict:
    ratios = []
    for model in MEMBERS:
        grouped = frame.groupby(["task_id", "recent28_mean"], sort=False)[f"prediction__{model}"].mean()
        for (task_id, recent), pred_mean in grouped.items():
            if recent >= 5:
                ratios.append(pred_mean / recent)
    array = np.asarray(ratios, dtype=float)
    return {
        "r_low": float(np.quantile(array, 0.005)),
        "r_high": float(np.quantile(array, 0.995)),
        "n_ratios": int(len(array)),
    }


def _timeouts() -> dict:
    path = TRACK1 / "compute_benchmark" / "benchmark.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            import json
            rows.append(json.loads(line))
    timeouts = {}
    for model in MEMBERS:
        walls = [row["wall_seconds"] for row in rows if row.get("model") == model and not row.get("warmup") and row.get("returncode") == 0]
        if not walls:
            timeouts[model] = None
            continue
        timeouts[model] = float(3 * np.quantile(walls, 0.95))
    return timeouts


def _topk(frame: pd.DataFrame) -> pd.DataFrame:
    extra = load_predictions(B_ROOT, ["moirai", "patchtst", "lag_llama", "informer"], destination="Macau")
    extra = extra[["task_id", "horizon_day", "model", "prediction"]]
    base = frame[["task_id", "horizon_day", "fold_id", "market", "destination", "actual", "mase7_scale"]].copy()
    rows = []
    groups = {
        "Top-1": MEMBERS[:1],
        "Top-2": MEMBERS[:2],
        "Top-3": MEMBERS[:3],
        "Top-4": MEMBERS[:3] + ["moirai"],
        "Top-7": TOP7,
    }
    for name, models in groups.items():
        pieces = []
        for model in models:
            if model in MEMBERS:
                column = frame[["task_id", "horizon_day", f"prediction__{model}"]].rename(columns={f"prediction__{model}": "prediction"})
            else:
                column = extra[extra["model"] == model][["task_id", "horizon_day", "prediction"]]
            column = column.rename(columns={"prediction": model})
            pieces.append(column)
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.merge(piece, on=["task_id", "horizon_day"], how="inner")
        merged["prediction"] = merged[models].mean(axis=1).clip(lower=0)
        scored = base.merge(merged[["task_id", "horizon_day", "prediction"]], on=["task_id", "horizon_day"], how="inner")
        rows.append({"combination": name, "models": ",".join(models), "mase": _score_prediction(scored, scored["prediction"].to_numpy(float)), "tasks": int(scored["task_id"].nunique())})
    return pd.DataFrame(rows)


def _choose(scores: dict[str, float]) -> str:
    order = ["T2-E2", "T2-E3", "T2-E4"]
    best_value = min(scores[name] for name in order)
    eligible = [name for name in order if scores[name] <= best_value + TIE]
    return eligible[0]


def main() -> None:
    frame = _panel()
    scale_gap = frame.attrs.get("scale_gap")
    output = RESULTS / "b_crossfit"
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    stored = {}
    weight_logs = []
    for method, models in (("T2-E0", MEMBERS[:1]), ("T2-E1", MEMBERS[:2])):
        prediction = _predict_equal(_matrix(frame, models))
        scores[method] = _score_prediction(frame, prediction)
        stored[method] = prediction
    for method in ("T2-E2", "T2-E3", "T2-E4"):
        prediction, logs = _crossfit(frame, method, per_horizon=False)
        scores[method] = _score_prediction(frame, prediction)
        stored[method] = prediction
        weight_logs.extend(logs)
    selected = _choose(scores)
    per_horizon = False
    horizon_score = None
    if selected in {"T2-E3", "T2-E4"}:
        horizon_prediction, horizon_logs = _crossfit(frame, selected, per_horizon=True)
        horizon_score = _score_prediction(frame, horizon_prediction)
        weight_logs.extend(horizon_logs)
        if horizon_score <= scores[selected] - IMPROVE:
            per_horizon = True
            stored[selected] = horizon_prediction
            scores[selected + "h"] = horizon_score
    ensemble = stored[selected]
    residual_table, residual_choice = _residual_grid(frame, ensemble)
    residual_table.to_csv(output / "residual_probe.csv", index=False)
    probability = _probability(frame, [row for row in weight_logs if row["method"] == selected and row["per_horizon"] == per_horizon])
    if probability["method"] == "split_conformal_by_horizon":
        # Refit the conformal widths on all B residuals of the selected ensemble.
        # The frozen outer interval uses this full-B width; coverage above is the
        # uncalibrated weighted-quantile coverage that triggered the switch.
        point = ensemble
        quantiles = _conformal_quantiles(frame, point)
        horizons = frame["horizon_day"].to_numpy(int)
        half = quantiles[horizons - 1]
        low = np.maximum(0.0, point - half)
        high = point + half
        mid = point
        ordered = np.sort(np.column_stack([low, mid, high]), axis=1)
        actual = frame["actual"].to_numpy(float)
        probability["conformal_coverage80"] = float(np.mean((actual >= ordered[:, 0]) & (actual <= ordered[:, 2])))
        probability["conformal_mean_width"] = float(np.mean(ordered[:, 2] - ordered[:, 0]))
        probability["conformal_abs_residual_q80"] = quantiles.tolist()
    quality = _quality(frame)
    timeouts = _timeouts()
    topk = _topk(frame)
    topk.to_csv(output / "topk_marginal.csv", index=False)
    pd.DataFrame(weight_logs).to_json(output / "weights_crossfit.jsonl", orient="records", lines=True)
    refit_status = "equal"
    if selected == "T2-E2":
        refit = np.full(3, 1.0 / 3.0)
        refit_status = "equal"
        refit_horizon = None
    elif not per_horizon:
        refit, refit_status = _weights_for(selected, frame)
        refit_horizon = None
    else:
        refit_horizon, refit_status = _horizon_weights(selected, frame)
        refit = refit_horizon.mean(axis=0)
    selection = [{"method": name, "crossfit_mase": value, "selected": name == selected} for name, value in scores.items()]
    pd.DataFrame(selection).to_csv(output / "selection_table.csv", index=False)
    write_json(output / "crossfit_summary.json", {
        "scale_check_max_abs_gap": scale_gap,
        "scores": scores,
        "selected_method": selected,
        "per_horizon": per_horizon,
        "horizon_mase": horizon_score,
        "residual": residual_choice,
        "probability": probability,
        "quality": quality,
        "timeouts_seconds": timeouts,
        "refit_status": refit_status,
        "refit_weights": {model: float(weight) for model, weight in zip(MEMBERS, np.atleast_1d(refit))},
        "outer_used_for_selection": False,
        "least_squares_solver": "simplex_enumeration",
    })
    frozen = {
        "schema_version": 1,
        "selected_method": selected,
        "epsilon": EPS,
        "per_horizon": per_horizon,
        "weights": {model: float(weight) for model, weight in zip(MEMBERS, np.atleast_1d(refit))},
        "horizon_weights": None if refit_horizon is None else {str(h + 1): {model: float(refit_horizon[h, i]) for i, model in enumerate(MEMBERS)} for h in range(HORIZON)},
        "residual": {"enabled": False, "lambda": 0.0, "corrector": None, "context_days": None, "max_frac": MAX_FRAC, "reason": residual_choice["freeze_reason"]},
        "probability": {"method": probability["method"], "quantiles": [0.1, 0.5, 0.9], "conformal_abs_residual_q80": probability.get("conformal_abs_residual_q80")},
        "point_source": "member_prediction_column",
        "min_prior_folds": MIN_PRIOR,
        "selection_metric": "task_MASE7_market_fold_macro",
        "outer_used_for_selection": False,
    }
    from pathlib import Path
    import json
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = config_dir / "track2_ensemble_config.yaml"
    fallback_path = config_dir / "track2_fallback_policy.yaml"
    ensemble_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fallback = {
        "schema_version": 1,
        "order": ["frozen_ensemble", "renormalize_remaining", "equal_remaining", "seasonal_naive"],
        "r_low": quality["r_low"],
        "r_high": quality["r_high"],
        "recent28_skip_below": 5,
        "timeout_seconds": timeouts,
        "residual_failure_closes_residual_only": True,
    }
    fallback_path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from common import sha256_file
    manifest = {
        "ensemble_config_sha256": sha256_file(ensemble_path),
        "fallback_policy_sha256": sha256_file(fallback_path),
        "selected_models_sha256": sha256_file(config_dir / "selected_track2_models.yaml"),
        "outer_used_for_selection": False,
        "selected_method": selected,
        "per_horizon": per_horizon,
    }
    manifest_path = config_dir / "track2_freeze_manifest.json"
    write_json(manifest_path, manifest)
    print(pd.DataFrame(selection).to_string(index=False))
    print("selected", selected, "per_horizon", per_horizon)
    print("residual", residual_choice["freeze_reason"], "probability", probability["method"], "coverage", probability["coverage80"])


if __name__ == "__main__":
    main()
