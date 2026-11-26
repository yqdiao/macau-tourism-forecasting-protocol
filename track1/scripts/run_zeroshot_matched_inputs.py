"""Forecast zero-shot models with the lightweight model's extra inputs.

The published univariate run saw only one market's history. This run adds the
two inputs used by the proposed lightweight models, and nothing after the origin:

- the seven statutory-calendar columns, known through the 28-day horizon
- the other 26 Macau markets' history through the same origin

Targets are never read. Each model process uses its own pinned environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
sys.path.insert(0, str(REPO / "track1/code"))

from light_models import CALENDAR_COLUMNS, Calendar  # noqa: E402

INPUTS = REPO / "shared/data/task_inputs.parquet"
CALENDAR_PATH = REPO / "shared/data/holiday_calendar.parquet"
TARGETS = REPO / "shared/data/targets.parquet"
REGISTRY = REPO / "track1/artifacts/server_runs/upstream_revisions_runtime.yaml"
HORIZON = 28
CACHE = {
    "chronos_2": Path("/root/DYQ/foundation-cache/chronos_2"),
    "timesfm_3": Path("/root/DYQ/foundation-cache/timesfm_3"),
    "moirai": Path("/root/DYQ/foundation-cache/moirai"),
    "granite_patchtst_fm": Path("/root/DYQ/foundation-cache/granite_patchtst_fm"),
    "lag_llama": Path("/root/DYQ/foundation-cache/lag_llama"),
}
CONTEXT = {"moirai": 1680, "lag_llama": 512}


def _registry(model: str) -> dict:
    import yaml
    body = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return body["models"][model]


def _load_groups(split: str, folds: set[str] | None) -> list[pd.DataFrame]:
    frame = pd.read_parquet(INPUTS)
    frame = frame.loc[(frame["split_name"] == split) & (frame["history_length"] == "all")
                      & (frame["destination"] == "Macau")].copy()
    if folds:
        frame = frame.loc[frame["fold_id"].isin(folds)]
    groups = []
    for _, part in frame.groupby(["fold_id", "origin_date"], sort=True):
        part = part.sort_values("market").reset_index(drop=True)
        if len(part) != 27:
            raise ValueError(f"{part['fold_id'].iloc[0]} has {len(part)} Macau markets")
        groups.append(part)
    return groups


def _panel(group: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Timestamp]:
    histories = [np.asarray(value, dtype=np.float32) for value in group["history"]]
    length = {len(item) for item in histories}
    if len(length) != 1:
        raise ValueError("markets at one origin have different history lengths")
    start = pd.Timestamp(group["history_start"].iloc[0])
    return np.stack(histories), group["market"].tolist(), start


def _calendar_block(calendar: Calendar, markets: list[str], start: pd.Timestamp, length: int) -> np.ndarray:
    days = (start + pd.to_timedelta(np.arange(length), unit="D")).to_numpy(dtype="datetime64[D]")
    return np.stack([calendar.features(market, days) for market in markets]).astype(np.float32)


def _scale(history: np.ndarray) -> float:
    difference = np.asarray(history[7:] - history[:-7], dtype=float)
    return float(np.mean(np.abs(difference)))


def _rows(group: pd.DataFrame, forecasts: np.ndarray, detail: dict) -> pd.DataFrame:
    records = []
    for task, forecast in zip(group.itertuples(index=False), forecasts):
        raw = np.asarray(forecast, dtype=float).reshape(-1)
        if len(raw) != HORIZON or not np.isfinite(raw).all():
            raise ValueError(f"{task.task_id}: expected {HORIZON} finite values")
        scale = _scale(np.asarray(task.history, dtype=float))
        for horizon_day, value in enumerate(raw, 1):
            records.append({
                "task_id": task.task_id, "fold_id": task.fold_id, "market": task.market,
                "origin_date": str(task.origin_date)[:10], "horizon_day": horizon_day,
                "prediction_raw": float(value), "prediction": float(max(0.0, value)),
                "mase7_scale": scale, "input_mode": detail["input_mode"],
            })
    return pd.DataFrame(records)


def _write(path: Path, frame: pd.DataFrame, detail: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / "predictions.parquet", index=False)
    (path / "input_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")


def _tail(values: np.ndarray, context: int | None) -> np.ndarray:
    if context is None or values.shape[-1] <= context:
        return values
    return values[..., -context:]


def run_chronos(groups, calendar, output_root: Path, device: str) -> None:
    from huggingface_hub import snapshot_download
    from chronos import Chronos2Pipeline
    entry = _registry("chronos_2")
    local = snapshot_download(entry["checkpoint"], revision=entry["checkpoint_revision"], cache_dir=str(CACHE["chronos_2"]))
    model = Chronos2Pipeline.from_pretrained(local, device_map=device)
    detail = {"input_mode": "calendar_known_future_and_27_market_cross_learning",
              "calendar_columns": CALENDAR_COLUMNS, "cross_learning": True,
              "companion_future_visitors": False}
    for group in groups:
        fold, origin = group["fold_id"].iloc[0], str(group["origin_date"].iloc[0])[:10]
        destination = output_root / "chronos_2" / f"{fold}_{origin}"
        if (destination / "predictions.parquet").is_file():
            continue
        panel, markets, start = _panel(group)
        length = panel.shape[1]
        calendar_all = _calendar_block(calendar, markets, start, length + HORIZON)
        past_dates = pd.date_range(start, periods=length, freq="D")
        future_dates = pd.date_range(past_dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        past_frames, future_frames = [], []
        for index, task in enumerate(group.itertuples(index=False)):
            past = pd.DataFrame(calendar_all[index, :length], columns=CALENDAR_COLUMNS)
            past["item_id"] = task.task_id
            past["timestamp"] = past_dates
            past["target"] = panel[index]
            future = pd.DataFrame(calendar_all[index, length:], columns=CALENDAR_COLUMNS)
            future["item_id"] = task.task_id
            future["timestamp"] = future_dates
            past_frames.append(past)
            future_frames.append(future)
        started = time.perf_counter()
        forecast = model.predict_df(
            pd.concat(past_frames, ignore_index=True),
            future_df=pd.concat(future_frames, ignore_index=True),
            prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id", timestamp_column="timestamp", target="target",
            batch_size=512, cross_learning=True, freq="D",
        )
        ordered = []
        for task in group.itertuples(index=False):
            rows = forecast.loc[forecast["item_id"] == task.task_id].sort_values("timestamp")
            column = "0.5" if "0.5" in rows.columns else "predictions"
            ordered.append(rows[column].to_numpy(float))
        _write(destination, _rows(group, np.stack(ordered), detail),
               {**detail, "seconds": time.perf_counter() - started, "history_days": int(length)})
        print(json.dumps({"model": "chronos_2", "fold": fold, "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def run_timesfm(groups, calendar, output_root: Path, device: str) -> None:
    from huggingface_hub import snapshot_download
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    entry = _registry("timesfm_3")
    local = snapshot_download(entry["checkpoint"], revision=entry["checkpoint_revision"], cache_dir=str(CACHE["timesfm_3"]))
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=local, per_core_batch_size=8, device=device))
    detail = {"input_mode": "calendar_known_future_plus_other_markets_past",
              "calendar_columns": CALENDAR_COLUMNS, "variate_cap": 32,
              "companion_keep": 24, "companion_future_visitors": False,
              "note": "TimesFM keeps at most 32 variates. One target, 7 calendar channels and 24 of 26 companion series fit. The 2 dropped companions are a fixed subsample, seed 42, inside the published evaluator."}
    for group in groups:
        fold, origin = group["fold_id"].iloc[0], str(group["origin_date"].iloc[0])[:10]
        destination = output_root / "timesfm_3" / f"{fold}_{origin}"
        if (destination / "predictions.parquet").is_file():
            continue
        panel, markets, start = _panel(group)
        length = panel.shape[1]
        calendar_all = _calendar_block(calendar, markets, start, length + HORIZON)
        started = time.perf_counter()
        forecasts = []
        for index in range(len(markets)):
            companions = np.delete(panel, index, axis=0)
            generated = list(model.predict_batch(
                contexts=[panel[index]],
                horizon=HORIZON,
                past_future_covariates=[calendar_all[index].T],
                past_only_covariates=[companions],
                return_quantiles=True,
                use_symmetric_averaging=False,
            ))
            forecasts.append(np.asarray(generated[0].forecast, dtype=float).reshape(-1)[:HORIZON])
        _write(destination, _rows(group, np.stack(forecasts), detail),
               {**detail, "seconds": time.perf_counter() - started, "history_days": int(length)})
        print(json.dumps({"model": "timesfm_3", "fold": fold, "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def run_moirai(groups, calendar, output_root: Path, device: str) -> None:
    from gluonts.dataset.common import ListDataset
    from huggingface_hub import snapshot_download
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    entry = _registry("moirai")
    local = snapshot_download(entry["checkpoint"], revision=entry["checkpoint_revision"], cache_dir=str(CACHE["moirai"]))
    module = Moirai2Module.from_pretrained(local)
    context = CONTEXT["moirai"]
    wrapper = Moirai2Forecast(module=module, prediction_length=HORIZON, context_length=context,
                              target_dim=1, feat_dynamic_real_dim=len(CALENDAR_COLUMNS),
                              past_feat_dynamic_real_dim=26)
    predictor = wrapper.create_predictor(batch_size=1, device=device)
    detail = {"input_mode": "calendar_known_future_plus_other_markets_past",
              "calendar_columns": CALENDAR_COLUMNS, "context_length": context,
              "companion_markets": 26, "companion_future_visitors": False}
    for group in groups:
        fold, origin = group["fold_id"].iloc[0], str(group["origin_date"].iloc[0])[:10]
        destination = output_root / "moirai" / f"{fold}_{origin}"
        if (destination / "predictions.parquet").is_file():
            continue
        panel, markets, start = _panel(group)
        panel = _tail(panel, context)
        length = panel.shape[1]
        window_start = start + pd.Timedelta(days=int(group["training_history_days"].iloc[0]) - length)
        calendar_all = _calendar_block(calendar, markets, window_start, length + HORIZON)
        started = time.perf_counter()
        forecasts = []
        for index, task in enumerate(group.itertuples(index=False)):
            series_start = pd.Period(window_start, freq="D")
            entry_row = {
                "item_id": task.task_id,
                "start": series_start,
                "target": panel[index],
                "feat_dynamic_real": calendar_all[index].T,
                "past_feat_dynamic_real": np.delete(panel, index, axis=0),
            }
            forecast = next(iter(predictor.predict(ListDataset([entry_row], freq="D"))))
            forecasts.append(np.asarray(forecast.quantile(0.5), dtype=float))
        _write(destination, _rows(group, np.stack(forecasts), detail),
               {**detail, "seconds": time.perf_counter() - started, "history_days": int(length)})
        print(json.dumps({"model": "moirai", "fold": fold, "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def run_unsupported(model: str, reason: str, output_root: Path) -> None:
    path = output_root / model
    path.mkdir(parents=True, exist_ok=True)
    (path / "unsupported.json").write_text(json.dumps({
        "model": model, "status": "checkpoint_cannot_accept_lightweight_inputs", "reason": reason,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "status": "unsupported"}), flush=True)


def probe_granite() -> str:
    return ("Granite PatchTST-FM-r2 maps each patch through a fixed projection of width d_patch*2 "
            "(value and observation mask). The published checkpoint has no known-future covariate "
            "channel and no channel for another market. Feeding the calendar or the other 26 series "
            "changes that projection and does not match the checkpoint.")


def probe_lag_llama() -> str:
    return ("Lag-Llama's published checkpoint uses a fixed lag set and, when time features are enabled, "
            "a fixed time-feature width from its training frequency. The network input width is part of "
            "the checkpoint. Adding the seven calendar columns or 26 companion series changes that width, "
            "so the frozen weights cannot be applied.")


def score(output_root: Path, evaluation: Path) -> pd.DataFrame:
    targets = pd.read_parquet(TARGETS, columns=["task_id", "horizon_day", "actual", "split_name", "history_length", "destination"])
    targets = targets.loc[(targets["split_name"] == "outer_backtest") & (targets["history_length"] == "all")
                          & (targets["destination"] == "Macau")]
    published = pd.read_parquet(REPO / "track1/evaluation/outer_main_table/task_metrics.parquet",
                                columns=["model", "task_id", "market", "MASE7"])
    rows = []
    for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        files = sorted(model_dir.glob("*/predictions.parquet"))
        if not files:
            continue
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        merged = frame.merge(targets[["task_id", "horizon_day", "actual"]], on=["task_id", "horizon_day"], how="inner")
        task_rows = []
        for task_id, part in merged.groupby("task_id"):
            part = part.sort_values("horizon_day")
            if len(part) != HORIZON:
                raise ValueError(f"{model_dir.name} {task_id}: {len(part)} rows")
            scale = float(part["mase7_scale"].iloc[0])
            mae = float((part["prediction"] - part["actual"]).abs().mean())
            task_rows.append({"task_id": task_id, "market": part["market"].iloc[0],
                              "MASE7": mae / scale if scale > 0 else np.nan})
        tasks = pd.DataFrame(task_rows)
        macro = float(tasks.groupby("market")["MASE7"].mean().mean())
        reference = published.loc[published["model"] == model_dir.name, ["task_id", "market", "MASE7"]]
        paired = tasks.merge(reference, on="task_id", suffixes=("_matched", "_univariate"))
        old = float(paired.groupby("market_univariate")["MASE7_univariate"].mean().mean()) if len(paired) else np.nan
        rows.append({"model": model_dir.name, "tasks": int(tasks["task_id"].nunique()),
                     "markets": int(tasks["market"].nunique()), "Macau_MASE7_matched_inputs": macro,
                     "Macau_MASE7_original_univariate": old,
                     "matched_minus_univariate": macro - old if np.isfinite(old) else np.nan})
    evaluation.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(evaluation / "summary.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["chronos_2", "timesfm_3", "moirai", "granite_patchtst_fm", "lag_llama", "score"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="outer_backtest")
    parser.add_argument("--folds", default="", help="Comma-separated fold ids. Empty means every fold.")
    parser.add_argument("--output-root", type=Path, default=REPO / "track1/results/zeroshot_matched_inputs")
    parser.add_argument("--evaluation", type=Path, default=REPO / "track1/evaluation/zeroshot_matched_inputs")
    args = parser.parse_args()
    if args.model == "score":
        print(score(args.output_root, args.evaluation).to_string(index=False))
        return
    if args.model == "granite_patchtst_fm":
        run_unsupported(args.model, probe_granite(), args.output_root)
        return
    if args.model == "lag_llama":
        run_unsupported(args.model, probe_lag_llama(), args.output_root)
        return
    folds = {item for item in args.folds.split(",") if item}
    groups = _load_groups(args.split, folds or None)
    calendar = Calendar(CALENDAR_PATH)
    runners = {"chronos_2": run_chronos, "timesfm_3": run_timesfm, "moirai": run_moirai}
    runners[args.model](groups, calendar, args.output_root, args.device)


if __name__ == "__main__":
    main()
