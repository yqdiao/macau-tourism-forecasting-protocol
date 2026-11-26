"""Shared paths, loading, and MASE aggregation for Track 2."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRACK1 = ROOT / "track1" / "results"
DATA = ROOT / "shared" / "data"
CONFIG = ROOT / "track2" / "configs"
RESULTS = ROOT / "track2" / "results"

MEMBERS = ["granite_patchtst_fm", "chronos_2", "timesfm_3"]
TOP7 = MEMBERS + ["moirai", "patchtst", "lag_llama", "informer"]
EPS = 1e-6
MIN_PRIOR = 3
HORIZON = 28
IMPROVE = 0.01
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTEXTS = (28, 56, 84)
MAX_FRAC = 0.25
TIE = 1e-4

KEYS = ["destination", "market", "origin_date", "horizon_day", "fold_id", "task_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = ROOT / "track2" / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def clip_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(values, dtype=float))


def macro_mase(task_scores: pd.DataFrame) -> float:
    """Market-equal mean of within-market fold means. One row per task."""
    if task_scores.empty:
        return float("nan")
    by_market = task_scores.groupby("market")["mase"].mean()
    return float(by_market.mean())


def task_scores_from_long(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["destination", "market", "fold_id", "task_id"]
    for identity, part in frame.groupby(group_cols, sort=False):
        part = part.sort_values("horizon_day")
        error = np.abs(part[prediction_column].to_numpy(float) - part["actual"].to_numpy(float))
        scale = float(part["mase7_scale"].iloc[0])
        mase = float(error.mean() / scale) if scale > 0 else float("nan")
        row = dict(zip(group_cols, identity))
        row["mase"] = mase
        row["mae"] = float(error.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_inverse_mase(scores: dict[str, float]) -> dict[str, float]:
    raw = {model: 1.0 / (score + EPS) for model, score in scores.items()}
    total = sum(raw.values())
    return {model: value / total for model, value in raw.items()}


def fit_constrained_least_squares(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str]:
    """Exact nonnegative least squares on the simplex for three members.

    SLSQP is not used: visitor counts make its line search report failure
    even at a feasible point. With three weights the vertices, edges, and
    interior stationary point can be checked directly.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    width = design.shape[1]
    if width != 3:
        raise ValueError("constrained least squares is implemented for 3 members")
    candidates: list[np.ndarray] = []
    for index in range(width):
        weights = np.zeros(width)
        weights[index] = 1.0
        candidates.append(weights)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        direction = design[:, left] - design[:, right]
        denom = float(direction @ direction)
        if denom == 0:
            share = 0.0
        else:
            share = float(np.clip(((target - design[:, right]) @ direction) / denom, 0.0, 1.0))
        weights = np.zeros(width)
        weights[left] = share
        weights[right] = 1.0 - share
        candidates.append(weights)
    gram = design.T @ design
    linear = design.T @ target
    system = np.zeros((4, 4))
    system[:3, :3] = gram
    system[:3, 3] = 1.0
    system[3, :3] = 1.0
    rhs = np.zeros(4)
    rhs[:3] = linear
    rhs[3] = 1.0
    try:
        solution = np.linalg.solve(system, rhs)
        interior = solution[:3]
        if np.all(np.isfinite(interior)) and interior.min() >= -1e-8:
            interior = np.clip(interior, 0.0, None)
            interior = interior / interior.sum()
            candidates.append(interior)
    except np.linalg.LinAlgError:
        pass

    def loss(weights: np.ndarray) -> float:
        residual = target - design @ weights
        return float(residual @ residual)

    best = min(candidates, key=loss)
    if abs(loss(best) - loss(np.full(width, 1.0 / width))) <= 1e-6 * max(1.0, loss(best)):
        return np.full(width, 1.0 / width), "equal"
    return best, "ok"


def apply_weights(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return clip_nonnegative(matrix @ weights)


def prediction_files(base: Path, models: list[str]) -> list[tuple[str, Path]]:
    """Return (model, path) for either fold/model or model/seed/fold layouts."""
    found: list[tuple[str, Path]] = []
    if not base.exists():
        return found
    direct = [path for path in base.iterdir() if path.is_dir() and "_" in path.name[:4]]
    if direct and (direct[0] / models[0] / "predictions.parquet").exists():
        for fold in sorted(direct):
            for model in models:
                path = fold / model / "predictions.parquet"
                if path.is_file():
                    found.append((model, path))
        return found
    for model in models:
        model_dir = base / model
        if not model_dir.exists():
            continue
        for path in sorted(model_dir.glob("seed*/**/predictions.parquet")):
            found.append((model, path))
    return found


def load_predictions(base: Path, models: list[str], destination: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, path in prediction_files(base, models):
        frame = pd.read_parquet(path)
        frame["model"] = model
        if destination is not None and "destination" in frame.columns:
            frame = frame[frame["destination"] == destination]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no predictions under {base} for {models}")
    combined = pd.concat(frames, ignore_index=True)
    combined["origin_date"] = pd.to_datetime(combined["origin_date"])
    combined["target_date"] = pd.to_datetime(combined["target_date"])
    return combined


def load_b_targets() -> pd.DataFrame:
    """B-group actuals only. Outer rows are not loaded."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(DATA / "targets.parquet")
    table = dataset.to_table(filter=ds.field("split_name") == "recovery_tuning")
    frame = table.to_pandas()
    frame = frame[(frame["destination"] == "Macau") & (frame["history_length"].astype(str) == "all")]
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


_TARGET_CACHE: dict[tuple, pd.DataFrame] = {}
_SCALE_CACHE: dict[tuple, pd.DataFrame] = {}


def load_targets(split_name: str, history: str, destination: str | None = None) -> pd.DataFrame:
    key = (split_name, str(history), destination)
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key].copy()
    import pyarrow.dataset as ds

    dataset = ds.dataset(DATA / "targets.parquet")
    table = dataset.to_table(filter=ds.field("split_name") == split_name)
    frame = table.to_pandas()
    frame = frame[frame["history_length"].astype(str) == str(history)]
    if destination is not None:
        frame = frame[frame["destination"] == destination]
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    _TARGET_CACHE[key] = frame
    return frame.copy()


def history_scales(split_name: str, history: str, destination: str) -> pd.DataFrame:
    """Recompute s7 from the task history that scoring is allowed to see."""
    key = (split_name, str(history), destination)
    if key in _SCALE_CACHE:
        return _SCALE_CACHE[key].copy()
    import pyarrow.dataset as ds

    dataset = ds.dataset(DATA / "task_inputs.parquet")
    table = dataset.to_table(
        filter=(ds.field("split_name") == split_name) & (ds.field("history_length") == str(history)),
        columns=["task_id", "destination", "market", "fold_id", "history"],
    )
    frame = table.to_pandas()
    frame = frame[frame["destination"] == destination]
    scales = []
    recent = []
    for task_id, history_values in zip(frame["task_id"], frame["history"]):
        series = np.asarray(list(history_values), dtype=float)
        if len(series) <= 7:
            scale = np.nan
        else:
            scale = float(np.mean(np.abs(series[7:] - series[:-7])))
        tail = series[-28:] if len(series) >= 28 else series
        scales.append(scale)
        recent.append(float(np.mean(tail)) if len(tail) else np.nan)
    frame["s7_recomputed"] = scales
    frame["recent28_mean"] = recent
    kept = frame[["task_id", "s7_recomputed", "recent28_mean"]].copy()
    _SCALE_CACHE[key] = kept
    return kept.copy()
