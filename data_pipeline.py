"""Build leakage-isolated daily tourism forecasting artifacts.

The preparation process may read the complete research data. It writes two
physically separate artifacts:

* ``task_inputs.parquet`` contains history available through each origin only.
* ``targets.parquet`` contains the following 28 observed days.

Prediction processes must receive only the first artifact. Evaluation processes
merge predictions with the second artifact after forecasting is complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from protocol_config import Protocol, load_protocol

HERE = Path(__file__).resolve().parent
RAW_DATA_ROOT = HERE / "raw_data"
MACAU_DIR = Path(os.environ.get("TOURISM_MACAU_DIR", RAW_DATA_ROOT / "macau"))
HONG_KONG_WORKBOOK = Path(
    os.environ.get(
        "TOURISM_HONG_KONG_WORKBOOK",
        RAW_DATA_ROOT / "hong_kong" / "Hong_Kong_daily_tourism_visualization.xlsx",
    )
)
TEST_START = date(2024, 1, 1)
TEST_END = date(2026, 7, 31)
HORIZON = 28
HISTORY_LENGTHS: tuple[int | None, ...] = (180, 365, None)


@dataclass(frozen=True)
class DailySeries:
    destination: str
    market: str
    values: pd.Series
    source_id: str

    @property
    def key(self) -> str:
        return f"{self.destination}:{self.market}"


@dataclass(frozen=True)
class ForecastInputTask:
    """A model-readable task. It deliberately has no actual/target values."""

    task_id: str
    destination: str
    market: str
    origin: date
    history_length_label: str
    history_start: date
    history: np.ndarray
    target_dates: tuple[date, ...]
    split_name: str = "outer_backtest"
    fold_id: str = ""

    @property
    def key(self) -> str:
        return self.task_id


def _validate_series(destination: str, market: str, dates, values, source_id: str) -> DailySeries:
    d = pd.to_datetime(pd.Index(dates), errors="raise").normalize()
    v = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if len(d) == 0 or len(d) != len(v):
        raise ValueError(f"{destination}:{market}: empty or unmatched dates/values")
    if d.has_duplicates:
        raise ValueError(f"{destination}:{market}: duplicate dates")
    if not d.is_monotonic_increasing:
        raise ValueError(f"{destination}:{market}: dates are not increasing")
    expected = pd.date_range(d[0], d[-1], freq="D")
    if not d.equals(expected):
        raise ValueError(f"{destination}:{market}: missing calendar days")
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"{destination}:{market}: missing, infinite, or negative arrivals")
    if not np.all(v == np.floor(v)):
        raise ValueError(f"{destination}:{market}: arrivals must be whole persons")
    return DailySeries(
        destination,
        market,
        pd.Series(v.astype(np.int64), index=d, name="arrivals"),
        source_id,
    )


def load_macau(directory: Path = MACAU_DIR) -> dict[str, DailySeries]:
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if len(files) != 27:
        raise ValueError(f"Expected 27 Macau source-market CSV files, found {len(files)}")
    out: dict[str, DailySeries] = {}
    for file in files:
        frame = pd.read_csv(file, encoding="utf-8-sig")
        if list(frame.columns) != ["data", "Tourism"]:
            raise ValueError(f"{file.name}: expected data,Tourism; got {list(frame.columns)}")
        source_id = f"macau/{file.name}"
        series = _validate_series("Macau", file.stem, frame["data"], frame["Tourism"], source_id)
        out[series.key] = series
    return out


def load_hong_kong(workbook: Path = HONG_KONG_WORKBOOK) -> dict[str, DailySeries]:
    workbook = Path(workbook)
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        sheet = book["每日数据"]
        rows = sheet.iter_rows(min_row=4, max_col=6, values_only=True)
        header = next(rows)
        expected_header = (
            "日期",
            "香港居民入境人次",
            "内地访客",
            "其他访客",
            "访客入境总量（Tourism demand）",
            "全部入境旅客人次",
        )
        if tuple(header) != expected_header:
            raise ValueError(f"Hong Kong sheet headers changed: {header}")
        dates, mainland, other = [], [], []
        for row_number, row in enumerate(rows, 5):
            if all(value is None for value in row):
                continue
            day, residents, m, o, visitor_total, all_arrivals = row
            if not isinstance(day, (date, datetime)) or any(
                not isinstance(value, (int, float))
                for value in (residents, m, o, visitor_total, all_arrivals)
            ):
                raise ValueError(f"Hong Kong row {row_number}: unexpected date or missing number")
            if m + o != visitor_total or residents + visitor_total != all_arrivals:
                raise ValueError(f"Hong Kong row {row_number}: category totals do not reconcile")
            dates.append(day)
            mainland.append(m)
            other.append(o)
    finally:
        book.close()
    if not dates:
        raise ValueError("Hong Kong workbook contains no daily rows")
    source_id = f"hong_kong/{workbook.name}"
    return {
        s.key: s
        for s in (
            _validate_series("HongKong", "Mainland visitors", dates, mainland, source_id),
            _validate_series("HongKong", "Other visitors", dates, other, source_id),
        )
    }


def load_all(
    macau_dir: Path = MACAU_DIR,
    hong_kong_workbook: Path = HONG_KONG_WORKBOOK,
) -> dict[str, DailySeries]:
    out = load_macau(macau_dir)
    out.update(load_hong_kong(hong_kong_workbook))
    return out


def quality_summary(series: dict[str, DailySeries]) -> list[dict]:
    return [
        {
            "destination": item.destination,
            "market": item.market,
            "first_date": item.values.index[0].date().isoformat(),
            "last_date": item.values.index[-1].date().isoformat(),
            "days": len(item.values),
            "zero_days": int((item.values == 0).sum()),
            "source_id": item.source_id,
        }
        for item in series.values()
    ]


def rolling_origins(
    series: DailySeries,
    test_start: date = TEST_START,
    test_end: date = TEST_END,
    horizon: int = HORIZON,
    stride_days: int = 1,
    min_history_days: int = 365,
) -> tuple[date, ...]:
    """Choose origins before looking at model errors; all targets stay observed."""
    if min(horizon, stride_days, min_history_days) < 1:
        raise ValueError("horizon, stride and minimum history must be positive")
    first = max(
        test_start - timedelta(days=1),
        series.values.index[0].date() + timedelta(days=min_history_days - 1),
    )
    last = min(
        test_end - timedelta(days=horizon),
        series.values.index[-1].date() - timedelta(days=horizon),
    )
    if first > last:
        return ()
    return tuple(
        first + timedelta(days=n)
        for n in range(0, (last - first).days + 1, stride_days)
    )


def _task_id(
    destination: str,
    market: str,
    origin: date,
    history_label: str,
    horizon: int,
) -> str:
    identity = "\x1f".join(
        (destination, market, origin.isoformat(), history_label, str(horizon))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def make_input_task(
    series: DailySeries,
    origin: date,
    history_length: int | None = None,
    horizon: int = HORIZON,
) -> ForecastInputTask:
    if horizon < 1 or (history_length is not None and history_length < 1):
        raise ValueError("horizon and history length must be positive")
    index = series.values.index
    end = index.get_indexer([pd.Timestamp(origin)])[0]
    if end < 0:
        raise ValueError(f"{series.key}: origin {origin} is outside the data")
    if end + horizon >= len(index):
        raise ValueError(f"{series.key}: targets after {origin} are not observed")
    start = 0 if history_length is None else end - history_length + 1
    if start < 0:
        raise ValueError(f"{series.key}: insufficient history for {history_length} days at {origin}")
    history = series.values.iloc[start : end + 1].to_numpy(dtype=float).copy()
    if len(history) < 28:
        raise ValueError("history window is incomplete")
    label = "all" if history_length is None else str(history_length)
    target_dates = tuple(origin + timedelta(days=h) for h in range(1, horizon + 1))
    return ForecastInputTask(
        _task_id(series.destination, series.market, origin, label, horizon),
        series.destination,
        series.market,
        origin,
        label,
        index[start].date(),
        history,
        target_dates,
    )


def _minimum_history(history_lengths: tuple[int | None, ...]) -> int:
    finite = [value for value in history_lengths if value is not None]
    return max(finite, default=365)


def iter_input_tasks(
    series: dict[str, DailySeries],
    *,
    stride_days: int = 28,
    history_lengths: tuple[int | None, ...] = HISTORY_LENGTHS,
    destinations: tuple[str, ...] = ("Macau", "HongKong"),
) -> Iterator[ForecastInputTask]:
    minimum = _minimum_history(history_lengths)
    for item in series.values():
        if item.destination not in destinations:
            continue
        for origin in rolling_origins(
            item, stride_days=stride_days, min_history_days=minimum
        ):
            for length in history_lengths:
                yield make_input_task(item, origin, length)


def iter_protocol_tasks(
    series: dict[str, DailySeries], protocol: Protocol,
    destinations: tuple[str, ...] = ("Macau", "HongKong"),
) -> Iterator[ForecastInputTask]:
    """Use only the explicitly listed folds; A/B never include Hong Kong."""
    for item in series.values():
        if item.destination not in destinations:
            continue
        for fold in protocol.folds:
            if item.destination not in fold.destinations:
                continue
            for length in protocol.history_lengths:
                base = make_input_task(item, fold.origin, length, protocol.horizon_days)
                yield ForecastInputTask(
                    base.task_id, base.destination, base.market, base.origin,
                    base.history_length_label, base.history_start, base.history,
                    base.target_dates, fold.split_name, fold.fold_id,
                )


def make_task(*args, **kwargs) -> ForecastInputTask:
    """Compatibility alias. Returned tasks never contain observed targets."""
    return make_input_task(*args, **kwargs)


def iter_tasks(*args, **kwargs) -> Iterator[ForecastInputTask]:
    """Compatibility alias for model-input tasks only."""
    return iter_input_tasks(*args, **kwargs)


def _target_values(series: DailySeries, origin: date, horizon: int) -> np.ndarray:
    end = series.values.index.get_indexer([pd.Timestamp(origin)])[0]
    if end < 0 or end + horizon >= len(series.values):
        raise ValueError(f"{series.key}: unavailable target window after {origin}")
    return series.values.iloc[end + 1 : end + horizon + 1].to_numpy(dtype=np.int64)


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def write_task_artifacts(
    series: dict[str, DailySeries],
    folder: Path,
    *,
    protocol: Protocol | None = None,
    destinations: tuple[str, ...] = ("Macau", "HongKong"),
) -> dict:
    """Write model inputs and hidden targets as separate Parquet artifacts."""
    folder = Path(folder)
    protocol = protocol or load_protocol()
    input_rows: list[dict] = []
    target_rows: list[dict] = []
    for task in iter_protocol_tasks(series, protocol, destinations):
        input_rows.append(
            {
                "task_id": task.task_id,
                "split_name": task.split_name,
                "fold_id": task.fold_id,
                "destination": task.destination,
                "market": task.market,
                "origin_date": task.origin,
                "history_length": task.history_length_label,
                "history_start": task.history_start,
                "history_end": task.origin,
                "training_history_days": len(task.history),
                "requested_context_days": (
                    None if task.history_length_label == "all" else int(task.history_length_label)
                ),
                "effective_context_days": len(task.history),
                "horizon_days": len(task.target_dates),
                "target_start_date": task.target_dates[0],
                "target_end_date": task.target_dates[-1],
                "history_sha256": _array_sha256(task.history),
                "history": task.history.astype(np.int64).tolist(),
            }
        )
        item = series[f"{task.destination}:{task.market}"]
        actual = _target_values(item, task.origin, len(task.target_dates))
        for horizon_day, (target_date, value) in enumerate(
            zip(task.target_dates, actual), 1
        ):
            target_rows.append(
                {
                    "task_id": task.task_id,
                    "split_name": task.split_name,
                    "fold_id": task.fold_id,
                    "destination": task.destination,
                    "market": task.market,
                    "origin_date": task.origin,
                    "history_length": task.history_length_label,
                    "target_date": target_date,
                    "horizon_day": horizon_day,
                    "actual": int(value),
                }
            )
    if not input_rows:
        raise ValueError("No forecast tasks generated")
    inputs = pd.DataFrame(input_rows)
    targets = pd.DataFrame(target_rows)
    if inputs["task_id"].duplicated().any():
        raise ValueError("Duplicate task_id in model inputs")
    target_counts = targets.groupby("task_id").size()
    if not target_counts.eq(protocol.horizon_days).all():
        raise ValueError("Every task must have exactly 28 hidden targets")
    if not (pd.to_datetime(inputs["history_end"]) == pd.to_datetime(inputs["origin_date"])).all():
        raise ValueError("History extends beyond a task origin")
    if not (pd.to_datetime(inputs["target_start_date"]) == pd.to_datetime(inputs["origin_date"]) + pd.Timedelta(days=1)).all():
        raise ValueError("Target start does not immediately follow its origin")
    if not (pd.to_datetime(inputs["target_end_date"]) == pd.to_datetime(inputs["origin_date"]) + pd.Timedelta(days=protocol.horizon_days)).all():
        raise ValueError("Target end does not match the horizon")
    input_path = folder / "task_inputs.parquet"
    target_path = folder / "targets.parquet"
    index_path = folder / "task_index.parquet"
    index_columns = [
        "task_id", "split_name", "fold_id", "destination", "market", "origin_date",
        "history_length", "history_start", "history_end", "training_history_days",
        "target_start_date", "target_end_date", "horizon_days", "history_sha256",
    ]
    _write_parquet_atomic(inputs[index_columns], index_path)
    _write_parquet_atomic(inputs, input_path)
    _write_parquet_atomic(targets, target_path)
    task_index = "\n".join(
        f"{row.split_name}\t{row.fold_id}\t{row.task_id}"
        for row in inputs[["split_name", "fold_id", "task_id"]].itertuples(index=False)
    ) + "\n"
    b_main = inputs.loc[
        (inputs["split_name"] == "recovery_tuning") & (inputs["history_length"] == "all"),
        ["split_name", "fold_id", "task_id"],
    ]
    b_task_index = "\n".join(
        f"{row.split_name}\t{row.fold_id}\t{row.task_id}"
        for row in b_main.itertuples(index=False)
    ) + "\n"
    split_counts = {str(key): int(value) for key, value in inputs.groupby("split_name").size().items()}
    manifest = {
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "task_list_sha256": hashlib.sha256(task_index.encode("utf-8")).hexdigest(),
        "B_main_task_list_sha256": hashlib.sha256(b_task_index.encode("utf-8")).hexdigest(),
        "task_count": len(inputs),
        "split_task_counts": split_counts,
        "target_row_count": len(targets),
        "horizon_days": protocol.horizon_days,
        "history_lengths": [value if value is not None else "all" for value in protocol.history_lengths],
        "destinations": list(destinations),
        "artifacts": {
            "task_index.parquet": {
                "sha256": _file_sha256(index_path),
                "contains_history_values": False,
                "contains_targets": False,
            },
            "task_inputs.parquet": {
                "sha256": _file_sha256(input_path),
                "contains_targets": False,
            },
            "targets.parquet": {
                "sha256": _file_sha256(target_path),
                "model_access": "forbidden",
            },
        },
    }
    temporary = folder / ".task_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, folder / "task_manifest.json")
    return manifest


def read_task_inputs(path: Path) -> Iterator[ForecastInputTask]:
    """Load model-readable inputs. This function has no targets argument."""
    frame = pd.read_parquet(path, engine="pyarrow")
    required = {
        "task_id",
        "destination",
        "market",
        "origin_date",
        "history_length",
        "split_name",
        "fold_id",
        "history_start",
        "horizon_days",
        "history_sha256",
        "history",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"task inputs missing columns: {sorted(missing)}")
    if frame["task_id"].duplicated().any():
        raise ValueError("task inputs contain duplicate task_id")
    for row in frame.itertuples(index=False):
        history = np.asarray(row.history, dtype=float)
        if _array_sha256(history) != row.history_sha256:
            raise ValueError(f"{row.task_id}: history SHA256 mismatch")
        horizon = int(row.horizon_days)
        origin = pd.Timestamp(row.origin_date).date()
        yield ForecastInputTask(
            str(row.task_id),
            str(row.destination),
            str(row.market),
            origin,
            str(row.history_length),
            pd.Timestamp(row.history_start).date(),
            history,
            tuple(origin + timedelta(days=h) for h in range(1, horizon + 1)),
            str(row.split_name),
            str(row.fold_id),
        )


def export_normalized_jsonl(series: dict[str, DailySeries], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item in series.values():
            for day, value in item.values.items():
                stream.write(
                    json.dumps(
                        {
                            "destination": item.destination,
                            "market": item.market,
                            "date": day.date().isoformat(),
                            "arrivals": int(value),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    os.replace(temporary, path)


def source_manifest(
    macau_dir: Path = MACAU_DIR,
    hong_kong_workbook: Path = HONG_KONG_WORKBOOK,
) -> dict[str, str]:
    files = [*sorted(Path(macau_dir).glob("*.csv")), Path(hong_kong_workbook)]
    logical_ids = [*(f"macau/{path.name}" for path in files[:-1]), f"hong_kong/{files[-1].name}"]
    return {
        source_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for source_id, path in zip(logical_ids, files)
    }


def write_data_snapshot(
    folder: Path,
    macau_dir: Path = MACAU_DIR,
    hong_kong_workbook: Path = HONG_KONG_WORKBOOK,
    protocol: Protocol | None = None,
) -> dict:
    protocol = protocol or load_protocol()
    series = load_all(macau_dir, hong_kong_workbook)
    folder.mkdir(parents=True, exist_ok=True)
    export_normalized_jsonl(series, folder / "normalized_daily.jsonl")
    report = {
        "series_count": len(series),
        "macau_series": 27,
        "hong_kong_series": 2,
        "horizon_days": protocol.horizon_days,
        "test_start": TEST_START.isoformat(),
        "test_end": TEST_END.isoformat(),
        "history_lengths": [180, 365, "all"],
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "quality": quality_summary(series),
        "source_sha256": source_manifest(macau_dir, hong_kong_workbook),
        "source_paths_included": False,
    }
    temporary = folder / ".data_manifest.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, folder / "data_manifest.json")
    return report
