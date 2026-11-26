"""Secondary fault-injection comparison on Macau O01–O33."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, HORIZON, MEMBERS, RESULTS, TRACK1, load_targets, write_json

SEED = 20261005
RATES = (0.01, 0.05, 0.10)
FALLBACK = json.loads((CONFIG / "track2_fallback_policy.yaml").read_text(encoding="utf-8"))
WEIGHTS = np.array([json.loads((CONFIG / "track2_ensemble_config.yaml").read_text())["weights"][model] for model in MEMBERS])


def _load_pattern(root: Path) -> pd.DataFrame:
    files = list(root.rglob("predictions.parquet"))
    if not files:
        raise FileNotFoundError(root)
    frame = pd.read_parquet(files[0])
    frame = frame[frame["destination"] == "Macau"]
    return frame.sort_values(["market", "horizon_day"])


def _patterns() -> dict[str, dict[str, np.ndarray]]:
    inf = _load_pattern(TRACK1 / "foundation_b_search" / "informer" / "INF01" / "seed42")
    llama = _load_pattern(TRACK1 / "foundation_smoke" / "recovery_tuning" / "B01_2023-01-07" / "lag_llama")
    patterns = {}
    for label, frame in (("R1", inf), ("R2", llama)):
        patterns[label] = {
            market: part["prediction"].to_numpy(float)
            for market, part in frame.groupby("market")
            if len(part) >= HORIZON
        }
    return patterns


def _schedule(tasks: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED)
    hard = ["R1", "R2", "R3", "R4"]
    soft = ["S1", "S2", "S3", "S4", "S5"]
    rows = []
    order = tasks[["task_id", "market", "fold_id"]].drop_duplicates().sort_values("task_id")
    identifiers = order["task_id"].tolist()
    markets = dict(zip(order["task_id"], order["market"]))
    folds = dict(zip(order["task_id"], order["fold_id"]))
    for rate in RATES:
        count = int(round(rate * len(identifiers)))
        chosen = rng.choice(len(identifiers), size=count, replace=False)
        for index in chosen:
            task_id = identifiers[int(index)]
            family = hard if rng.random() < 0.5 else soft
            fault = str(rng.choice(family))
            member = str(rng.choice(MEMBERS))
            rows.append({"rate": rate, "task_id": task_id, "market": markets[task_id], "fold_id": folds[task_id], "fault": fault, "member": member})
    return rows


def _inject(member_rows: dict[str, np.ndarray], fault: str, market: str, patterns: dict, member: str) -> tuple[dict[str, np.ndarray | None], bool]:
    """Return possibly failed member vectors. None means hard failure."""
    output = {model: values.copy() for model, values in member_rows.items()}
    target = member if member in output else MEMBERS[0]
    detected_by_structure = False
    if fault == "R1":
        output[target] = np.asarray(patterns["R1"][market][:HORIZON], dtype=float)
    elif fault == "R2":
        output[target] = np.asarray(patterns["R2"][market][:HORIZON], dtype=float)
    elif fault in {"R3", "R4", "S1", "S5"}:
        output[target] = None
        detected_by_structure = True
    elif fault == "S2":
        output[target][0] = np.nan
        detected_by_structure = True
    elif fault == "S3":
        output[target] = output[target][:27]
        detected_by_structure = True
    elif fault == "S4":
        detected_by_structure = True
        output[target] = None
    return output, detected_by_structure


def _level_ok(values: np.ndarray | None, recent: float) -> bool:
    if values is None or len(values) != HORIZON or not np.isfinite(values).all():
        return False
    if recent < FALLBACK["recent28_skip_below"]:
        return True
    ratio = float(np.mean(values) / recent)
    return FALLBACK["r_low"] <= ratio <= FALLBACK["r_high"]


def _deliver(vectors: dict[str, np.ndarray | None], recent: float, naive: np.ndarray, system: str) -> tuple[np.ndarray | None, bool, bool]:
    """Return prediction, detected, delivered_ok."""
    if system == "T2-A0p":
        values = vectors[MEMBERS[0]]
        if values is None or len(values) != HORIZON or not np.isfinite(values).all():
            return None, False, False
        return np.maximum(0.0, values), False, True
    if system == "T2-C2":
        values = vectors[MEMBERS[0]]
        if _level_ok(values, recent):
            return np.maximum(0.0, values), False, True
        return np.maximum(0.0, naive), True, True
    usable = [model for model in MEMBERS if _level_ok(vectors[model], recent)]
    if not usable:
        return np.maximum(0.0, naive), True, True
    if usable == MEMBERS:
        weights = WEIGHTS
        path_models = MEMBERS
    else:
        prior = np.array([WEIGHTS[MEMBERS.index(model)] for model in usable])
        path_models = usable
        weights = prior / prior.sum() if prior.sum() > 0 else np.full(len(usable), 1.0 / len(usable))
    acc = np.zeros(HORIZON)
    for weight, model in zip(weights, path_models):
        acc += weight * vectors[model]
    detected = usable != MEMBERS
    return np.maximum(0.0, acc), detected, True


def main() -> None:
    base = pd.read_parquet(RESULTS / "outer_macau" / "T2-01" / "predictions.parquet")
    members = {}
    for model, label in zip(MEMBERS, ("T2-01",)):
        pass
    raw = {}
    for model in MEMBERS:
        frame = pd.read_parquet(TRACK1 / "foundation_outer_default" / "O01_2023-12-31" / model / "predictions.parquet")
    from common import load_predictions
    frame = load_predictions(TRACK1 / "foundation_outer_default", MEMBERS, destination="Macau")
    seasonal = pd.concat((pd.read_parquet(path) for path in (TRACK1 / "outer_server_all" / "predictions" / "seasonal_naive" / "SN01" / "42").glob("*.parquet")), ignore_index=True)
    seasonal = seasonal[seasonal["destination"] == "Macau"]
    scales = pd.read_parquet(RESULTS / "outer_macau" / "T2-01" / "task_mase.parquet")[["task_id", "market", "fold_id"]]
    recent = frame.groupby("task_id")["prediction"].mean()  # placeholder replaced below
    from common import history_scales
    recent_map = history_scales("outer_backtest", "all", "Macau").set_index("task_id")["recent28_mean"].to_dict()
    actuals = load_targets("outer_backtest", "all", "Macau")
    scale_map = frame.groupby("task_id")["mase7_scale"].first().to_dict()
    patterns = _patterns()
    schedule = _schedule(frame)
    output = RESULTS / "fault_injection"
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(schedule, ensure_ascii=False, indent=2)
    (output / "schedule.json").write_text(text + "\n", encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (output / "schedule_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    write_json(CONFIG / "fault_sources.yaml", {
        "R1": "track1/results/foundation_b_search/informer/INF01/seed42",
        "R2": "track1/results/foundation_smoke/recovery_tuning/B01_2023-01-07/lag_llama/predictions.parquet",
        "note": "R1 and R2 replace M1 with the archived 28-day pattern of the same market. R3, R4, S1, S4 and S5 are hard member failures.",
    })
    by_task = {task_id: part for task_id, part in frame.groupby("task_id")}
    naive_by_task = {task_id: part.sort_values("horizon_day")["prediction"].to_numpy(float) for task_id, part in seasonal.groupby("task_id")}
    actual_by_task = {task_id: part.sort_values("horizon_day")["actual"].to_numpy(float) for task_id, part in actuals.groupby("task_id")}
    rows = []
    systems = ["T2-A0p", "T2-C2", "T2-A1"]
    faulted = {(row["rate"], row["task_id"]): row for row in schedule}
    task_ids = sorted(by_task)
    for rate in RATES:
        for system in systems:
            delivered = 0
            detected = 0
            false_pos = 0
            silent = 0
            injected = 0
            mases = []
            for task_id in task_ids:
                parts = by_task[task_id]
                vectors = {
                    model: parts[parts["model"] == model].sort_values("horizon_day")["prediction"].to_numpy(float)
                    for model in MEMBERS
                }
                event = faulted.get((rate, task_id))
                is_fault = event is not None
                if is_fault:
                    injected += 1
                    member = event["member"] if system == "T2-A1" else MEMBERS[0]
                    vectors, _structure = _inject(vectors, event["fault"], event["market"], patterns, member)
                prediction, was_detected, ok = _deliver(vectors, float(recent_map[task_id]), naive_by_task[task_id], system)
                if ok:
                    delivered += 1
                if is_fault and was_detected:
                    detected += 1
                if is_fault and ok and not was_detected:
                    silent += 1
                if not is_fault and was_detected:
                    false_pos += 1
                if ok:
                    actual = actual_by_task[task_id]
                    scale = float(scale_map[task_id])
                    mases.append(float(np.mean(np.abs(prediction - actual)) / scale))
            rows.append({
                "system": system,
                "rate": rate,
                "tasks": len(task_ids),
                "injected": injected,
                "delivery_rate": delivered / len(task_ids),
                "detection_rate": detected / injected if injected else None,
                "false_positive_rate": false_pos / (len(task_ids) - injected),
                "silent_failure_rate": silent / injected if injected else None,
                "mase": float(np.mean(mases)) if mases else None,
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "fault_summary.csv", index=False)
    print(table.to_string(index=False))
    print("schedule", digest)


if __name__ == "__main__":
    main()
