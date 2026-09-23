"""Run one registered model against one physically isolated forecast origin."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # Windows development host
    resource = None

from data_pipeline import read_task_inputs
from foundation_models.adapters import ADAPTER_PATHS, create_adapter
from foundation_models.contracts import AdapterSettings, SeriesInput
from foundation_models.origin_io import file_sha256, validate_single_origin_input, write_json_atomic

HORIZON = 28


def _registry_entry(path: Path, model: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    entry = value["models"].get(model)
    if not entry:
        raise ValueError(f"{model} is not present in {path}")
    if entry.get("adapter_status") == "pending":
        raise ValueError(f"{model} adapter is still registered as pending")
    for key in ("source_revision",):
        revision = str(entry.get(key, ""))
        if len(revision) != 40 or revision.startswith("REQUIRED"):
            raise ValueError(f"{model}.{key} must be pinned before execution")
    if entry.get("checkpoint"):
        revision = str(entry.get("checkpoint_revision", ""))
        if len(revision) != 40 or revision.startswith("REQUIRED"):
            raise ValueError(f"{model}.checkpoint_revision must be pinned before execution")
    return entry


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def run(*, model_name: str, inputs: Path, output: Path, registry: Path,
        context_length: int | None, batch_size: int, num_samples: int,
        device: str, seed: int, history_lengths: tuple[str, ...]) -> dict:
    validate_single_origin_input(inputs)
    entry = _registry_entry(registry, model_name)
    tasks = [task for task in read_task_inputs(inputs)
             if not history_lengths or task.history_length_label in history_lengths]
    if not tasks:
        raise ValueError("origin shard contains no tasks")
    if any(len(task.target_dates) != HORIZON for task in tasks):
        raise ValueError("all model tasks must have a 28-day horizon")

    output.mkdir(parents=True, exist_ok=True)
    runtime = entry.get("runtime", {})
    settings = AdapterSettings(
        checkpoint=entry.get("checkpoint"),
        checkpoint_revision=entry.get("checkpoint_revision"),
        source_revision=str(entry["source_revision"]),
        device=device,
        context_length=context_length if context_length is not None else runtime.get("context_length"),
        batch_size=batch_size,
        num_samples=num_samples,
        seed=seed,
        cache_dir=Path(os.environ.get("HF_HOME", "/model-cache/huggingface")),
        checkpoint_dir=output / "checkpoints",
        extra=dict(runtime.get("extra", {})),
    )
    adapter = create_adapter(model_name, settings)
    load_start = time.perf_counter()
    adapter.load()
    load_seconds = time.perf_counter() - load_start

    try:
        import torch
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    rows, timing_rows = [], []
    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[start:start + batch_size]
        batch_inputs = [SeriesInput(task.task_id, task.origin, task.history) for task in batch_tasks]
        inference_start = time.perf_counter()
        outputs = adapter.predict_batch(batch_inputs, HORIZON)
        elapsed = time.perf_counter() - inference_start
        if len(outputs) != len(batch_tasks):
            raise ValueError("adapter output count differs from input count")
        per_task = elapsed / len(batch_tasks)
        for task, result in zip(batch_tasks, outputs):
            raw = np.asarray(result.raw_prediction, dtype=float)
            clipped = np.maximum(0.0, raw)
            scale_values = task.history[7:] - task.history[:-7]
            mase_scale = float(np.mean(np.abs(scale_values))) or None
            rmsse_scale = float(np.sqrt(np.mean(scale_values ** 2))) or None
            threshold = float(np.quantile(task.history, 0.9))
            for index, (target_date, raw_value, prediction) in enumerate(
                zip(task.target_dates, raw, clipped), 1
            ):
                row = {
                    "task_id": task.task_id, "split_name": task.split_name,
                    "fold_id": task.fold_id, "destination": task.destination,
                    "market": task.market, "origin_date": task.origin,
                    "history_length": task.history_length_label,
                    "history_start": task.history_start,
                    "training_history_days": len(task.history),
                    "effective_context_days": result.effective_context_days,
                    "target_date": target_date, "horizon_day": index,
                    "model": model_name, "candidate_id": None, "seed": seed,
                    "prediction_raw": float(raw_value), "prediction": float(prediction),
                    "was_clipped": bool(raw_value < 0), "mase7_scale": mase_scale,
                    "rmsse7_scale": rmsse_scale, "high_demand_threshold": threshold,
                    "point_source": result.point_source,
                }
                for name, values in result.quantiles.items():
                    row[name] = float(np.asarray(values)[index - 1])
                rows.append(row)
            timing_rows.append({
                "task_id": task.task_id, "model": model_name,
                "load_seconds_shared": load_seconds, "inference_seconds": per_task,
                "training_mode": adapter.training_mode, "model_detail": result.detail,
            })

    predictions = pd.DataFrame(rows)
    expected = len(tasks) * HORIZON
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} output rows, got {len(predictions)}")
    prediction_path = output / "predictions.parquet"
    _atomic_parquet(predictions, prediction_path)
    timing_path = output / "timings.jsonl"
    temporary = output / ".timings.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in timing_rows), encoding="utf-8")
    os.replace(temporary, timing_path)

    peak_gpu_mb = None
    if torch is not None and device.startswith("cuda"):
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_rss_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                   if resource is not None else None)
    manifest = {
        "schema_version": 1, "kind": "foundation_model_predictions",
        "model": model_name, "training_mode": adapter.training_mode,
        "source_repository": entry["repository"],
        "source_revision": entry["source_revision"],
        "checkpoint": entry.get("checkpoint"),
        "checkpoint_revision": entry.get("checkpoint_revision"),
        "device": device, "seed": seed, "batch_size": batch_size,
        "history_lengths": list(history_lengths),
        "requested_context_days": context_length or runtime.get("context_length") or "all",
        "forecast_tasks": len(tasks), "prediction_rows": len(predictions),
        "load_seconds": load_seconds, "peak_gpu_memory_mb": peak_gpu_mb,
        "peak_process_rss_mb": peak_rss_mb,
        "input_artifact": {"name": inputs.name, "sha256": file_sha256(inputs)},
        "registry_sha256": file_sha256(registry), "target_artifact_accessed": False,
        "outputs": {"predictions.parquet": file_sha256(prediction_path),
                    "timings.jsonl": file_sha256(timing_path)},
    }
    write_json_atomic(manifest, output / "run_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(ADAPTER_PATHS))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/upstream_revisions.yaml"))
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", default="all", help="Comma-separated all,180,365")
    args = parser.parse_args()
    print(json.dumps(run(model_name=args.model, inputs=args.inputs, output=args.output,
                         registry=args.registry, context_length=args.context_length,
                         batch_size=args.batch_size, num_samples=args.num_samples,
                         device=args.device, seed=args.seed,
                         history_lengths=tuple(x.strip() for x in args.history.split(",") if x.strip())),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
