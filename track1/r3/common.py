"""Shared paths, hashes, and CPU limits for the R3 light-model study."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/DYQ/macau-tourism-forecasting-protocol")
RUN_ID = "r3_20261018"
CODE_DIR = REPO / "track1/r3"
OUT = REPO / "track1/results/light_r3" / RUN_ID
TASK_INPUTS = REPO / "shared/data/task_inputs.parquet"
TARGETS = REPO / "shared/data/targets.parquet"
CALENDAR = REPO / "shared/data/holiday_calendar.parquet"
TASK_MANIFEST = REPO / "shared/data/task_manifest.json"
ORIGINAL_CODE = REPO / "track1/code/light_models.py"
BASELINE_ROOT = REPO / "track1/results/light_models"
SEEDS_LINEAR = (11, 29, 42)
SEED_TREE = 42
HORIZON = 28
MARKETS = 27


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(CODE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(temporary)
    os.replace(temporary, path)


def write_json(path: Path, body: dict) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))


def limit_cpu(torch_threads: int = 2, blas_threads: int = 1) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    try:
        import torch
        torch.set_num_threads(torch_threads)
    except Exception:
        pass
