"""Resolve immutable Hugging Face snapshots without putting weights in Git."""
from __future__ import annotations

from pathlib import Path


def resolve_snapshot(repo_id: str, revision: str, cache_dir: Path) -> str:
    if len(revision) != 40:
        raise ValueError("a full Hugging Face revision SHA is required")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - model container dependency
        raise RuntimeError("huggingface_hub is required to resolve model weights") from exc
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
    )
