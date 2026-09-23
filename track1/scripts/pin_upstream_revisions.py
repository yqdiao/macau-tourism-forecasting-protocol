"""Resolve upstream Git heads and Hugging Face snapshots, then write immutable SHAs."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]

def git_head(repository: str) -> str:
    output = subprocess.check_output(["git", "ls-remote", repository, "HEAD"], text=True)
    revision = output.split()[0]
    if len(revision) != 40:
        raise RuntimeError(f"could not resolve {repository}")
    return revision


ENV_NAMES = {
    "timesfm_3": "TIMESFM_COMMIT", "chronos_2": "CHRONOS_COMMIT",
    "lag_llama": "LAG_LLAMA_COMMIT", "moirai": "MOIRAI_COMMIT",
    "granite_patchtst_fm": "GRANITE_COMMIT", "patchtst": "PATCHTST_COMMIT",
    "informer": "INFORMER_COMMIT",
}


def pin(path: Path, compose_env: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    api = HfApi()
    for entry in config["models"].values():
        entry["source_revision"] = git_head(entry["repository"])
        if entry.get("checkpoint"):
            entry["checkpoint_revision"] = api.model_info(entry["checkpoint"]).sha
        entry["adapter_status"] = "implemented_pending_gpu_smoke"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    compose_env.write_text("\n".join(
        f"{ENV_NAMES[name]}={entry['source_revision']}"
        for name, entry in config["models"].items()
    ) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "track1" / "configs" / "upstream_revisions.yaml")
    parser.add_argument("--compose-env", type=Path, default=REPO_ROOT / "track1" / ".env.pinned")
    args = parser.parse_args()
    pin(args.registry, args.compose_env)
    print(f"Pinned revisions in {args.registry} and wrote {args.compose_env}; review and commit the registry before a paper run.")


if __name__ == "__main__":
    main()
