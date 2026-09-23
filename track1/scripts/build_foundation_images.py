"""Build one or all CUDA images using revisions recorded in the registry."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILES = {
    "timesfm_3": "track1/docker/timesfm3.Dockerfile",
    "chronos_2": "track1/docker/chronos2.Dockerfile",
    "lag_llama": "track1/docker/lag_llama.Dockerfile",
    "moirai": "track1/docker/moirai.Dockerfile",
    "granite_patchtst_fm": "track1/docker/granite.Dockerfile",
    "patchtst": "track1/docker/patchtst.Dockerfile",
    "informer": "track1/docker/informer.Dockerfile",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=sorted(DOCKERFILES))
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "track1" / "configs" / "upstream_revisions.yaml")
    args = parser.parse_args()
    config = json.loads(args.registry.read_text(encoding="utf-8"))
    selected = args.models or list(DOCKERFILES)
    for name in selected:
        revision = str(config["models"][name]["source_revision"])
        if len(revision) != 40 or revision.startswith("REQUIRED"):
            raise SystemExit(f"{name}: pin source_revision before building")
        subprocess.run([
            "docker", "build", "-f", DOCKERFILES[name],
            "--build-arg", f"UPSTREAM_COMMIT={revision}",
            "-t", f"macau-tourism/{name}:{revision[:12]}", ".",
        ], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
