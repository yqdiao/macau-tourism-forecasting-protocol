"""Build one or all CUDA images using revisions recorded in the registry."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DOCKERFILES = {
    "timesfm_3": "docker/timesfm3.Dockerfile",
    "chronos_2": "docker/chronos2.Dockerfile",
    "lag_llama": "docker/lag_llama.Dockerfile",
    "moirai": "docker/moirai.Dockerfile",
    "granite_patchtst_fm": "docker/granite.Dockerfile",
    "patchtst": "docker/patchtst.Dockerfile",
    "informer": "docker/informer.Dockerfile",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=sorted(DOCKERFILES))
    parser.add_argument("--registry", type=Path, default=Path("configs/upstream_revisions.yaml"))
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
        ], check=True)


if __name__ == "__main__":
    main()
