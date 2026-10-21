# GPU execution guide for seven time-series models

**Historical implementation guide.** The September repository snapshot contained common adapters, isolated origin shards, seven separate CUDA environments, and an origin launcher. Its recorded status was `implemented_pending_gpu_smoke`: local interface/isolation tests passed, but this repository did not yet contain formal CUDA smoke acceptance for each model. This status describes that snapshot; use the [current paper inventory](../../paper/README.md) to distinguish later evidence from code included here.

## Common boundary

Each predictor mounts one origin directory containing only `task_inputs.parquet` and `origin_shard_manifest.json`. It must not mount targets, raw data, or later-origin histories. Inputs are single-market daily series, logically `[batch, context_length, 1]`; adapters translate this to native APIs. Outputs cover 28 days and retain raw values, nonnegative values, point-source metadata, and effective context.

TimesFM 3.0, Chronos-2, Lag-Llama, Moirai 2, and Granite PatchTST-FM-r2 use frozen pretrained weights in this design. PatchTST and Informer train on windows entirely within each origin's history, with task checkpoints. Every training label must end by the forecast origin.

## 1. Server prerequisites

Install a compatible NVIDIA driver, Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
git clone git@github.com:yqdiao/macau-tourism-forecasting-protocol.git
cd macau-tourism-forecasting-protocol
python -m pip install -r shared/requirements.txt
```

Keep GitHub/Hugging Face credentials and weights outside Git. The basic CUDA probe is not proof that every model image is driver-compatible; check each Dockerfile.

## 2. Create origin-isolated inputs

In the protected preparation environment:

```bash
python track1/code/prepare_origin_shards.py \
  --inputs shared/data/task_inputs.parquet \
  --output shared/data/origin_shards
```

The script partitions by `split_name + fold_id + origin_date` and records SHA256. GPU containers use these shards, not the multi-origin input artifact.

## 3. Pin source and checkpoints

For a new, explicitly versioned run:

```bash
python track1/scripts/pin_upstream_revisions.py
git diff -- track1/configs/upstream_revisions.yaml
```

This resolves seven source repositories and five checkpoints to full 40-character revisions and generates local `track1/.env.pinned`. Review licenses and versions and commit the approved registry before the run. Do not repin during replication of an already frozen experiment.

Build one environment, or all environments:

```bash
docker compose --env-file track1/.env.pinned \
  -f track1/compose/foundation-gpu.yaml build chronos_2
python track1/scripts/build_foundation_images.py
```

Dockerfiles clone official source and verify its commit. Environments are separate; weights are cached on the host and excluded from Git.

## 4. Smoke one origin

```bash
export ORIGIN_SHARD_DIR="$(pwd)/shared/data/origin_shards/recovery_tuning/B01_2023-01-07"
export RUN_OUTPUT_SUBDIR="smoke/B01_2023-01-07"
export HISTORY_LENGTHS="all"
docker compose --env-file track1/.env.pinned \
  -f track1/compose/foundation-gpu.yaml run --rm chronos_2
```

Acceptance requires exactly 28 aligned rows per task; finite raw forecasts; retained clipped values and flags; explicit q0.5/sample-median/native-point source; effective context; history ending by the origin; `target_artifact_accessed=false`; no target mount; source/checkpoint/input/output hashes.

Suggested sequential smoke order: `chronos_2`, `timesfm_3`, `granite_patchtst_fm`, `moirai`, `lag_llama`, `patchtst`, `informer`. Mark `validated_gpu_smoke` only after recorded acceptance.

## 5. Origin-level runs and scoring

Historical launcher entry point:

```bash
python track1/scripts/run_foundation_origins.py chronos_2 \
  --splits recovery_tuning --history all
```

**Resume scope:** the original launcher recognizes an origin by an existing `run_manifest.json`; it does not independently disambiguate every candidate, seed, and history setting. Use separate output roots and audit resume behavior before formal multi-configuration runs. See the [sequential checklist](GPU_SEQUENTIAL_RUN_CHECKLIST.md).

Merge only a root containing the intended model/split/history/configuration:

```bash
python track1/code/combine_foundation_outputs.py \
  --root track1/results/foundation \
  --model chronos_2 \
  --output track1/results/combined/chronos_2_B
python track1/code/evaluate_predictions.py \
  --predictions track1/results/combined/chronos_2_B/predictions.parquet \
  --targets shared/data/targets.parquet \
  --output track1/evaluation/chronos_2_B
```

Only the evaluator sees targets. Register and freeze B-group candidate ranges and selection rules separately; `runtime` smoke defaults are not final paper parameters. After freezing, run the outer set, then separately labeled 180/365-day and stress analyses.

## 6. Archive outputs

Retain source/version registries, frozen configurations, environment and image digests, small summaries, coverage, and failures. Archive raw predictions, logs, and checkpoints with hashes outside Git. Record GPU/CUDA/driver versions, peak memory, loading/training/inference time, p50/p95 latency, retries, and the precise resource-to-price mapping actually measured. A requested telemetry field is not evidence that it was measured.
