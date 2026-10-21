# Server execution architecture for seven time-series models

**Historical engineering guide.** Seven adapters, origin isolation, a common runner, and separate CUDA Dockerfiles were added in the September snapshot; actual GPU acceptance was still pending in that snapshot. For current release evidence, see [paper/](../../paper/README.md). Runnable entry points and acceptance checks are in the [GPU guide](FOUNDATION_MODELS_GPU.md).

## Why upstream repositories are not copied into this repository

Clone the research repository first. Each model container then downloads official source at a pinned commit. Source remains in the server's image/build cache rather than duplicating seven upstream histories here. Model weights are likewise downloaded at an exact checkpoint revision.

The research repository should retain the common tasks, adapters, evaluator, output schemas, source URLs/commits, checkpoint revisions, Dockerfiles, dependency locks, commands, applicable licenses/notices, manifests, and result summaries. This is the intended archival contract, not a claim that every field is already present.

## 1. Clone and authenticate

Use an existing deploy key, SSH key, or short-lived authorized credential:

```bash
git clone git@github.com:yqdiao/macau-tourism-forecasting-protocol.git
cd macau-tourism-forecasting-protocol
```

Credentials stay in an SSH agent, CLI credential store, or secrets manager, never in source, Dockerfiles, notebooks, or logs.

## 2. Build separate environments

Illustrative pinned-source patterns:

```dockerfile
ARG UPSTREAM_COMMIT
RUN git clone https://github.com/OWNER/REPOSITORY.git /opt/model \
    && git -C /opt/model checkout --detach ${UPSTREAM_COMMIT}
```

```dockerfile
ARG UPSTREAM_COMMIT
RUN pip install "PACKAGE_NAME @ git+https://github.com/OWNER/REPOSITORY.git@${UPSTREAM_COMMIT}"
```

Build TimesFM, Chronos, Lag-Llama, Moirai, Granite, PatchTST, and Informer separately to avoid CUDA, PyTorch, GluonTS, JAX, and other dependency conflicts. A floating `main` or `latest` is not an immutable paper version.

## 3. Cache exact weights

Download at startup or in a separate setup stage using the registered revision. Example host cache locations:

```text
/srv/tourism-model-cache/huggingface
/srv/tourism-model-cache/torch
/srv/tourism-model-cache/timesfm
```

Record checkpoint ID/revision, file SHA256, first-download time, and cached loading time. Do not commit weights.

## 4. Separate prediction from scoring

Predictor mounts:

```text
/artifacts/task_inputs.parquet   read-only, one origin shard
/outputs                         read-write
/model-cache                     read-only or a dedicated cache volume
```

No target, raw-data, later-origin, or evaluation-results directory is mounted. The predictor writes forecasts and run records, then exits.

Evaluator mounts:

```text
/outputs/predictions.parquet     read-only
/protected/targets.parquet       read-only
/evaluation                      read-write
```

Join by `task_id + target_date + horizon_day`. `track1/compose/baseline.yaml` demonstrates the baseline boundary; the seven-model workflow uses the same separation.

## 5. Common adapter responsibilities

1. Read one market's history through its origin.
2. Convert to the upstream tensor, dataframe, or tokenizer representation.
3. Generate 28 future days.
4. Use q0.5 when supplied, otherwise a sample median or the official native point forecast.
5. Save `prediction_raw`, nonnegative `prediction`, and `was_clipped`.
6. Record `training_history_days`, `requested_context_days`, and `effective_context_days`.
7. Do not access other markets, weather, holidays, or future targets in the historical univariate comparison.

## Layout and execution

```text
shared/configs/protocol.yaml
shared/data/task_inputs.parquet
shared/data/targets.parquet
shared/data/task_manifest.json
track1/configs/upstream_revisions.yaml
track1/code/foundation_models/{contracts.py,origin_io.py,adapters/}
track1/{docker,compose,scripts}/
track1/results/                  # Generated, not committed
```

The upstream registry should record URL, full commit, code license, checkpoint, exact checkpoint revision, and weight license. Replace run-blocking placeholders with approved versions before formal execution.

```bash
# Prepare artifacts in the protected preparation environment.
python track1/code/prepare_tasks.py --macau-dir /protected/raw/macau \
  --hong-kong-workbook /protected/raw/hong_kong.xlsx \
  --output shared/data --protocol shared/configs/protocol.yaml

# Predictor service cannot see targets.
docker compose -f track1/compose/baseline.yaml run --rm predictor

# Seal and verify the prediction artifact before independent scoring.
docker compose -f track1/compose/baseline.yaml run --rm evaluator
```

Model-specific services change the image and adapter while preserving the input/output contract. Actual reproducibility still requires the corresponding final versions, inputs, and configurations.
