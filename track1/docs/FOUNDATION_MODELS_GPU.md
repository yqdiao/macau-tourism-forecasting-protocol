# 七个时序模型的 GPU 服务器运行手册

当前代码已经包含统一 adapter、单起点输入隔离、七个独立 CUDA 镜像和可断点续跑的逐起点启动器。状态为 `implemented_pending_gpu_smoke`：接口和隔离测试已在本机通过，但正式权重尚未在 CUDA 服务器上完成 smoke test，因此暂时不能写成“七个模型已经全部验证可运行”。

## 统一实验边界

每个模型进程只挂载一个预测起点目录。目录中只有 `task_inputs.parquet` 和 `origin_shard_manifest.json`，没有 `targets.parquet`、原始数据或更晚预测起点的历史。输入为单市场日度历史；逻辑形状是 `[batch, context_length, 1]`，adapter 转成各上游库的原生格式。输出统一为未来 28 天，保存原始预测、非负裁剪值、中位数来源和实际上下文长度。

TimesFM 3.0、Chronos-2、Lag-Llama、Moirai 2、Granite PatchTST-FM-r2 使用固定预训练权重做零样本预测。PatchTST 和 Informer 在每个预测起点以内的历史窗口上训练，并保存每个任务的 checkpoint；训练窗口的标签也必须在该预测起点或之前结束。

## 1. 服务器准备

服务器应安装 NVIDIA 驱动、Docker Engine、Docker Compose v2 和 NVIDIA Container Toolkit。先确认：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

克隆私有研究仓库后，不要把 GitHub token、Hugging Face token或模型权重写入仓库：

```bash
git clone git@github.com:yqdiao/macau-tourism-forecasting-protocol.git
cd macau-tourism-forecasting-protocol
python -m pip install -r shared/requirements.txt
```

## 2. 建立按起点隔离的输入

在受保护的数据准备环境执行：

```bash
python track1/code/prepare_origin_shards.py \
  --inputs shared/data/task_inputs.parquet \
  --output shared/data/origin_shards
```

脚本按照 `split_name + fold_id + origin_date` 生成物理分片并记录 SHA256。模型容器必须使用这些分片，不能直接挂载包含多个起点的总输入文件。

## 3. 固定源码和权重版本

联网执行一次：

```bash
python track1/scripts/pin_upstream_revisions.py
git diff -- track1/configs/upstream_revisions.yaml
```

脚本把七个官方仓库 HEAD 和五个 Hugging Face checkpoint 解析为完整 40 位 revision，并生成本地 `track1/.env.pinned`。正式论文运行前应审查许可和版本，将 `track1/configs/upstream_revisions.yaml` 提交进 Git；环境文件可以随时从登记表重新生成。

构建单个模型，例如：

```bash
docker compose --env-file track1/.env.pinned -f track1/compose/foundation-gpu.yaml build chronos_2
```

也可以构建全部镜像：

```bash
python track1/scripts/build_foundation_images.py
```

每个 Dockerfile 在镜像中克隆官方仓库并校验 `git rev-parse HEAD` 等于登记的完整 SHA。七个依赖环境互相独立，权重下载到宿主机 `model-cache`，不提交 Git。

## 4. 单任务 smoke test

先选择 B01 分片并只运行一个模型：

```bash
export ORIGIN_SHARD_DIR="$(pwd)/shared/data/origin_shards/recovery_tuning/B01_2023-01-07"
export RUN_OUTPUT_SUBDIR="smoke/B01_2023-01-07"
export HISTORY_LENGTHS="all"
docker compose --env-file track1/.env.pinned -f track1/compose/foundation-gpu.yaml run --rm chronos_2
```

验收条件：每个任务恰好 28 行；所有 `prediction_raw` 有限；保存 `prediction` 和 `was_clipped`；`q=0.5` 或样本中位数来源明确；`history_end` 等于预测起点；manifest 显示 `target_artifact_accessed=false`；容器内不存在 targets 挂载；记录源码、权重、输入和输出 SHA。

按照 `chronos_2 → timesfm_3 → granite_patchtst_fm → moirai → lag_llama → patchtst → informer` 的顺序完成 smoke test。只有通过后才把对应状态改成 `validated_gpu_smoke`。

## 5. 逐起点批量运行

先运行 B01–B12：

```bash
python track1/scripts/run_foundation_origins.py chronos_2 \
  --splits recovery_tuning --history all
```

每个起点完成后都有独立 `run_manifest.json`。重新执行会跳过已完成起点。合并后才交给评价器：

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

七个模型的 B 组候选范围与选择规则仍需单独登记并冻结。当前 `runtime` 参数只用于 smoke test，不能直接作为论文最终参数。完成 B 组选择后，再运行 `outer_backtest`；随后运行 180/365 天敏感性和 A 组疫情压力测试。

## 6. 服务器输出归档

需要保存并回传 GitHub 的是代码、版本登记、配置和较小的汇总结果。模型权重、镜像层、缓存和大量逐任务 checkpoint 不提交 Git。正式运行还应记录镜像 digest、GPU 型号、CUDA/驱动版本、峰值显存、加载时间、训练时间、p50/p95 推理延迟、失败与重试次数及等价费用。
