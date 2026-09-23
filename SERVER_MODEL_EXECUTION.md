# 服务器运行七个时序模型的方法

> 实现状态：七个 adapter、单起点输入分片、统一运行器和独立 CUDA Dockerfile 已加入仓库，但尚未在实际 GPU 上完成逐模型 smoke test。可执行命令和验收条件见 `FOUNDATION_MODELS_GPU.md`。

## 为什么不需要把七个上游仓库复制进本仓库

服务器首先克隆本研究仓库，然后在构建每个模型的独立容器时，从官方仓库下载**锁定 commit** 的源码。源码存在服务器的容器镜像或构建缓存中，不进入本研究仓库的 Git 历史。模型权重同样按锁定的 checkpoint revision 下载到服务器缓存。

本研究仓库负责保存：

- 统一任务、adapter、评估器和输出 schema；
- 每个上游仓库 URL 与 commit SHA；
- checkpoint 名称与 revision；
- Dockerfile、依赖锁和运行命令；
- 许可证与 NOTICE 副本；
- 实验 manifest 和结果摘要。

## 服务器工作流

### 1. 克隆私有研究仓库

在服务器配置 GitHub deploy key、SSH key 或短期 token，然后运行：

```bash
git clone git@github.com:yqdiao/macau-tourism-forecasting-protocol.git
cd macau-tourism-forecasting-protocol
```

认证信息只保存在服务器的 SSH agent、GitHub CLI 或 secrets manager 中，不写入仓库、Dockerfile 或 notebook。

### 2. 构建各模型的独立环境

每个模型使用独立 Dockerfile。构建文件采用以下两种方式之一取得官方源码：

```dockerfile
ARG UPSTREAM_COMMIT
RUN git clone https://github.com/OWNER/REPOSITORY.git /opt/model \
    && git -C /opt/model checkout --detach ${UPSTREAM_COMMIT}
```

如果官方仓库是可安装 Python 包，也可以锁定 commit：

```dockerfile
ARG UPSTREAM_COMMIT
RUN pip install "PACKAGE_NAME @ git+https://github.com/OWNER/REPOSITORY.git@${UPSTREAM_COMMIT}"
```

TimesFM、Chronos、Lag-Llama、Moirai、Granite、PatchTST、Informer 分别构建镜像，避免 CUDA、PyTorch、GluonTS、JAX 和其他依赖互相冲突。不得使用 `main` 或 `latest` 作为论文运行版本。

### 3. 下载锁定的模型权重

镜像启动时或预下载步骤使用 checkpoint revision 获取权重。服务器挂载共享只读缓存，例如：

```text
/srv/tourism-model-cache/huggingface
/srv/tourism-model-cache/torch
/srv/tourism-model-cache/timesfm
```

运行 manifest 记录 checkpoint、revision、文件 SHA256、首次下载时间和缓存后加载时间。权重不提交 Git。

### 4. 严格分开预测和评价

预测容器仅挂载：

```text
/artifacts/task_inputs.parquet   read-only
/outputs                         read-write
/model-cache                     read-only 或专用缓存卷
```

预测容器不挂载 `targets.parquet`、原始数据目录或评价结果目录。它生成 `predictions.parquet` 和运行记录后退出。

评价容器随后挂载：

```text
/outputs/predictions.parquet     read-only
/protected/targets.parquet       read-only
/evaluation                      read-write
```

评价容器按照 `task_id + target_date + horizon_day` 合并真实值。`compose.yaml` 已给出 baseline 的挂载示例；七个大模型复用同一边界。

### 5. 七个 adapter 的统一责任

每个 adapter 只做以下转换：

1. 从 `task_inputs.parquet` 读取一个市场截至起点的历史数组；
2. 转换为模型原生张量、数据框或 tokenizer 输入；
3. 生成未来 28 天；
4. 分位数模型取 `q=0.5`，采样模型取样本中位数；
5. 保存 `prediction_raw`、非负 `prediction` 与 `was_clipped`；
6. 记录 `training_history_days`、`requested_context_days` 与 `effective_context_days`；
7. 不读取其他市场、天气、节假日或 targets。

## 推荐的仓库结构

```text
configs/
  protocol.yaml
  upstream_revisions.yaml
foundation_models/
  contracts.py
  origin_io.py
  adapters/
    timesfm3.py
    chronos2.py
    lag_llama.py
    moirai.py
    granite.py
    patchtst.py
    informer.py
docker/
  baseline.Dockerfile
  timesfm.Dockerfile
  chronos.Dockerfile
  ...
data/
  task_inputs.parquet
  targets.parquet
  task_manifest.json
results/
scripts/
```

`upstream_revisions.yaml` 应记录 URL、完整 commit SHA、代码许可、checkpoint、checkpoint revision 和权重许可。第一次正式运行前把全部占位符替换为实际版本并冻结。

## GPU 服务器上的执行顺序

```bash
# 1. 在数据准备环境生成隔离制品
python prepare_tasks.py --macau-dir /protected/raw/macau \
  --hong-kong-workbook /protected/raw/hong_kong.xlsx \
  --output data --protocol configs/protocol.yaml

# 2. 运行某个预测镜像；该容器看不到 targets
docker compose run --rm predictor

# 3. 确认 predictions.parquet 完整并记录 SHA256 后再评价
docker compose run --rm evaluator
```

七个大模型的 predictor 服务只需替换镜像和 adapter，输入与输出契约保持一致。这样服务器每次连接本研究仓库后都能重建相同环境，同时避免在研究仓库中复制七份不断变化的第三方源码。
