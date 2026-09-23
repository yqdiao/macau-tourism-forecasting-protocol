# 澳门主样本与香港延伸：统一日度预测管道

> **七个时序模型状态：implemented/pending GPU smoke。** TimesFM 3.0、Chronos-2、Lag-Llama、Moirai 2、Granite PatchTST-FM-r2、PatchTST 与 Informer 已有统一 adapter、单起点隔离输入和独立 CUDA Dockerfile；源码与 checkpoint revision 尚待在 GPU 服务器锁定并逐个完成 smoke test，因此当前不能声称七个模型已经全部验证可运行。详见 `FOUNDATION_MODELS_GPU.md`。

## 数据边界

一次任务为“一个目的地/市场 × 一个预测起点 × 一条未来 28 天路径”。模型只读取该市场截至预测起点的历史人数。数据准备阶段生成两个物理分离的制品：

- `data/task_inputs.parquet`：模型可读，含 task_id、日期、历史数组和历史校验值，不含真实目标；
- `data/targets.parquet`：仅评估器可读，含未来 28 天真实值；
- `data/task_index.parquet`：可审阅的完整任务索引，仅含日期、折次和历史哈希；
- `data/task_manifest.json`：记录两个制品的 SHA256、任务数和协议参数。

`configs/protocol.yaml`（JSON 兼容 YAML 1.2）明确列出 A01–A13、B01–B12 和 O01–O33；每条输入带 `split_name` 与 `fold_id`。A 组仅澳门且只作疫情压力测试，B 组仅澳门且用于调参，外层为澳门主排名与香港延伸。`configs/baseline_search_spaces.yaml` 登记 13 个 baseline 的 33 个有限候选；`configs/frozen_baseline_configs.yaml` 当前为 `pending_B_group_tuning`，尚无经 B 组验证的参数赢家。

数据准备还生成 `data/splits/recovery_tuning/task_inputs.parquet` 等分组输入。B 组预测程序只接受经 manifest 校验的 B 组专用输入，不能从外层历史读取未来日期。专用输入依然不含真实目标。

预测容器不挂载 `targets.parquet`。预测完成后，独立评估器才按照 `task_id + target_date + horizon_day` 合并真实值。

## 文件

- `prepare_tasks.py`：读取授权源数据并生成清理过的数据 manifest、标准化快照和分离的 Parquet 制品；
- `data_pipeline.py`：数据验证、任务生成、稳定 task_id、历史 SHA256 和 Parquet I/O；
- `models.py`：13 个 baseline 的统一 28 天 adapter；
- `run_backtest.py`：只读 `task_inputs.parquet`，输出 raw/clipped 预测和时间记录；
- `evaluate_predictions.py`：单独读取 targets，计算 28 天任务级指标及等权宏平均；
- `compose.yaml`：预测器与评估器的分离挂载示例；
- `prepare_origin_shards.py`：把总输入拆成模型进程只能看到的单起点制品；
- `run_foundation_backtest.py`：七个时序模型的统一 28 天 GPU 运行器；
- `compose.gpu.yaml`：七个互相隔离的 CUDA 服务；
- `FOUNDATION_MODELS_GPU.md`：版本锁定、构建、smoke、批量运行和验收命令；
- `SERVER_MODEL_EXECUTION.md`：服务器拉取七个上游模型并运行的具体方案。
- `TRACK2_三模型集成_级联残差与多Agent协同实验方案.md`：在 B 组从七模型中冻结前三名后，开展确定性集成、样本外残差修正、概率校准和多 Agent 协调的第二轨实验协议。

## 源数据配置

仓库内不保存用户目录。可在命令行显式传入源数据路径，也可以配置：

```text
TOURISM_MACAU_DIR
TOURISM_HONG_KONG_WORKBOOK
```

数据 manifest 只记录类似 `macau/CN_Chinese_mainland.csv` 的逻辑来源标识和 SHA256，不记录电脑用户名或绝对路径。

澳门读取 27 个 `data,Tourism` CSV，各客源地分别建模，2017-01-01 至 2026-07-31，每条 3,499 天。香港读取工作簿的 `每日数据` 工作表，只选“内地访客”和“其他访客”两条序列，2021-01-01 至 2026-07-31，各 2,038 天。

管道拒绝重复日、缺日、空值、负数和非整数人数；真实零值保留。澳门与香港不混合训练。

## 建立隔离制品

在 PowerShell 中：

```powershell
python prepare_tasks.py `
  --macau-dir "<澳门CSV文件夹>" `
  --hong-kong-workbook "<香港工作簿.xlsx>" `
  --output data `
  --protocol configs/protocol.yaml
```

协议产生三组任务：A 组 `27×13=351`、B 组 `27×12=324`、外层 `29×33=957` 条主历史 `all` 路径。加上 180/365 天敏感性任务，共 4,896 条 28 天路径。外层首个起点为 2023-12-31，最后一个规则起点为 2026-06-14，最后目标日为 2026-07-12。`all` 表示使用起点以前全部可得历史。

生成后运行：

```powershell
python audit_protocol_artifacts.py --data data --protocol configs/protocol.yaml
python baseline_config.py --task-manifest data/task_manifest.json
```

任务 manifest 记录协议 SHA256、全任务列表 SHA256、B 组主历史任务列表 SHA256、各组数量和输入/目标文件 SHA256。只有 `frozen_baseline_configs.yaml` 完成 B 组调参并经校验后，才可把外层 baseline 输出称为论文正式排名；当前模型实现中的参数仍为硬编码 smoke 默认值。

## B01–B12 baseline 调参与冻结

`models.py` 已支持候选配置参数。先在预测环境运行；该进程仅需挂载 `data/splits/recovery_tuning/task_inputs.parquet` 和配置，不需目标文件。Windows 本机运行时建议把 BLAS 线程限制为 1，避免四个预测进程互相争用：

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python run_b_validation.py --workers 4
```

预测按“模型 × 候选 × seed × 任务”原子保存到 `results/b_validation`；同一 Git commit、搜索空间和输入 SHA 下重新执行会跳过已完成的路径。失败与尝试次数保存在 `results/b_validation/failures`。需要重试相同配置时加 `--retry-failures`。

全部 324 条 B 组主历史任务、所有候选与规定 seed 完成后，在独立评价环境运行：

```powershell
python evaluate_b_validation.py --freeze
```

评价器才读取 `targets.parquet`，先算任务级 MASE(7)，对 seed、折次和市场等权平均。覆盖不足、代码未提交或 pilot 限制任务数时不会冻结。选择结果写入 `configs/frozen_baseline_configs.yaml`，并记录配置、协议、B 任务清单和候选汇总的 SHA256。之后正式 baseline 回测须显式传入 `--frozen-config configs/frozen_baseline_configs.yaml`。

本机长时运行可调用 `scripts/run_b_local.ps1 -Workers 4`。脚本顺序执行候选预测、同配置失败重试、独立评价与有条件冻结；进度保存在 `results/b_validation/driver.log` 和 `driver_status.json`。当前电脑只有 CPU，树模型和 Transformer 的完整 B 组搜索可能需要很长时间；运行期间不要修改模型代码、搜索配置或 B 输入制品。

## 预测阶段

预测程序没有 `--targets` 参数：

```powershell
python run_backtest.py `
  --inputs data\task_inputs.parquet `
  --destinations Macau `
  --markets CN_Chinese_mainland `
  --origin-date 2023-12-31 `
  --history all `
  --splits outer_backtest `
  --models seasonal_naive,ets,sarima `
  --output results\smoke_isolated
```

输出：

- `predictions.parquet`：`prediction_raw`、非负 `prediction`、`was_clipped`、缩放分母和有效上下文；
- `timings.jsonl`：每任务模型拟合、推理和总时间；
- `run_manifest.json`：输入 SHA256、模型、seed、任务数、输出 SHA256，并明确 `target_artifact_accessed=false`。

## 独立评价阶段

预测完成并封存输出后运行：

```powershell
python evaluate_predictions.py `
  --predictions results\smoke_isolated\predictions.parquet `
  --targets data\targets.parquet `
  --output evaluation\smoke_isolated
```

输出：

- `scored_predictions.parquet`：真实值与逐日误差；
- `task_metrics.parquet`：每条 28 天路径的 MAE、RMSE、WAPE、MASE(7)、RMSSE(7)、bias 和高需求日指标；
- `aggregate_metrics.parquet`：先算任务指标，再对起点和市场等权宏平均；
- `evaluation_manifest.json`：输入和输出 SHA256。

## 容器隔离

```powershell
docker compose run --rm predictor
docker compose run --rm evaluator
```

默认 Compose 命令是协议 smoke test：澳门 `CN_Chinese_mainland`、起点 2023-12-31、`all` 历史，依次运行 Seasonal Naïve、ETS、SARIMA。三个模型共输出 84 行。完整实验应通过命令行覆盖市场、起点和模型过滤条件。

`predictor` 只有 `task_inputs.parquet` 的只读挂载；`evaluator` 才有 `targets.parquet`。正式服务器应把 targets 放在受权限保护的目录，并禁止模型容器访问原始数据根目录。

## 模型

13 个 baseline 为 Seasonal Naïve、ETS、SARIMA、Ridge、Linear Regression、Random Forest、Gradient Boosting、SVR、Lasso、Elastic Net、Parrot Random Forest 800、LightGBM 和 Local Transformer。

Local Transformer 是本地从头训练的神经网络 baseline，不是 TimesFM、Chronos、Lag-Llama、Moirai 或 Granite 等预训练时序基础模型。七个大模型采用独立容器和统一 adapter 接入，详见 `SERVER_MODEL_EXECUTION.md`。
