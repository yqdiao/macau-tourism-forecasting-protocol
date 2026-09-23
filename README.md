# 澳门主样本与香港延伸：统一日度预测管道

> **七个大模型状态：planned/pending。** TimesFM 3.0、Chronos-2、Lag-Llama、Moirai、Granite PatchTST-FM、PatchTST 与 Informer 尚未完成 adapter、独立 Dockerfile、checkpoint revision 锁定和 smoke test，当前仓库不能声称这些模型已经可以运行。现有 `configs/upstream_revisions.yaml` 只是版本登记模板。

## 数据边界

一次任务为“一个目的地/市场 × 一个预测起点 × 一条未来 28 天路径”。模型只读取该市场截至预测起点的历史人数。数据准备阶段生成两个物理分离的制品：

- `data/task_inputs.parquet`：模型可读，含 task_id、日期、历史数组和历史校验值，不含真实目标；
- `data/targets.parquet`：仅评估器可读，含未来 28 天真实值；
- `data/task_manifest.json`：记录两个制品的 SHA256、任务数和协议参数。

预测容器不挂载 `targets.parquet`。预测完成后，独立评估器才按照 `task_id + target_date + horizon_day` 合并真实值。

## 文件

- `prepare_tasks.py`：读取授权源数据并生成清理过的数据 manifest、标准化快照和分离的 Parquet 制品；
- `data_pipeline.py`：数据验证、任务生成、稳定 task_id、历史 SHA256 和 Parquet I/O；
- `models.py`：13 个 baseline 的统一 28 天 adapter；
- `run_backtest.py`：只读 `task_inputs.parquet`，输出 raw/clipped 预测和时间记录；
- `evaluate_predictions.py`：单独读取 targets，计算 28 天任务级指标及等权宏平均；
- `compose.yaml`：预测器与评估器的分离挂载示例；
- `SERVER_MODEL_EXECUTION.md`：服务器拉取七个上游模型并运行的具体方案。

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
  --stride-days 28 `
  --history 180,365,all
```

共同样本从 2023-12-31 起每隔 28 天滚动，共 33 个起点；最后一个规则起点为 2026-06-14，最后目标日为 2026-07-12。`all` 表示使用起点以前全部可得历史。

## 预测阶段

预测程序没有 `--targets` 参数：

```powershell
python run_backtest.py `
  --inputs data\task_inputs.parquet `
  --destinations Macau `
  --markets CN_Chinese_mainland `
  --origin-date 2023-12-31 `
  --history all `
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
