# 澳门日度旅游需求：七个时序仓库与 13 个基准模型的统一 GPU 实验方案

> **2026-09-23 决策更新：** 主实验采用 TimesFM 3.0、Chronos-2、Lag-Llama、Moirai、Granite TimeSeries PatchTST-FM、PatchTST 和 Informer。主输入为单市场单变量日度历史，输出为未来 28 天；每个预测起点使用此前全部可得历史并采用 expanding window，180/365 天降为敏感性分析。执行清单见 [统一实验方案](./澳门日度旅游需求_统一实验方案与执行清单.md)。

## 1. 本文档采用的口径

用户提供的《时序大模型七个开源库.md》是仓库资料来源，其中关于“分钟级股票、订单簿、未来 30 分钟”的内容不属于本项目要求。本方案只采用其中列出的仓库、接口、依赖和维护状态，并把任务改为已经固定的澳门及香港日度旅游需求预测。

现有数据与回测协议保持不变：

- 澳门 27 条客源市场序列，香港 2 条类别序列，共 29 条日序列；
- 每个任务为“一个目的地/市场 × 一个预测起点 × 未来 28 天完整路径”；
- 测试目标日期为 2024-01-01 至 2026-07-31；
- 每日滚动共有 916 个预测起点；
- 共同计算样本从 2023-12-31 起每隔 28 天取一个起点，共 33 个起点；
- 主实验使用截至每个预测起点的全部可得历史；180 天和 365 天作为历史长度敏感性分析；
- 主实验只给模型该市场在预测起点及以前的访客量，不给未来实际值、其他市场、事后天气或活动标签；
- 预测结果统一为非负访客量，同时保留裁剪前预测值和裁剪标记。

当前 13 个基准配置是：Seasonal Naive、ETS、SARIMA、Ridge、Linear Regression、Random Forest、Gradient Boosting、SVR、Lasso、Elastic Net、Parrot Random Forest 800、LightGBM 和本地小型 Transformer。最后一个是本地训练的编码器回归基准，不是预训练时序基础模型。

## 2. 可行性结论

把代码、环境定义、统一适配器和实验配置放在一个 GitHub 仓库中完全可行。所有模型共用一个 Python 环境不可行，因为 TimesFM、Chronos、Lag-Llama、Informer、PatchTST 和 DeepLOB 的 PyTorch、TensorFlow、GluonTS、CUDA 与 Python 版本要求不同。

推荐使用“一个 GitHub 单仓库 + 每个模型一个锁定的容器环境 + 一个统一调度层”：

1. GitHub 保存数据管道、模型适配器、Dockerfile、锁定依赖、实验配置和汇总结果；
2. 每个模型构建独立 Docker 镜像，并推送到 GitHub Container Registry；
3. 原始数据、模型权重、缓存和大体量逐点预测不提交 Git，运行时通过只读数据卷和输出卷挂载；
4. 任意装有兼容 NVIDIA 驱动和 NVIDIA Container Toolkit 的 Linux GPU 主机，可拉取同一镜像并运行同一命令；
5. GitHub Actions 负责格式检查、单元测试、输出格式检查和 CPU 小样本；GPU 正式实验在云 GPU、实验室服务器或自托管 runner 上运行；
6. 每次运行生成不可覆盖的 `run_id`，写出配置、版本、预测、指标、耗时、资源、失败和费用记录，可断点续跑。

“随时连接 GPU”仍受两个条件限制：宿主机驱动必须支持镜像所用 CUDA，显存必须满足所选权重和批量大小。容器固定用户态 CUDA，但不能替代宿主机 NVIDIA 驱动。

## 3. 七个仓库能否进入旅游需求主比较

| 仓库 | 实际性质 | 日度旅游需求适用性 | 主实验处理 |
|---|---|---:|---|
| TimesFM | 预训练时序基础模型 | 高 | 进入零样本主比较；研究使用可选 3.0，另保留 2.5 复现配置 |
| Chronos | 预训练时序基础模型 | 高 | 进入零样本主比较；建议 Chronos-2 |
| Lag-Llama | 概率时序基础模型 | 中高 | 进入零样本主比较；单独锁定旧依赖 |
| PatchTST | 需要本地训练的 Transformer | 高 | 进入“本地训练深度模型”比较 |
| Informer2020 | 需要本地训练的 Transformer | 中高 | 进入“本地训练深度模型”比较 |
| Amazon Forecast Samples | 云服务示例库，不是本地模型 | 低 | 仅已有 Amazon Forecast 客户可做可选云服务实验，不计作开源大模型 |
| DeepLOB | 订单簿方向分类 CNN/TCN | 不适用 | 不进入旅游需求准确率排行榜；保留排除原因和环境审计 |

### 3.1 为什么 Amazon Forecast 不能作为普通开源模型

该仓库演示通过 AWS API 导入数据、训练 predictor 和产生预测，本身没有一个可下载、可本地推理的模型权重。AWS 已停止向新客户开放 Amazon Forecast，示例仓库也已归档。因此它只能是“云服务基准”，并且只有已有服务权限时才可运行。其 API、训练、存储、导出和失败重试费用应单独记录。

### 3.2 为什么 DeepLOB 不应直接改造成旅游模型

DeepLOB 的输入是多档订单簿张量，输出是价格方向类别；本项目的输入是一条日度访客量序列，输出是 28 个连续访客量。替换输入层和输出头后重新训练得到的是一个新的 CNN/TCN 回归模型，不能再声称复现了 DeepLOB。可把该仓库保存在工程清单中，并在排除表中记录 `domain_mismatch`。

### 3.3 若论文需要“7 个可比较模型”的推荐集合

建议用下面七个实际能产生同口径 28 天旅游需求预测的模型：

1. TimesFM 3.0 零样本；
2. Chronos-2 零样本；
3. Lag-Llama 零样本概率预测；
4. Moirai 零样本；
5. Granite TimeSeries PatchTST-FM 零样本；
6. PatchTST 本地训练；
7. Informer 本地训练。

这样仍保留用户选定的五个适用仓库，并用 Moirai 与 Granite 替代不能公平比较的 Amazon Forecast Samples 和 DeepLOB。若必须展示原始七个仓库，应称为“七个候选仓库”，准确率主表中 Amazon Forecast 和 DeepLOB 标为 `N/A`，不得填入虚构结果。

### 3.4 TimesFM 版本选择

TimesFM 3.0 权重使用非商业许可；TimesFM 2.5 权重是 Apache-2.0。论文研究可在完成许可检查后使用 3.0，但运行清单必须记录权重许可、模型卡 URL 和精确 revision。为了长期复现，建议同时保留 `timesfm_2_5` 环境配置，但不把两个版本同时算作两个不同模型来凑数量。

## 4. 推荐仓库结构

```text
tourism-tsfm-benchmark/
├─ README.md
├─ LICENSES.md
├─ CITATION.cff
├─ Makefile
├─ pyproject.toml                 # 统一调度层，不安装各大模型
├─ uv.lock
├─ src/tourism_benchmark/
│  ├─ data/
│  │  ├─ build_dataset.py
│  │  ├─ splits.py
│  │  └─ validation.py
│  ├─ adapters/
│  │  ├─ base.py
│  │  ├─ baselines.py
│  │  ├─ timesfm.py
│  │  ├─ chronos.py
│  │  ├─ lag_llama.py
│  │  ├─ patchtst.py
│  │  ├─ informer.py
│  │  ├─ moirai.py
│  │  └─ granite_tsfm.py
│  ├─ evaluation/
│  │  ├─ metrics.py
│  │  ├─ statistical_tests.py
│  │  └─ aggregate.py
│  ├─ telemetry/
│  │  ├─ timers.py
│  │  ├─ resources.py
│  │  ├─ costs.py
│  │  └─ provenance.py
│  └─ runner.py
├─ configs/
│  ├─ protocol.yaml
│  ├─ models/*.yaml
│  ├─ hardware/*.yaml
│  └─ prices/*.yaml
├─ environments/
│  ├─ core/Dockerfile
│  ├─ timesfm/Dockerfile
│  ├─ chronos/Dockerfile
│  ├─ lag_llama/Dockerfile
│  ├─ patchtst/Dockerfile
│  ├─ informer/Dockerfile
│  ├─ moirai/Dockerfile
│  ├─ granite_tsfm/Dockerfile
│  ├─ amazon_forecast/Dockerfile
│  └─ deeplob/Dockerfile
├─ upstream/
│  └─ revisions.yaml              # 仓库 URL、commit SHA、模型 revision、许可
├─ scripts/
│  ├─ build_images.ps1
│  ├─ run_smoke.ps1
│  ├─ run_common_sample.ps1
│  └─ resume_run.ps1
├─ tests/
│  ├─ test_no_leakage.py
│  ├─ test_adapter_contract.py
│  ├─ test_metrics.py
│  └─ test_output_schema.py
├─ data/
│  └─ README.md                   # 只说明数据挂载和校验值
└─ results/
   └─ runs/<run_id>/
      ├─ run_manifest.json
      ├─ predictions/*.parquet
      ├─ task_metrics.parquet
      ├─ aggregate_metrics.parquet
      ├─ timings.jsonl
      ├─ resources.jsonl
      ├─ costs.parquet
      ├─ failures.jsonl
      └─ logs/
```

不建议把七个上游仓库的当前 `main` 分支直接复制进主仓库。`upstream/revisions.yaml` 应固定 URL 与 commit SHA；需要改动上游代码时，用明确补丁或 Git submodule，并记录补丁哈希。模型权重按精确 revision 下载至共享只读缓存。

## 5. 统一模型适配器

每个模型容器只需要实现同一个命令行契约：

```text
forecast \
  --input /work/tasks/task_batch.parquet \
  --config /work/configs/models/chronos_2.yaml \
  --output /work/results/<run_id>/model=chronos_2
```

输入每行至少包含：

| 字段 | 含义 |
|---|---|
| `task_id` | 由目的地、市场、起点、历史设置确定的稳定 ID |
| `destination` | Macau 或 HongKong |
| `market` | 客源市场或游客类别 |
| `origin_date` | 最后一个可观察日期 |
| `history_label` | 主实验为 all_available_expanding；敏感性分析为 180 或 365 |
| `history_start` | 实际历史起点 |
| `history_values` | 截至 origin 的日度访客量数组 |
| `forecast_horizon` | 固定为 28 |
| `frequency` | D |

主实验输出至少包含：

| 字段 | 含义 |
|---|---|
| `run_id, task_id, model_id, attempt` | 追踪与重试 |
| `target_date, horizon_day` | 目标日期与 1–28 步 |
| `actual` | 仅由评估器在预测完成后合并 |
| `prediction_raw` | 模型原始点预测 |
| `prediction` | 统一裁剪为不小于 0 的点预测 |
| `q_0_1, q_0_5, q_0_9` | 模型提供时保存分位数 |
| `sample_id, sample_value` | 模型提供样本时另存长表 |
| `effective_context_days` | 模型真正读取的历史长度 |
| `batch_size, device, precision` | 推理配置 |

真实值不能写入模型输入文件。评估器应在全部预测文件落盘后按 `task_id + target_date` 合并实际值，以形成结构性防泄漏。

## 6. 公平比较规则

### 6.1 主实验输入

主排行榜采用单变量目标历史。即使 TimesFM 3.0 或 Chronos-2 支持多变量与协变量，也不能在主表中额外使用其他市场、天气、节假日或未来协变量。多变量和协变量作为第二个能力实验单独报告。

GPU 批处理可以同时计算多个 task，但不同市场的值不能通过模型上下文发生交互。若模型会跨 item 做 in-context learning，应逐 task 推理，或显式建立相互隔离的 group。

### 6.2 点预测与概率预测

- 有官方中位数的模型使用 `q=0.5` 作为点预测；
- 只有预测样本的模型使用样本中位数；
- 只有官方点预测的模型使用该点预测；
- 所有模型同时保存裁剪前和裁剪后的值；
- 概率指标只在能产生合法分布或分位数的模型子集内比较，不能因此删除点预测基准。

### 6.3 历史长度限制

主实验在每个预测起点使用此前全部可得历史。需要区分“用于生成训练窗口的全部历史”和“模型最后一次前向预测读取的上下文”。所有模型记录 `training_history_days`、`requested_context_days` 和 `effective_context_days`。如果模型因架构上限截断最后的输入上下文，应使用最近的可支持历史并如实记录；180 天和 365 天只作为敏感性分析。

### 6.4 本地训练模型

PatchTST 和 Informer需要训练，TimesFM、Chronos、Lag-Llama、Moirai、Granite 的主配置采用零样本推理。为与现有每任务拟合的基准严格对齐，PatchTST/Informer 在每个市场、预测起点和历史长度上只使用该任务的历史训练；固定超参数后再进入测试。

调参只允许使用 2024-01-01 以前的校准区间，不能根据测试期结果选择超参数。训练内部可从历史末端划出验证段用于 early stopping，但验证段必须早于预测起点。训练失败后的重试使用同一配置；任何更改超参数的重跑算新的 `model_id`。

### 6.5 随机性

- 零样本确定性点预测运行一次；概率采样固定 seed 并保存样本数；
- PatchTST 和 Informer建议用 3 个预先固定 seed，报告均值及 seed 间标准差；
- 保存 Python、NumPy、PyTorch/TensorFlow seed 与确定性开关；
- CUDA 算子仍可能存在非确定性时，在 manifest 中明确标记。

## 7. 规模和分阶段运行

共同主样本中，每个模型有：

```text
29 条序列 × 33 个起点 × 1 个 expanding-history 设置 = 957 个预测任务
957 × 28 = 26,796 个逐日预测值
20 个模型配置 = 19,140 个任务、535,920 个逐日预测值
```

若加入 180/365 天敏感性分析，20 个模型总计为 57,420 个任务和 1,607,760 个逐日预测值。

完整每日滚动中，每个模型有：

```text
29 × 916 × 3 = 79,692 个预测任务
20 个模型配置 = 1,593,840 个任务、44,627,520 个逐日预测值
```

建议阶段：

1. **接口 smoke test**：1 个市场 × 1 个起点 × 1 个历史长度；检查形状、日期、非负处理和泄漏；
2. **工程 pilot**：3 个预先指定市场 × 3 个预先指定起点 × 3 个历史长度；测显存、批量大小、失败率和单任务时间；
3. **共同样本**：29 × 33 × 3；所有可比较模型必须完成，用于论文主表；
4. **完整每日滚动**：先运行 13 个基准和零样本基础模型；PatchTST/Informer 是否扩展至全部 916 个起点，由 pilot 的预算估算决定；
5. **稳健性分析**：历史长度、目的地、市场规模、高需求日、疫情/复苏阶段和概率预测分开报告。

不能在看到模型结果后更换共同样本起点。Pilot 只决定工程参数和是否有资源进入下一阶段。

## 8. 准确性指标

### 8.1 MASE

对每个 28 天任务，星期季节周期 `s=7`：

```text
MASE = mean_h(|y_h - yhat_h|) /
       mean_{t=8..T}(|y_t - y_{t-7}|)
```

分母只能用该预测起点之前的历史。聚合时先计算每个任务的 MASE，再在任务间取均值、中位数和四分位数；同时报告无法计算分母的任务数。不要把所有任务的误差和分母先整体相加后再构造另一个未声明指标。

### 8.2 RMSSE

```text
RMSSE = sqrt(
  mean_h((y_h - yhat_h)^2) /
  mean_{t=8..T}((y_t - y_{t-7})^2)
)
```

### 8.3 其他指标

- MAE；
- MASE、RMSSE；
- bias，即 `mean(prediction - actual)`；
- 高需求日 MASE；
- 高需求日平均低估量 `mean(max(actual - prediction, 0))`；
- 高需求日低估率；
- 分位数覆盖率与平均区间宽度；
- CRPS：模型能输出预测样本或完整分布时计算；只有少数分位数时改报 pinball loss/WIS，并写明近似方式；
- 分预测步长 `h=1, 7, 14, 28` 以及完整 28 天路径；
- 各市场、澳门合计组、香港类别和全体的平均、中位数、IQR、最差 10% 与相对 Seasonal Naive 的胜率。

真实零值较多，MAPE 不作为主要指标。R² 也不适合作为多步滚动预测主指标，可放入附录但不参与主排名。

### 8.4 统计比较

- 使用配对区块 bootstrap，对“市场 × 起点”的损失差给出置信区间；
- 对逐起点损失做 Diebold–Mariano 检验时，使用适合 28 步重叠预测的 HAC 方差；
- 多模型整体比较可用 Friedman 检验与事后成对比较，并控制多重检验；
- 同时报告效应量和置信区间，不能只报告 p 值。

## 9. 时间与资源记录

### 9.1 时间拆分

每个模型至少记录：

1. 首次下载权重时间；
2. 缓存存在时的冷启动和首次加载时间；
3. 数据预处理时间；
4. 训练或拟合时间；
5. 预热后单任务推理时间；
6. 后处理与写盘时间；
7. 单次推理延迟 p50、p95；
8. 澳门全部共同样本的总 wall time；
9. 成功任务吞吐量 tasks/second。

权重下载不混入常规推理延迟。GPU 计时前后调用同步操作，避免只测到异步提交时间。先运行固定数量 warm-up，再开始记录。批量推理需要记录批量大小；除单任务延迟外，再报告实际批处理吞吐量。

### 9.2 资源

- 模型权重文件实际字节数；
- Docker 镜像压缩和解压后大小；
- 峰值进程 RSS；
- 峰值 GPU allocated、reserved 和设备总显存占用；
- GPU 型号、显存、驱动、CUDA、cuDNN；
- CPU 型号、物理核数、RAM、操作系统；
- PyTorch/TensorFlow、核心依赖版本；
- 可选记录平均功率、峰值功率和能耗 Wh。

资源采样写入带时间戳的 `resources.jsonl`。峰值 GPU 指标在每个模型运行前重置，不能沿用同一进程此前模型的峰值。

## 10. 费用计算

费用统一换算为每 1,000 个**成功完成**任务：

```text
cost_per_1000 = 1000 × total_cost_including_failed_attempts / successful_tasks
```

其中：

```text
total_cost = API 实际费用
           + 训练实例小时 × 同日单价
           + 推理实例小时 × 同日单价
           + 存储/流量费用
           + 失败和重试消耗的实例与 API 费用
```

费用表至少保留：provider、region、instance_type、currency、price_per_hour、price_date、billed_seconds、successful_tasks、failed_attempts、training_cost、inference_cost、retry_cost、storage_cost、total_cost 和 cost_per_1000。

本地 GPU 按一个预先选定的等价云实例折算，并同时报告真实运行秒数。所有模型用同一云厂商、地区、计价方式和价格记录日期；不能给不同模型选择对自己更有利的价格。共享的模型加载时间分别报告“包含冷启动”和“已缓存稳态”两种口径。

PatchTST/Informer 的训练是任务计算的一部分；零样本模型的预训练成本不计入，因为研究者没有发生该成本。调参计算费用单独报告，不能埋入最终一次训练费用。研究者工时独立报告，不换算为货币。

## 11. 实施负担记录

使用 `implementation_log.csv` 人工记录：

| 字段 | 内容 |
|---|---|
| `model_id` | 模型配置 |
| `activity` | 环境安装、适配、调参、故障排查等 |
| `started_at, ended_at` | 实际开始与结束时间 |
| `active_minutes` | 研究者实际投入时间，不含无人值守训练 |
| `person` | 执行者 |
| `issue_url` | 对应 issue 或日志 |
| `outcome` | 成功、失败、放弃及原因 |

另由程序自动记录：

- 是否必须使用 GPU，以及实际运行设备；
- 从全新环境到首次正确预测的 wall time；
- 调参次数、调参 GPU 时间；
- 运行任务总数、失败数、重试数、最终缺失数；
- 外部 API 数量与名称；
- 自动化代理或独立服务数量；
- 模型和镜像大小；
- 峰值 RAM/VRAM。

这些字段分别报告，不合成主观复杂度总分。

## 12. 运行清单与复现信息

每个 `run_manifest.json` 应自动写入：

- `run_id`、开始/结束 UTC 时间、命令行和完整 YAML 配置；
- 本基准仓库 commit SHA 和 dirty 状态；
- 上游仓库 URL 与 commit SHA；
- Hugging Face 模型 ID、精确 revision 和权重校验值；
- Docker image digest；
- 输入数据 SHA256、序列数量、日期范围；
- 预测起点列表、历史长度、horizon、frequency；
- seed、精度模式、batch size、设备；
- 硬件和软件版本；
- 成功、失败、重试、跳过任务数；
- 是否使用联网下载、外部 API、协变量、跨序列信息；
- 许可名称和许可文件位置。

任务状态表用 `task_id + model_id + seed` 作为唯一键。预测先写临时文件，校验 28 行、日期连续、值有限后原子重命名。重启时只跳过已通过校验的任务；失败尝试保留原日志，不能被成功重试覆盖。

## 13. GitHub 与数据/输出管理

Git 仓库提交：

- 源代码、Dockerfile、锁定依赖、配置、测试；
- 小型 smoke fixture；
- 汇总指标、图表生成代码、运行 manifest；
- 上游版本和许可证清单。

Git 仓库不提交：

- 澳门和香港原始数据，除非已确认再分发许可；
- Hugging Face 权重和缓存；
- AWS/Hugging Face token、云密钥；
- Docker layer；
- 完整每日滚动的数千万行预测。

大文件使用 DVC、对象存储或 GitHub Release artifact，并保存 SHA256。密钥只通过云 secret manager、GitHub Secrets 或本机环境变量注入。正式运行将数据目录只读挂载，将结果目录读写挂载。

## 14. 验收门槛

每个进入主排行榜的模型必须同时满足：

1. 在固定 smoke task 上返回恰好 28 个日期对齐的有限值；
2. 输入文件不包含预测期真实值；
3. Seasonal Naive 验证为重复预测起点前最后一周；
4. 在相同 task list 上运行，或明确列出缺失任务；
5. 保存原始预测和非负处理后的预测；
6. MASE/RMSSE 分母只使用起点前历史；
7. 有完整版本、硬件、时间、资源和失败日志；
8. 从全新 GPU 主机按 README 可以复现 smoke test；
9. 不允许因结果不佳而静默更换 seed、历史长度、起点或模型版本；
10. 最终任务成功率未达预设门槛时，不进入平均排名，只报告可用性结果。

## 15. Todo list

### A. 冻结研究设计

- [x] 将现有回测协议写入 `shared/configs/protocol.yaml`，固定 29 条序列、28 天 horizon、33 个共同起点、3 个历史长度。
- [x] 导出带 SHA256 的 `task_index.parquet`，为每个任务生成稳定 `task_id`。
- [ ] 决定论文采用“原始七仓库审计”还是“七个可比较模型”；推荐后者以 Moirai、Granite 替换 Amazon Forecast、DeepLOB。
- [ ] 决定 TimesFM 3.0 非商业许可是否符合研究和后续发布用途；同时保留 2.5 复现配置。
- [ ] 预先固定高需求阈值定义、概率指标、统计检验和缺失任务处理规则。
- [ ] 预先固定 PatchTST/Informer 的调参预算、seed 和 early stopping 规则。

### B. 建立单仓库

- [ ] 把现有 `paper_pipeline` 重构为可安装的 `src/` 包，保持当前数据管道和 13 个 baseline 输出兼容。
- [ ] 创建 `upstream/revisions.yaml`，记录七个原始仓库及替代模型的 commit/model revision。
- [ ] 添加 `.gitignore`、`LICENSES.md`、`CITATION.cff` 和数据许可说明。
- [ ] 配置 Git LFS/DVC 或对象存储，只存放结果引用和哈希。
- [ ] 新建私有 GitHub repo；代码和配置稳定后再决定是否公开。

### C. 统一输入输出

- [ ] 把当前 JSONL 数据转换为 Arrow/Parquet 任务批次。
- [ ] 定义并实现 `ForecastAdapter` 契约。
- [ ] 将 13 个 baseline 接入同一 adapter 和 run manifest。
- [ ] 编写日期、长度、NaN、负数、重复 task、泄漏校验。
- [ ] 增加概率输出 schema 和 point-only 兼容路径。

### D. 独立环境

- [ ] 构建 `core` CPU 镜像并跑通数据和指标测试。
- [ ] 构建 TimesFM 镜像，固定代码、权重 revision、许可和 CUDA。
- [ ] 构建 Chronos 镜像并实现批量及逐 task 隔离模式。
- [ ] 构建 Lag-Llama 镜像，锁定 GluonTS 等兼容版本。
- [ ] 构建 PatchTST 镜像，接入日频、28 步输出和训练日志。
- [ ] 构建 Informer 镜像，设置 `freq=d`、`pred_len=28` 和训练日志。
- [ ] 若采用推荐七模型，构建 Moirai 与 Granite TSFM 镜像。
- [ ] Amazon Forecast 仅在确认现有 AWS 服务权限后建立 API adapter。
- [ ] DeepLOB 只做仓库可运行性审计，不改名进入主预测表。
- [ ] 把成功镜像推送 GHCR，并记录不可变 image digest。

### E. 遥测与费用

- [ ] 增加冷启动、加载、预处理、训练、推理、后处理、写盘计时。
- [ ] GPU 计时加入同步和预热。
- [ ] 使用 psutil 采集峰值 RSS，使用 NVML/PyTorch 采集显存和可选功率。
- [ ] 建立 `hardware.yaml` 和按日期记录的 `prices.yaml`。
- [ ] 实现每 1,000 个成功任务费用，分离训练、推理、API、失败重试。
- [ ] 建立人工 `implementation_log.csv`。

### F. 逐级运行

- [ ] 对每个 adapter 运行 1×1×1 smoke test。
- [ ] 对 3 市场 × 3 起点 × 3 历史运行 pilot，确定 batch size 和预算。
- [ ] 冻结 pilot 后的配置，不再根据测试精度调整。
- [ ] 运行全部 13 baseline 的 33 起点共同样本。
- [ ] 运行所有适用时序模型的 33 起点共同样本。
- [ ] 检查 task coverage、失败率和日期完全对应后再计算排名。
- [ ] 根据预先记录的预算规则决定是否运行 916 起点完整样本。

### G. 分析与报告

- [ ] 输出任务级、市场级、目的地级和总体 MASE/MAE/RMSSE。
- [ ] 输出高需求日低估、高需求 MASE、bias。
- [ ] 对概率模型输出 CRPS 或 WIS、覆盖率和区间宽度。
- [ ] 输出 p50/p95 延迟、总时间、吞吐量、RAM/VRAM、模型/镜像大小。
- [ ] 输出每 1,000 个任务费用和失败重试费用。
- [ ] 完成区块 bootstrap、DM/Friedman 等预先声明检验。
- [ ] 生成完整 coverage table，Amazon Forecast/DeepLOB 的 N/A 必须附原因。
- [ ] 用一个全新 GPU 环境从镜像复现 smoke test 和一个共同样本 shard。

## 16. 最先执行的五项工作

1. 固定最终“七模型”名单，不把仓库数量等同于可比较模型数量；
2. 从现有数据管道生成不可变 `task_index.parquet`；
3. 先实现统一 adapter、run manifest、断点续跑和遥测；
4. 先接 TimesFM、Chronos，再接 Lag-Llama，随后接 PatchTST/Informer；
5. 在购买大量 GPU 时间前完成 3×3×3 pilot，并根据真实 wall time 计算共同样本预算。

该顺序能先验证比较是否公平、记录是否完整，再扩大计算量。环境是否能安装成功本身就是实施负担结果，失败与重试必须保留，而不是在最后只展示成功运行。
