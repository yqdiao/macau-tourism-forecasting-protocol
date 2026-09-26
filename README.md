# 澳门与香港日度旅游需求预测研究仓库

本仓库包含两个相互关联、分别评价的计算实验：

- **Track 1：**13 个 baseline 与 7 个时序模型的统一滚动回测；
- **Track 2：**从 Track 1 的七模型中根据 B01–B12 选择前三名，研究集成、样本外残差修正和多 Agent 协同；
- **Shared：**两个 Track 共用的时间划分、隔离数据制品和基础依赖。

七个时序模型的 adapter 和 CUDA 容器已经实现，当前状态为 `implemented/pending GPU smoke`。源码 commit 与 checkpoint revision 仍需在 GPU 服务器锁定后逐个验收。

13 个 baseline 的 B01–B12 验证已于 2026-09-26 完成：12,636 条候选预测路径全部完成，失败数为 0，参数状态为 `frozen_after_B_group_tuning`。冻结配置见 [`track1/configs/frozen_baseline_configs.yaml`](track1/configs/frozen_baseline_configs.yaml)，审计记录见 [`track1/audit/b_validation/`](track1/audit/b_validation/)。后续外层回测必须使用该冻结配置，不得用 O01–O33 的结果重新选择参数。

## 仓库结构

```text
.
├─ shared/                       # 两个 Track 共用
│  ├─ configs/protocol.yaml      # A01–A13、B01–B12、O01–O33
│  ├─ data/                      # 隔离后的输入、目标和 manifest
│  ├─ requirements.txt
│  └─ README.md
├─ track1/                       # 单模型和七模型比较
│  ├─ code/                      # 数据管道、baseline、七模型 adapter、评价器
│  ├─ configs/                   # baseline 搜索/冻结配置、上游版本登记
│  ├─ compose/                   # baseline 与 GPU Compose
│  ├─ docker/                    # 独立镜像定义和依赖
│  ├─ docs/                      # Track 1 协议、审计和 GPU 手册
│  ├─ notebooks/                 # 两个研究 notebook
│  ├─ scripts/                   # 长时运行、版本锁定和批量执行脚本
│  ├─ tests/
│  └─ README.md
├─ track2/                       # 三模型集成和多 Agent 实验
│  ├─ docs/
│  └─ README.md
├─ .dockerignore
└─ .gitignore
```

运行时生成的 `track1/results/`、`track1/evaluation/`、模型缓存、权重和密钥不会提交到 Git。

## 共用数据边界

一次任务为“一个目的地/市场 × 一个预测起点 × 一条未来 28 天路径”。预测器只读取：

- `shared/data/task_inputs.parquet`：历史输入，不含真实目标；
- `shared/data/splits/*/task_inputs.parquet`：按实验阶段隔离的输入；
- `shared/data/origin_shards/`：GPU 模型一次只能看到一个预测起点的分片。

独立评估器才可以读取：

- `shared/data/targets.parquet`：未来 28 天真实值。

Track 1 和 Track 2 共用 `shared/configs/protocol.yaml`，从而保证预测起点、horizon、季节周期和任务 ID 完全一致。

## Track 1 快速入口

协议和执行清单：

- [`track1/docs/澳门日度旅游需求_统一实验方案与执行清单.md`](track1/docs/澳门日度旅游需求_统一实验方案与执行清单.md)
- [`track1/docs/FOUNDATION_MODELS_GPU.md`](track1/docs/FOUNDATION_MODELS_GPU.md)

数据与协议审计：

```powershell
python track1/code/audit_protocol_artifacts.py
python track1/code/baseline_config.py --task-manifest shared/data/task_manifest.json
```

Baseline smoke：

```powershell
python track1/code/run_backtest.py `
  --inputs shared/data/task_inputs.parquet `
  --destinations Macau `
  --markets CN_Chinese_mainland `
  --origin-date 2023-12-31 `
  --history all `
  --splits outer_backtest `
  --models seasonal_naive,ets,sarima `
  --output track1/results/smoke_isolated
```

独立评价：

```powershell
python track1/code/evaluate_predictions.py `
  --predictions track1/results/smoke_isolated/predictions.parquet `
  --targets shared/data/targets.parquet `
  --output track1/evaluation/smoke_isolated
```

本机 B 组长时运行：

```powershell
powershell -ExecutionPolicy Bypass -File track1/scripts/run_b_local.ps1 -Workers 4
```

Docker smoke：

```powershell
docker compose -f track1/compose/baseline.yaml run --rm predictor
docker compose -f track1/compose/baseline.yaml run --rm evaluator
```

GPU 七模型：

```powershell
python track1/scripts/pin_upstream_revisions.py
docker compose --env-file track1/.env.pinned -f track1/compose/foundation-gpu.yaml build chronos_2
```

完整 GPU 执行步骤见 [`track1/docs/FOUNDATION_MODELS_GPU.md`](track1/docs/FOUNDATION_MODELS_GPU.md)。

## Track 2 快速入口

Track 2 当前为预注册方案；具体三个模型必须在七模型完成 B01–B12 后才冻结：

- [`track2/docs/TRACK2_三模型集成_级联残差与多Agent协同实验方案.md`](track2/docs/TRACK2_三模型集成_级联残差与多Agent协同实验方案.md)

正式选择只能使用 B 组任务级 MASE(7) 宏平均及预注册并列规则。O01–O33 不参与前三名、权重、残差模型或回退阈值的选择。

## 安装依赖

```powershell
python -m pip install -r shared/requirements.txt
```

七个时序模型使用各自的 CUDA 镜像，不建议把所有上游依赖安装进同一个 Python 环境。

## 数据和许可证

仓库仅应保存已确认允许向协作者分发的数据制品。若原始数据没有明确再分发许可，应仅保留下载说明、SHA256 和小型样例。TimesFM 3.0、Moirai 2 等模型权重还需遵守各自模型卡和许可证。
