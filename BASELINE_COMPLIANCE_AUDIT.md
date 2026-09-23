# Baseline 主实验协议合规审计

审计日期：2026-09-23
审计对象：`data_pipeline.py`、`models.py`、`run_backtest.py`、现有数据快照与 smoke 输出。
依据：`澳门日度旅游需求_统一实验方案与执行清单.md`。

## 结论

当前代码已经形成统一的 28 天单变量 baseline 接口，并包含 13 个模型；模型训练样本没有把预测起点后的目标值作为特征。inputs/targets 物理隔离、`all` 模式、raw/clipped 双输出、任务级指标、原子写入和 A/B/外层任务日历已实现。13 个 baseline 的 33 个候选配置已登记，但 B 组选参、参数冻结、完整资源统计和七个大模型 adapter 仍未完成，因此现有输出仍不能作为论文最终排名。

13 个模型为：Seasonal Naive、ETS、SARIMA、Ridge、Linear Regression、Random Forest、Gradient Boosting、SVR、Lasso、Elastic Net、Parrot Random Forest 800、LightGBM、Local Transformer。

## 已符合的项目

| 项目 | 结论 | 证据 |
|---|---|---|
| 预测期 | 符合 | 所有 adapter 默认输出 28 天。 |
| 单变量输入 | 基本符合 | `forecast()` 只接收当前市场的一维历史序列；语义等价于单个 `[1, context_length, 1]` 输入。 |
| 预测起点边界 | 符合 | tabular 与 Transformer 的训练窗口均满足目标日不超过当前起点。 |
| Seasonal Naive | 符合 | 28 天预测循环复制起点前最后 7 个实测值，没有读取预测期实际值。 |
| 外层共同起点 | 符合 | 实际运行得到 33 个起点：2023-12-31 至 2026-06-14，最后目标日为 2026-07-12。 |
| 非负输出 | 符合 | 同时保存原始值、裁剪后值与裁剪标记。 |
| A/B/外层任务日历 | 符合 | `configs/protocol.yaml` 列出 A01–A13、B01–B12、O01–O33；任务索引和 manifest 保存 SHA256。 |
| 180/365 历史 | 可运行 | 已有对应参数和 smoke 结果，但尚未完成全模型、全市场共同样本回测。 |

## 未符合或证据不足的项目

| 严重度 | 问题 | 当前实现 | 协议要求/影响 |
|---|---|---|---|
| 已修复 | 目标值物理隔离 | `ForecastInputTask` 不含 actual；预测器只读 `task_inputs.parquet`，独立评估器才读取 `targets.parquet`。 | `compose.yaml` 的 predictor 没有 targets 挂载。 |
| 已修复 | `all` 主实验 | 已生成 all-available 任务并通过单任务 smoke；不再把 `all` 转换成 0。 | 主排名可以使用 all-available expanding。 |
| 阻断 | B 组选参和冻结未完成 | A/B 起点与 33 个候选已登记；`frozen_baseline_configs.yaml` 仍为 `pending_B_group_tuning`，尚无 trial 记录或参数赢家。 | B 组选择超参数，A 组仅作压力测试；外层回测不得再调参。 |
| 已修复 | 原始预测 | `ForecastResult` 和 `predictions.parquet` 同时保存 `prediction_raw`、非负 `prediction`、`was_clipped`。 | 可检查裁剪对结果的影响。 |
| 已修复 | 指标聚合单位 | 独立评估器先生成 `task_metrics.parquet`，再对任务等权宏平均。 | 不再用逐日观测数量隐式加权任务。 |
| 高 | 缺少完整运行信息 | 数据 manifest 已记录协议、任务列表和数据 SHA；单次运行仍缺代码 commit、上游 commit、checkpoint revision、容器 digest、成功/失败/重试数。 | 尚不能完整复现模型与数据状态。 |
| 高 | 断点续跑仍不完整 | 已改为临时文件和原子重命名，但仍没有任务完成索引、分区续跑或 failures 文件。 | 长时间 GPU 回测需要补齐任务级恢复机制。 |
| 中 | 超参数仍为硬编码 | 搜索空间与 seed 已登记，但 ETS、SARIMA、tabular、Transformer 预测函数尚未读取候选或冻结配置。 | 先实现配置驱动调参，再经 B 组选择并冻结。 |
| 中 | Transformer 可复现性不足 | 固定 `torch.manual_seed`，但未保存 checkpoint，也未启用完整确定性设置；只运行一个 seed。 | 保存训练损失、checkpoint、seed 与硬件信息，并按预注册规则处理随机性。 |
| 中 | 资源/成本记录不完整 | 只有 fit、inference、total 秒数。 | 尚缺模型加载、预热、p50/p95、RSS、显存、吞吐量、费用、失败和重试。 |
| 中 | 输出审计字段仍需扩展 | 已有稳定 `task_id`、split/fold、effective context、raw prediction、clip 标记；尚缺完整模型和容器版本、资源与失败记录。 | 各模型必须按相同 task_id 合并与比较。 |
| 中 | 现有 smoke 覆盖不足 | 只检查部分模型、单市场、单起点和 180 天历史。 | 每模型先通过共同 smoke，再运行 33 个外层起点和规定敏感性分析。 |

## 各类模型判断

| 模型类别 | 是否可以继续使用 | 修复重点 |
|---|---|---|
| Seasonal Naive | 可以 | 保留 raw/clipped 双输出并进入独立评估。 |
| ETS、SARIMA | 可以 | 把规格写入配置；若存在候选规格，只能用 B 组选择；记录拟合失败和收敛状态。 |
| 9 个 tabular 模型 | 可以 | 直接多步训练逻辑无明显未来泄漏；需用 B 组选择参数并冻结，保存每个 horizon 的配置。 |
| Local Transformer | 可以作为本地神经网络 baseline | 它不是预训练时序大模型；需要验证集选参、checkpoint、训练日志、随机性与 GPU 资源记录。 |

## 上传到私有 GitHub 仓库的边界

### 可以上传

- 统一数据管道、baseline adapter、评估器、配置、环境锁文件、Dockerfile、测试和本审计报告。
- 不含真实记录的小样例数据、schema、字段字典、数据质量摘要和 SHA256。
- 七个上游模型的仓库 URL、锁定 commit、checkpoint revision、许可证清单和自动下载脚本。

### 修复或确认后再上传

- `normalized_daily.jsonl`：文件约 9 MB，GitHub 技术上可以存放；必须先确认澳门与香港源数据的再分发许可。私有仓库仍会向获授权协作者分发数据。
- `data_manifest.json` 与 README 已删除本机绝对路径；manifest 只保存逻辑来源标识与 SHA256。
- 学生项目、notebook 与 report：须确认作者许可，并删除姓名、学号、邮箱、访问令牌及其他个人信息。

### 不应直接提交

- 模型权重、缓存、虚拟环境、API 密钥、大型预测结果和 GPU 日志全集。
- 七个第三方仓库的无版本复制。优先记录 commit 并由脚本拉取；确需固定源码时使用 Git submodule 或 vendor 目录，同时保留原许可证和 NOTICE。
- 外层 targets 与模型可读输入放在同一个可挂载目录。应生成独立的 `task_inputs` 和 `targets` 制品。

## 上传前修复顺序

1. 已将 A/B 折和 33 个共同起点写入版本化协议配置与任务索引。
2. 将已登记的候选参数接入模型代码；完成 B 组选参、冻结、A 组压力测试。
3. 增加任务级恢复索引、分区输出、failures 文件和失败重试记录。
4. 增加 Seasonal Naive 胜率、IQR、最差 10% 与预注册置信区间输出。
5. 补齐环境锁、上游 commit、checkpoint revision、许可证、资源和成本记录。
6. 接入七个时序大模型的锁定容器和统一 adapter。
