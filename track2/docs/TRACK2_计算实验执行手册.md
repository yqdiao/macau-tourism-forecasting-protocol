# Track 2 计算实验执行手册

> 日期：2026-10-11。本文件把 `track2/docs/TRACK2_三模型集成_级联残差与多Agent协同实验方案.md` 收成可按顺序执行的计算实验。研究方案仍以该文件为准；本手册只规定输入路径、实验编号、冻结顺序和运行命令。
>
> 通用语言模型底座只使用 DeepSeek，经 OpenAI 兼容接口调用。密钥由运行者在执行第 4 节前写入环境变量，不写入本文件，也不写入 Git。
>
> 工作目录：`macau-tourism-forecasting-protocol/`。下文路径都相对该目录。

## 0. 执行边界

Track 2 只回答四件事：前三名集成是否优于最佳单模型；B 组学到的权重是否优于等权；样本外残差修正能否稳定改进；多 Agent 是否提高成功率、故障检出和自动恢复。数值预测、权重、裁剪、指标和回退都由确定性程序完成。DeepSeek 只做任务解析、工具调度、异常解释和审计摘要，不得改写预测值、分位数或指标。

已冻结、本手册不再改动的决定：

| 项目 | 取值 | 来源 |
|---|---|---|
| M1 | Granite TimeSeries PatchTST-FM，`GRA01`，B 组 MASE(7) 0.9274904736807095 | `track2/configs/selected_track2_models.yaml` |
| M2 | Chronos-2，`CH01`，0.9818279776843402 | 同上 |
| M3 | TimesFM 3.0，`TFM01`，0.9891630076845611 | 同上 |
| 未入选 | Moirai 1.054、PatchTST 1.364、Lag-Llama 1.912、Informer 2.031 | 同上 |
| 外层性质 | O01–O33 标为探索性 | 冻结文件写于 2026-10-05，晚于 2026-10-03 首次读取外层真实值 |
| 确认性检验 | O34–O36，数据尚未导入，本手册不运行 | 方案第 7.5 节 |

O01–O33、香港、180/365 天和 A01–A13 都不得回头修改前三名、集成方法、权重、`epsilon`、残差修正器、收缩系数、最大修正幅度、分位数方案、回退顺序或质量检查阈值。这些元参数只在 B01–B12 的时间顺序交叉拟合上选定，写入第 6 节的冻结文件后才应用到外层。

本手册在首次运行 Track 2 外层评分之前登记以下常数。它们不是从 O 组看起来的，后续不得按外层结果改写：

| 常数 | 登记值 | 用途 |
|---|---|---|
| `epsilon` | `1e-6` | 误差倒数权重的分母平滑 |
| `min_prior_folds` | `3` | B04 起才允许学习权重；B01–B03 用等权和 `lambda=0` |
| 步长权重启用门槛 | 交叉拟合宏平均 MASE(7) 至少再低 `0.01` | 否则冻结为全 horizon 一组权重 |
| 残差 `lambda` 网格 | `0, 0.25, 0.5, 0.75, 1` | 全 horizon 共享；分步长同样要过 `0.01` 门槛 |
| 残差上下文候选 | `28, 56, 84` 天已发生残差 | 只使用当时已经观测到的起点 |
| 单步修正上限 | 未修正预测绝对值的 `25%` | 超出则截断 |
| 水平检查分位 | B 组干净输出比值的 `0.5%` 与 `99.5%` | 起点前 28 天均值 `< 5` 的任务跳过 |
| 故障时间表种子 | `20261005` | 1%、5%、10% 三档 |
| 区块 bootstrap | 2000 次，市场分层，起点块长 3，种子 `20261003` | 与 Track 1 主表相同 |
| DeepSeek 温度 | `0` | 协调器；`deepseek-reasoner` 不进入数值路径 |

主指标与 Track 1 相同。一个任务是一个市场、一个起点、28 天。点预测先做 `max(0, prediction)`。

\[
s_7(o)=\frac{1}{T-7}\sum_{t=8}^{T}\lvert y_t-y_{t-7}\rvert,\qquad
\mathrm{MASE}(7)=\frac{\frac{1}{28}\sum_{h=1}^{28}\lvert\hat y_{o+h}-y_{o+h}\rvert}{s_7(o)}.
\]

汇总是：多种子先在任务内平均（本轨三个成员各种子数为 1），再市场内对折次平均，再 27 个澳门市场等权。香港 2 条序列单独报告，不并入这个数。\(s_7=0\) 的任务不进平均；现有数据中没有这种任务。预测文件里的 `mase7_scale` 只作核对，评分时用历史重算。

## 1. 全部计算实验

下表是本轨要跑的全部计算实验。`选参` 为“是”的实验只能读 B 组；为“否”的实验只应用已经冻结的配置。

| 编号 | 实验 | 选参 | 计算位置 | 依赖 |
|---|---|---|---|---|
| T2-X0 | 清点三成员及对照预测，核对任务键、28 个有限值、分位数 | 否 | CPU | 已有 Track 1 预测 |
| T2-E0 | M1 单模型 | 否 | 复用预测 | T2-X0 |
| T2-E1 | M1+M2 等权 | 否 | CPU | T2-X0 |
| T2-E2 | M1+M2+M3 等权 | 否 | CPU | T2-X0 |
| T2-E3 | 验证 MASE 倒数加权，全 horizon 一组权重 | 是，仅 B 交叉拟合 | CPU | T2-X0 |
| T2-E4 | 非负且和为 1 的约束最小二乘 | 是，仅 B 交叉拟合 | CPU | T2-X0 |
| T2-EH | 28 组步长权重；未过 `0.01` 门槛则不启用 | 是，仅 B | CPU | T2-E3、T2-E4 |
| T2-K | B 组 Top-1/2/3/4/7 等权边际；不改前三名名单 | 否（只报告） | CPU | 七模型 B 预测 |
| T2-R | 样本外残差修正；无稳定改进则冻结 `lambda=0` | 是，仅 B | CPU，可选 GPU 读残差 | 选定的集成 |
| T2-P | 分位数加权、单调化；必要时按 horizon 的 split conformal | 是，仅 B | CPU | 选定权重 |
| T2-Q | 用 B 组干净输出冻结水平检查阈值和超时 | 是，仅 B | CPU | T2-X0、计时文件 |
| T2-C1 | ETS03、RID02、EN03 等权 | 否 | CPU | 已有基线预测 |
| T2-01 | 外层 M1 | 否 | 复用 | 冻结后 |
| T2-02 | 外层 M1+M2 等权 | 否 | CPU | 冻结后 |
| T2-03 | 外层三模型等权 | 否 | CPU | 冻结后 |
| T2-04 | 外层冻结权重 | 否 | CPU | T2-E3 或 T2-E4 的冻结结果 |
| T2-05 | T2-04 加冻结残差 | 否 | CPU | T2-R |
| T2-06 | T2-05 加冻结概率校准 | 否 | CPU | T2-P |
| T2-C2 | M1；未过 T2-Q 则回退季节性朴素 | 否 | CPU | T2-Q |
| T2-A0 | 直接脚本执行冻结集成与回退 | 否 | CPU | 冻结配置 |
| T2-A1 | 确定性工作流协调器，数值规则与 T2-A0 相同 | 否 | CPU | T2-A0 |
| T2-A2 | DeepSeek 协调器，数值规则与 T2-A0 相同 | 否 | DeepSeek API | 第 4 节密钥 |
| T2-HK | 香港延伸，权重与阈值不重学 | 否 | CPU | 澳门冻结配置 |
| T2-H180 | O01–O33，`history=180` | 否 | CPU | 已有 180 天预测 |
| T2-H365 | O01–O33，`history=365` | 否 | CPU | 已有 365 天预测 |
| T2-A | A01–A13 疫情压力，`history=all` | 否 | CPU | 已有疫情窗预测 |
| T2-FI | 故障注入 1%、5%、10%；系统为 A0′、C2、A1、A2 | 否 | CPU + DeepSeek | O 组探索性；确认期不做 |
| T2-BOOT | 配对区块 bootstrap | 否 | CPU | 外层预测 |
| T2-COST | 每 1,000 任务费用、增量成本效益比、体量平衡点、帕累托 | 否 | CPU | 计时与 token 日志 |
| T2-O34 | 前瞻确认期 | 否 | 数据到达后 | 全部配置已提交；当前跳过 |

无故障时，T2-A0、T2-A1、T2-A2 的 28 天数值必须逐位一致。不一致则 T2-A2 失败，不得把 DeepSeek 输出当作预测。

## 2. 直接复用的预测，不重跑成员模型

三个成员在 B 组、外层、180/365 和疫情窗上的零样本预测已经存在。Track 2 的集成实验读取这些 parquet，不重新加载 Granite、Chronos-2、TimesFM 权重。只有残差修正器在 B 组交叉拟合中被选中、且必须对残差序列重新前向时，才单独开一次 GPU；若冻结 `lambda=0`，则全程不再占用 GPU。

点预测列用 `prediction`（已非负裁剪）。原始值在 `prediction_raw`。三个成员都有 `q0.1`、`q0.5`、`q0.9`。

| 数据 | 成员路径模式 | 任务规模 |
|---|---|---|
| B01–B12，`history=all` | `track1/results/foundation_b_default/{fold}_{origin}/{model}/predictions.parquet` | 澳门 \(27\times 12=324\) / 模型 |
| O01–O33，`history=all` | `track1/results/foundation_outer_default/{fold}_{origin}/{model}/predictions.parquet` | 澳门 891；香港另计，不混入主指标 |
| 180 天 | `track1/results/sensitivity/outer_backtest/180/{model}/seed42/{fold}_{origin}/predictions.parquet` | 与外层相同的目标日 |
| 365 天 | `track1/results/sensitivity/outer_backtest/365/{model}/seed42/{fold}_{origin}/predictions.parquet` | 同上 |
| A01–A13 | `track1/results/sensitivity/pandemic_stress/all/{model}/seed42/{fold}_{origin}/predictions.parquet` | 澳门 \(27\times 13=351\) |

`{model}` 取 `granite_patchtst_fm`、`chronos_2`、`timesfm_3`。Top-4/Top-7 消融再读同目录下的 `moirai`、`patchtst`、`lag_llama`、`informer`。PatchTST 与 Informer 的外层主结果在 `track1/results/foundation_outer_frozen/`，多种子；B 组消融用 `foundation_b_default` 里各种子文件时，先在任务内把种子平均，再进入等权。

对照基线，只用冻结候选，不使用其他候选：

| 角色 | 模型 | B 组路径 | 外层路径 |
|---|---|---|---|
| T2-C1 | ETS03 | `track1/results/b_validation_server_repro/predictions/ets/ETS03/` | `track1/results/outer_server_all/predictions/ets/ETS03/42/` |
| T2-C1 | RID02 | `.../ridge/RID02/` | `.../ridge/RID02/42/` |
| T2-C1 | EN03 | `.../elasticnet/EN03/` | `.../elasticnet/EN03/42/` |
| 最终回退 | 季节性朴素 SN01 | `.../seasonal_naive/SN01/` | `.../seasonal_naive/SN01/42/` |

真实值只在预测文件写完并计算 SHA256 之后，由评分进程读取 `shared/data/targets.parquet`。协调器和 DeepSeek 的工作目录不得挂载该文件。协议日期在 `shared/configs/protocol.yaml`。

180 天和 365 天的 \(s_7\) 在被截短的历史上重算，这两列 MASE 不与 `history=all` 相减。

## 3. 环境

```bash
cd /root/DYQ/macau-tourism-forecasting-protocol
python -m pip install -r shared/requirements.txt
mkdir -p track2/configs track2/code track2/results/{inventory,b_crossfit,outer_macau,extension_hong_kong,sensitivity_180,sensitivity_365,pandemic_stress,agents,fault_injection,evaluation}
```

DeepSeek 相关实验（T2-A2、T2-FI 中的 A2）在导出下面三个变量之前不要启动。把密钥放在 shell 环境或未跟踪的 `track2/.env`，并确认 `.gitignore` 已忽略该文件。

```bash
export DEEPSEEK_API_KEY="在此填入密钥"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

调用形态固定为 `POST ${DEEPSEEK_BASE_URL}/chat/completions`，请求头 `Authorization: Bearer ${DEEPSEEK_API_KEY}`。`model` 只用 `DEEPSEEK_MODEL`。默认 `deepseek-chat`。若要更换 DeepSeek 型号，只改这个变量并在 `run_manifest.json` 记录型号字符串；不得改用其他厂商。`temperature=0`，`stream=false`。每次调用把模型名、`finish_reason`、输入 token、输出 token、延迟和 HTTP 状态写入 `track2/results/agents/llm_calls.jsonl`。

协调器允许 DeepSeek 调用的工具只有：`audit_task`、`run_member`、`check_output`、`apply_frozen_ensemble`、`apply_frozen_residual`、`record_fallback`、`write_explanation`。工具返回值是结构化 JSON。系统提示写明：模型不得输出 28 个数，不得读取目标，不得更换冻结权重。解析器丢弃任何含 `prediction`、`q0.1`、`q0.5`、`q0.9` 的助手正文；数值只接受确定性工具的返回。出现一次这种正文记一条违规，该任务改由 T2-A1 的同一条规则完成，并在结果表把该任务标为 LLM 违规回退。

## 4. T2-X0：清点

先确认三个成员在 B 组与外层都是 12 个和 33 个起点、每个文件可打开。这一步不评分。

```bash
python - << 'PY'
from pathlib import Path
root = Path("track1/results")
members = ["granite_patchtst_fm", "chronos_2", "timesfm_3"]
b = root / "foundation_b_default"
o = root / "foundation_outer_default"
for split, base in [("B", b), ("O", o)]:
    folds = sorted(p for p in base.iterdir() if p.is_dir())
    print(split, "folds", len(folds))
    for m in members:
        n = sum((f / m / "predictions.parquet").is_file() for f in folds)
        print(f"  {m}: {n}/{len(folds)}")
PY
```

通过标准：B 为 12/12，O 为 33/33，三个成员都满足。接着做行级核对，输出 `track2/results/inventory/member_inventory.csv`。每一行是一个起点文件：行数是否为 `市场数 × 28`，`horizon_day` 是否为 1..28，`prediction` 是否全部有限，`q0.1 ≤ q0.5 ≤ q0.9` 是否成立，`origin_date` 是否等于目录名中的日期。澳门主实验只保留 `destination == Macau` 的 27 个市场。任一成员缺任务或出现非有限值时停止，不进入交叉拟合。

三个成员在同一 `(destination, market, origin_date, horizon_day)` 上内连接。连接失败的键写入 `track2/results/inventory/key_mismatches.csv`。同时写出 B 组任务级误差相关：先只在库存阶段保存三成员预测矩阵，相关矩阵留到评分进程计算，避免这一步打开 `targets.parquet`。

## 5. B 组交叉拟合（唯一允许选元参数的计算）

对验证起点 \(B_k\)，权重和残差规则只用 \(B_1,\ldots,B_{k-1}\)。\(k\le 3\) 时权重为等权、`lambda=0`。市场不随机拆分。主权重在 27 个澳门市场上共享。

### 5.1 T2-E0、T2-E1、T2-E2

这三项没有要学习的参数，直接在每个起点上计算，供交叉拟合比较。

- T2-E0：\(\hat y = \hat y^{(M1)}\)
- T2-E1：\(\hat y_h = (\hat y^{(M1)}_h + \hat y^{(M2)}_h) / 2\)
- T2-E2：\(\hat y_h = (\hat y^{(M1)}_h + \hat y^{(M2)}_h + \hat y^{(M3)}_h) / 3\)

结果逐日再做 `max(0, ·)`。保存 `prediction_raw` 与裁剪后的 `prediction`。

### 5.2 T2-E3 倒数权重

在训练起点集合上，先按与主指标相同的宏平均得到每个成员的 MASE(7)，记为 \(E_m\)。全 horizon 共享：

\[
w_m=\frac{1/(E_m+\epsilon)}{\sum_{j=1}^{3}1/(E_j+\epsilon)},\qquad \epsilon=10^{-6}.
\]

预测 \(\hat y_h=\sum_m w_m \hat y^{(m)}_h\)，再非负裁剪。把该折的 \(E_m\)、\(w_m\) 和训练起点列表写入 `track2/results/b_crossfit/weights_E3.jsonl`。

### 5.3 T2-E4 约束最小二乘

只用训练起点上、已经发生的 28 天真实值。这些真实值属于 B 组调参期，允许使用。优化

\[
\min_{w\ge 0,\;\sum w_m=1}\sum_{i,h}\Bigl(y_{i,h}-\sum_{m=1}^{3}w_m\hat y^{(m)}_{i,h}\Bigr)^2.
\]

实现用 `scipy.optimize.minimize`，SLSQP，初值 \((1/3,1/3,1/3)\)。失败或无可行点时该折退回等权，并在日志写 `fallback=equal`。不得改用神经网络门控。

### 5.4 选择 E2、E3 或 E4

把每个起点的样本外元预测收成任务级 MASE(7)，再做市场等权宏平均。三者取最低者。差距小于 \(10^{-4}\) 时依次选更简单的方法：E2 优先于 E3，E3 优先于 E4。选定后，用全部 B01–B12 重拟合一次最终权重，写入冻结文件。交叉拟合分数本身保留，不被这次重拟合覆盖。

### 5.5 T2-EH 步长权重

对已选定的 E3 或 E4，再做一版 \(h=1..28\) 各一组权重，训练方式与全 horizon 版相同，只是损失和 \(E_{m,h}\) 按步长分开。若交叉拟合宏平均 MASE(7) 比全 horizon 版低不到 `0.01`，冻结 `per_horizon: false`。E2 不设步长权重。

### 5.6 T2-K 边际报告

只在 B 组报告，不改变 M1–M3。等权组合为：

| 组合 | 成员 |
|---|---|
| Top-1 | Granite |
| Top-2 | Granite、Chronos-2 |
| Top-3 | 再加 TimesFM 3.0 |
| Top-4 | 再加 Moirai |
| Top-7 | 再加 PatchTST、Lag-Llama、Informer |

输出 `track2/results/b_crossfit/topk_marginal.csv`。第四名及以后即使 B 组等权更好，也不进入外层主集成。

### 5.7 T2-R 残差

残差用交叉拟合集成，不用样本内拟合残差：

\[
e_{i,h}=y_{i,h}-\hat y^{ENS}_{i,h}.
\]

修正器候选只有 M1、M2、M3 的统一单变量接口，输入是该市场在当前起点之前已经完成的起点上的残差序列。上下文试 `28/56/84`。收缩 \(\lambda\in\{0,0.25,0.5,0.75,1\}\)，默认全 horizon 一个 \(\lambda\)。

\[
\hat y^{RES}_{i,h}=\hat y^{ENS}_{i,h}+\lambda\hat e_{i,h}.
\]

单步 \(|\lambda\hat e|\) 不得超过 \(|\hat y^{ENS}|\) 的 25%。最终再 `max(0, ·)`。保存修正前、原始修正量、截断后修正量和最终值。高需求日与低需求日分开统计修正方向；高需求日沿用预测文件中的 `high_demand_threshold`。

若最佳 \(\lambda>0\) 的交叉拟合宏平均仍不能比 \(\lambda=0\) 低至少 `0.01`，冻结 `lambda=0`，主结果停在并行集成。历史残差不足一个上下文长度的起点同样使用 `lambda=0`。

冻结 `lambda=0` 时不要启动 GPU。只有选中某个成员作为残差修正器时，才对该成员按 Track 1 已冻结的 checkpoint 跑残差序列，并另写 `residual_runs/`。

### 5.8 T2-P 概率

三个成员都有 q0.1、q0.5、q0.9。用与点预测相同的冻结权重做分位数加权，然后按步长把三个分位数排序，保证 \(q_{0.1}\le q_{0.5}\le q_{0.9}\)。点预测仍用第 5.4 节的点组合，不被 q0.5 覆盖，除非冻结文件写明 `point_source: weighted_q50`。默认 `point_source` 保持各成员自己的 `prediction` 列。

在 B 组交叉拟合上报告 pinball loss、WIS、80% 区间覆盖率和平均宽度。覆盖率用区间 \([q_{0.1}, q_{0.9}]\)。若交叉拟合覆盖率偏离 80% 超过 10 个百分点，追加按 horizon 的 split conformal：用训练起点的样本外残差取经验分位，加减到集成点预测上，再做非负裁剪和单调化。未超过则冻结 `calibration: weighted_quantiles_only`。

### 5.9 T2-Q 质量阈值

只用三个成员的 B 组干净输出。对每个任务计算

\[
r=\frac{\mathrm{mean}(\hat y_{o+1:o+28})}{\mathrm{mean}(y_{o-27:o})}.
\]

分母是起点及以前最近 28 天的实际值，不是未来目标。`r_low`、`r_high` 取全部合格任务上 \(r\) 的 0.5% 与 99.5% 分位数。起点前 28 天均值低于 5 的任务不参与分位数，外层也不做这项检查。

超时阈值：读取 `track1/results/compute_benchmark/`，取该成员单起点墙钟 p95 的 3 倍。计时文件缺少该成员时，超时先记为缺失，T2-FI 的 S1 暂停，其余实验继续。

结构检查不依赖数据：恰好 28 个有限值，日期连续，起点等于历史末日，模型版本等于冻结配置。

### 5.10 冻结

交叉拟合和阈值计算完成后，一次写齐下面三个文件，并记录各自的 SHA256：

- `track2/configs/track2_ensemble_config.yaml`：入选方法、最终权重、`epsilon`、`per_horizon`、残差修正器、上下文、`lambda`、25% 上限、概率方案
- `track2/configs/track2_fallback_policy.yaml`：第 7 节的回退顺序、`r_low`、`r_high`、超时
- `track2/configs/track2_freeze_manifest.json`：上述文件及 `selected_track2_models.yaml` 的 SHA256、B 组预测清单 SHA256、代码 commit、`outer_used_for_selection: false`

写完后打印宏平均表 `track2/results/b_crossfit/selection_table.csv`。从这一刻起，禁止再改这三个文件。外层脚本启动时核对 SHA256，不一致就退出。

建议的冻结字段：

```yaml
schema_version: 1
selected_method: T2-E2   # 运行后改成实际胜出的 E2 / E3 / E4
epsilon: 1.0e-6
per_horizon: false
weights: {granite_patchtst_fm: 0.333333333333, chronos_2: 0.333333333333, timesfm_3: 0.333333333333}
residual: {enabled: false, lambda: 0.0, corrector: null, context_days: null, max_frac: 0.25}
probability: {method: weighted_quantiles_only, quantiles: [0.1, 0.5, 0.9]}
point_source: member_prediction_column
min_prior_folds: 3
selection_metric: task_MASE7_market_fold_macro
outer_used_for_selection: false
```

权重在 `selected_method` 为 T2-E2 时保持各 `1/3`。E3 或 E4 胜出时，把全量 B 组重拟合的权重写入 `weights`，三者之和为 1，且都 `≥ 0`。

## 6. 外层及应用实验（只套用冻结配置）

启动前检查 `track2/configs/track2_freeze_manifest.json` 存在，且 `outer_used_for_selection` 为 false。每个实验写独立目录，文件至少含 `predictions.parquet`、`run_manifest.json`。预测列与 Track 1 对齐：`task_id`、`fold_id`、`destination`、`market`、`origin_date`、`target_date`、`horizon_day`、`prediction_raw`、`prediction`、`model`、`candidate_id`。集成行的 `model` 用实验编号，例如 `T2-03`。

| 实验 | 输出目录 | 做法 |
|---|---|---|
| T2-01 | `track2/results/outer_macau/T2-01/` | 拷贝 M1 的澳门行，不改数值 |
| T2-02 | `.../T2-02/` | M1 与 M2 等权 |
| T2-03 | `.../T2-03/` | 三模型等权 |
| T2-04 | `.../T2-04/` | 冻结权重；某成员缺失时先不要在这里回退，回退交给 T2-A0 |
| T2-05 | `.../T2-05/` | 在 T2-04 上加冻结残差；`lambda=0` 时与 T2-04 逐位相同 |
| T2-06 | `.../T2-06/` | 在 T2-05 的点预测之外写入校准后的 q0.1/q0.5/q0.9 |
| T2-C1 | `.../T2-C1/` | ETS03、RID02、EN03 等权，只使用 CPU 预测文件 |
| T2-C2 | `.../T2-C2/` | M1 通过 T2-Q 则保留，否则整条 28 天换成 SN01 |

香港用同一配置，输出到 `track2/results/extension_hong_kong/`，不重新估计权重。180 天、365 天、疫情窗分别写入 `sensitivity_180`、`sensitivity_365`、`pandemic_stress`，方法集合与上表相同。短历史实验的尺度用对应历史重算。

评分在预测目录关闭之后单独运行。评估进程读取 `shared/data/targets.parquet`，调用与 Track 1 相同的任务级 MASE(7) 定义。每个实验至少输出：

- 宏平均 MASE(7)、MAE、RMSSE(7)、WAPE、偏差百分比
- 高需求日 MASE 和平均低估
- 相对 M1 的任务胜率
- 最差 10% 任务的 MASE 均值、单任务最大 MASE
- 33 个起点宏平均的标准差
- 事后最优差距：每个任务取三成员中 MASE 最低者作为上限，报告集成填上了 M1 到该上限之间的比例
- 三成员任务级误差相关矩阵
- q10/q50/q90 的 pinball、WIS、80% 覆盖率、平均宽度（T2-06 必有；成员文件可另附）
- 覆盖率、失败率、回退率

主比较是 T2-01、T2-03、T2-04、T2-05。澳门与香港分开。论文表述保持：O01–O33 为探索性。

T2-BOOT：对上述主比较的配对损失差做 2000 次区块 bootstrap，市场分层，相邻起点块长度为 3，种子 `20261003`。报告效应量和 95% 区间。若另做 Diebold–Mariano，对方差使用能覆盖 28 天路径重叠的稳健估计，且不把 p 值当作唯一结论。

## 7. 回退、T2-A0 与 T2-A1

回退顺序固定，写在 `track2_fallback_policy.yaml`：

```text
三模型冻结集成
  → 一个成员失败：其余成员按冻结权重重新归一化
  → 权重不可用：剩余成员简单平均
  → 三个成员都失败：季节性朴素 SN01
```

残差工具失败只把 `lambda` 视为 0，已经通过检查的集成预测仍然交付。每次回退写 `fallbacks.jsonl`：任务键、失败成员、异常类型、尝试次数、最终路径、28 天输出是否完整。

T2-A0 是一段没有语言模型的脚本：读冻结配置，跑质量检查，按上面的顺序产出预测。正常的 O 组任务上，它的数值应与 T2-04（残差关闭时）或 T2-05（残差开启时）一致；不一致先停止 Agent 实验。

T2-A1 使用同一个函数。差别只在调度记录：任务 ID、输入 SHA256、三个成员版本、重试次数、幂等键。同一任务重复提交不得生成第二套数值。协调器进程的文件权限不包含 `shared/data/targets.parquet`。用下面的检查确认：

```bash
python - << 'PY'
import os
from pathlib import Path
target = Path("shared/data/targets.parquet").resolve()
# 协调器工作目录只应看到 track2/results 与成员预测
print("target_exists_for_evaluator", target.is_file())
print("coordinator_must_not_open_this_path")
PY
```

正式运行时，T2-A1 的进程用单独的工作目录启动，环境中不设置目标路径。运行后在 `run_manifest.json` 写 `actuals_visible_to_coordinator: false`。

## 8. T2-A2：DeepSeek 协调器

前提：第 3 节三个环境变量已设置，`track2_freeze_manifest.json` 已存在。未设置密钥时，本实验标记为 `blocked_on_api`，其余实验继续。

一次任务的消息顺序：

1. 系统提示：角色是流程协调器；数值只能来自工具；禁止编造协变量；禁止根据语言理由更换模型。
2. 用户消息：只含 `task_id`、目的地、市场、起点、历史末日、历史长度、输入 SHA256、冻结配置 SHA256。不含游客数序列，不含未来日期的真实值。
3. 模型发出工具调用。运行器执行工具并把 JSON 结果送回。
4. 模型最后调用 `write_explanation`，引用 `run_id` 和回退记录。解释文本不参与评分。

成员预测由 `run_member` 读取第 2 节已经封存的 parquet，不在 Agent 回路里重新推理。这样无故障任务的数值路径与 T2-A0 相同，API 费用只来自协调，不来自重新生成 28 天预测。

验收：

- 抽外层全部澳门任务，T2-A2 与 T2-A0 的 `prediction` 最大绝对差为 0。
- 同一任务用温度 0 重复 3 次，工具序列和最终数值一致。若 DeepSeek 在温度为 0 时仍有字节级波动，以第一次工具返回的确定性数值为准，并把三次响应的差异写入 `track2/results/agents/nondeterminism.jsonl`。数值本身不允许取三次平均。
- 统计违规次数、调用次数、输入/输出 token、按调用日志汇总的 API 费用。
- 研究者工时单独登记，不计入货币成本。

费用单价在第一次成功调用当天记入 `track2/configs/deepseek_price.yaml`（输入单价、输出单价、货币、价格日期）。之后改价只影响敏感性表，不回写已封存的 `llm_calls.jsonl`。

## 9. T2-FI 故障注入

这是次要实验，只在澳门 O01–O33 上做，结果仍标探索性。确认期不做。无故障任务直接复用已封存预测。

故障目录：

| 编号 | 注入 |
|---|---|
| R1 | 用 Track 1 中 Informer INF01 的存档塌缩预测替换一个成员的该任务输出 |
| R2 | 用 Lag-Llama smoke 的近零输出，按同一市场替换一个成员 |
| R3 | 记录为成员启动失败：导入冲突。不重新训练，直接把该成员标为错误 |
| R4 | 记录为检查点参数不匹配，该成员标为错误 |
| S1 | 推理时间超过 T2-Q 的超时 |
| S2 | 在 1 至 3 个步长写入 NaN 或 inf |
| S3 | 删掉 1 天，使输出长度为 27 |
| S4 | 把历史末日改到起点之后 |
| S5 | 该成员记为 CUDA OOM |

R1–R4 在抽中的故障里占一半，S1–S5 占一半；类型在各自半区内均匀。每个系统各跑故障率 1%、5%、10% 一次。时间表种子 `20261005`，生成后立刻计算 SHA256，写入 `track2/results/fault_injection/schedule_sha256.txt`。四个系统共用这一份时间表：

| 系统 | 规则 |
|---|---|
| T2-A0′ | 只跑 M1，无质量检查、无回退 |
| T2-C2 | M1 加 T2-Q，失败则 SN01 |
| T2-A1 | 三模型确定性协调器，按第 7 节回退 |
| T2-A2 | DeepSeek 协调器；工具内的数值规则与 T2-A1 相同 |

报告：最终结构检查通过率、故障检出率、未注入任务的误报率、静默失败率、自动恢复率和额外耗时、分故障率与分类型的 MASE(7)、人工介入次数。DeepSeek 若试图改写预测，记违规并按 T2-A1 的工具结果交付。

R1、R2 的存档路径在开跑前写进 `track2/configs/fault_sources.yaml`。找不到对应 parquet 时，只跑 S2–S5 与可复现的 R3/R4 标志，并在报告里写明 R1/R2 缺档，不临时制造新的塌缩序列。

## 10. T2-COST 与合并比较

GPU 费用沿用 Track 1：RunPod Secure Cloud A6000，按 `track1/configs/volume_breakeven_plan.yaml` 与 `track1/evaluation/volume_breakeven/` 的计价日期。三成员费用相加，再加上协调开销。T2-A2 另加 DeepSeek token 费用。CPU 对照（T2-C1、季节性朴素、ETS）沿用 Track 1 的 CPU 口径。研究者工时不计入货币成本。

增量链固定为：

```text
季节性朴素 → ETS03 → T2-C1 → M1 → T2-C2 → 冻结集成（T2-03 或 T2-04 中冻结的那一个）→ T2-05 → T2-A1 → T2-A2
```

每一步报告：外层宏平均 MASE(7) 的变化、每 1,000 个成功任务费用的变化、故障实验成功率和静默失败率的变化。体量平衡点沿用 Track 1 的算法，分别计算冻结集成相对 M1、相对 ETS03 的任务量门槛。帕累托图的横轴是每 1,000 任务费用，纵轴是外层 MASE(7)，点包含 Track 1 主表模型与上表 Track 2 系统。

若等权优于学习权重，或 M1 优于集成，表格保留该结果。复杂度不作为排名依据。

## 11. T2-O34 前瞻确认期（当前不运行）

`shared/data` 的现有序列止于 2026-07-31。在下面三折的数据导入之前，不要创建确认期预测，也不要做故障注入。

| 折 | 起点 | 目标 |
|---|---|---|
| O34 | 2026-07-12 | 2026-07-13 至 2026-08-09 |
| O35 | 2026-08-09 | 2026-08-10 至 2026-09-06 |
| O36 | 2026-09-06 | 2026-09-07 至 2026-10-04 |

导入时记录下载日期、来源和 SHA256，并核对 2026-07-31 之前与旧文件重叠的日期是否被修订。O34 前 19 天（7 月 13 日至 31 日）已在旧文件中且从未作为评分目标；确认期报告同时给出含这 19 天和去掉这 19 天的两个数。确认期只有 54–81 个澳门任务，只比较方向、效应量和区块 bootstrap 区间，不单独做显著性结论。配置在导入前必须已经提交，运行中不再改。

## 12. 建议执行顺序

```text
T2-X0
  → 第 5 节交叉拟合（E0–E4、EH、K、R、P、Q）
  → 写三份冻结文件并核对 SHA256
  → T2-01..06、T2-C1、T2-C2、香港、180、365、疫情窗
  → 独立评分与 T2-BOOT
  → T2-A0、T2-A1
  → 填入 DeepSeek 密钥后跑 T2-A2
  → T2-FI
  → T2-COST
  → 数据到达后再跑 T2-O34
```

第 5 节结束前不要打开外层评分。T2-A2 与 T2-FI 可以晚于准确率表；缺密钥时把这两项标为未运行，不使用其他模型代替 DeepSeek。

## 13. 完成标准

- [ ] T2-X0：三成员 B 组 12/12、外层 33/33，澳门任务键对齐，28 个有限值。
- [ ] B 组交叉拟合表含 E0–E4、步长权重是否启用、残差 `lambda`、概率方案。
- [ ] `track2_ensemble_config.yaml`、`track2_fallback_policy.yaml`、`track2_freeze_manifest.json` 已写 SHA256，且外层脚本能核对。
- [ ] 外层澳门表含 T2-01 至 T2-06、T2-C1、T2-C2，并标明探索性。
- [ ] 香港、180、365、A 组使用同一冻结配置。
- [ ] 无故障任务上 A0、A1、A2 数值逐位一致；A2 的违规次数有日志。
- [ ] 故障时间表 SHA256 在开跑前已保存；四系统共用。
- [ ] bootstrap、费用和帕累托使用冻结预测，没有回头改权重。
- [ ] O34–O36 在新数据导入前保持未运行。
- [ ] `llm_calls.jsonl` 中的 `model` 全部以 DeepSeek 型号字符串开头，没有其他厂商。

