# Baseline B 组验证与冻结审计

## 完成状态

- 完成时间：2026-09-26 11:38（Asia/Shanghai）
- 验证折：B01–B12
- 候选配置总数：33
- 已完成预测路径：12,636 / 12,636
- 终止失败路径：0
- 未解决路径：0
- 选择指标：28 天任务级 MASE(7)，再对市场、折次和 seed 宏平均
- 参数状态：`frozen_after_B_group_tuning`
- 运行模型代码 commit：`256ff6652c68bc418ac4c4d43c5d66610ebc991b`

## 冻结结果

| 模型 | 入选候选 | B 组 MASE(7) | 平均拟合秒数 |
|---|---:|---:|---:|
| Seasonal Naive | SN01 | 固定基准 | 0.0000 |
| ETS | ETS03 | 1.047266 | 0.5076 |
| SARIMA | SAR03 | 1.200664 | 1.5655 |
| Ridge | RID02 | 1.093038 | 0.2325 |
| Linear Regression | LIN01 | 1.311860 | 0.2865 |
| Random Forest | RF03 | 1.258608 | 229.6868 |
| Gradient Boosting | GB01 | 1.262905 | 55.5233 |
| SVR | SVR02 | 1.154077 | 14.2630 |
| Lasso | LAS03 | 1.106368 | 1.3168 |
| Elastic Net | EN03 | 1.101505 | 1.8070 |
| Parrot Random Forest 800 | PRF01 | 1.259145 | 871.4802 |
| LightGBM | LGB01 | 1.254014 | 37.1744 |
| Transformer baseline | TR03 | 1.288447 | 56.1705 |

Seasonal Naive 是预先固定的基准，不参与候选优选，因此没有 B 组验证分数。

## 已提交的审计制品

- `validation_status.json`：完整性、任务覆盖、源文件哈希和冻结状态。
- `driver_status.json`：本机驱动程序最终状态与退出码。
- `candidate_summary.parquet`：33 个候选配置的汇总结果。
- `frozen_baseline_configs.generated.bin`：评价器完成冻结时写出的原始 CRLF 字节，用于复核完成时 SHA256。
- `../../configs/frozen_baseline_configs.yaml`：13 个 baseline 的最终冻结选择。

为避免提交大体积中间结果，本目录不包含 12,636 个预测分片、运行日志和逐任务指标文件。`validation_status.json` 保存了候选汇总、逐任务指标、受保护目标文件和冻结配置在完成时的 SHA256，可用于核验本地归档。

## 完成时关键 SHA256

| 制品 | SHA256 |
|---|---|
| B 组任务清单 | `1074e3be580a1a2cea63aaa502ad0c2267a15ff2dc01e5c5f78e04f8fcce928d` |
| 候选汇总 | `e51cab19c98f26257bd6db0bdd37ba02a01cedaa35e7a18fd6e244c0e8204cdf` |
| 逐任务指标 | `e7eb223db9f4e2e95df3dc2b2c4a0f1da08b4a1a8513599411f9812b1d7898a9` |
| 受保护目标 | `1ec5dee35f3846c8a51fbe4d5faab9ac536a853944c84d43bbb56dffdc092715` |
| 冻结配置原始归档 `frozen_baseline_configs.generated.bin` | `1771fcb4d4605a139db8e79c0a6533679afa002bce0aa48289de7eaa65889595` |

Git 工作配置按仓库规则规范化为 LF，语义内容与原始归档相同。完成时字节级 SHA256 应对照 `.bin` 归档；外层运行使用 `track1/configs/frozen_baseline_configs.yaml`。

外层回测只能读取冻结配置和模型输入。`targets.parquet` 仍只允许独立评价器读取。
