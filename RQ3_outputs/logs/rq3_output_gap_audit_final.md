# RQ3 输出核对与缺口说明（终版）

生成时间依据：DeepSeek 流水线 `rq3ds` 已完成（2376 API 全成功）；本文件为输出清理与缺口审计。

## 一、当前权威产出（请以此为准）

| 路径 | 角色 |
|------|------|
| `tables/table_S14_*.csv` | 固定组合基准（含 vs ETS/Granite/B3FM/G1） |
| `tables/table_S15_*.csv` | 2×2 动态消融；**LLM=DeepSeek API** |
| `tables/table_S16_*.csv` | 选模行为（市场/假日/星期）；与 DeepSeek 决策一致 |
| `tables/table_S17_*.csv` | λ/α/δ/K/驱动指标敏感性（G1） |
| `tables/table_S18_*.csv` | Mode1/2 成本与可靠性（已按 DeepSeek 实测时延重算） |
| `figures/figure11_*.pdf/png` | 主文图 11（含 DeepSeek 消融） |
| `figures/figure12_*.pdf/png` | 主文图 12 |
| `cache/llm_judgments_deepseek.jsonl` | **真实 LLM 判断缓存（权威）** |
| `cache/all_decisions_deepseek.parquet` | **动态方案决策轨迹（权威）** |

**过时/勿混用：** `cache/llm_judgments.jsonl`、`llm_judgments_fill.jsonl`、`all_decisions.parquet` 为早期 `llm_proxy` 产物，仅作对照，正文勿引用。

## 二、自动质量检查

通过 **12/12**（明细见 `logs/rq3_output_quality_checks.csv`）。

关键数值（重新开放期）：
- Granite MASE=1.0944；B_3FM Equal MAE=542.29（优于 Granite MAE）
- G1 MASE=1.1519；**G2 DeepSeek-Select=1.1411**；**G4=1.1407–1.1409**；G3≈G1（配权几乎无边际）
- DeepSeek API：ok=2376 fail=0；平均时延≈1.522s（p50≈1.422s）
- 权重 ∑w−1 最大偏差 < 1e-15 量级；动态方案 fallback=0

## 三、已齐 / 部分齐 / 仍缺

### A. 已齐（可直接写正文）
1. M5 候选池与 C1–C9 / B_3FM / B_5Pool 两时期四指标 + CI + DM  
2. 无泄露回放：`target_end(r)≤origin(t)`；冷启动等权 C2  
3. Stat-Select / Stat-Weight 与 2×2 消融（G1–G4，η∈{0.3,0.5,1.0}，Mode1/2）  
4. **真实 LLM**：DeepSeek `deepseek-chat`（非 Qwen3-4B；需在正文改名）  
5. S16 市场组 + 假日窗口 + 激活率；S17 网格；S18 成本情景  
6. Figure 11/12 已按 DeepSeek 结果重绘  
7. 单模型与 Table3 对齐（1e-13）

### B. 部分齐 / 表述需注意（= 汇总稿 **W1–W7** 写作口径标记）

> 下列不是“没跑”，而是**正文/表注必须显式交代**。汇总稿 `4.3_RQ3_实验数据核对汇总稿.md` 已用同一标记号标出。

| 标记 | 条目 | 说明 |
|:-----|:-----|:-----|
| — | **LLM 型号替换** | 规格原写 Qwen3-4B，现为 DeepSeek；有意替换，不是漏跑。 |
| **W1** | **G3≈G1** | DeepSeek 在固定 Stat 子集上的配权与统计逆误差几乎同质，边际≈0——**实证结果，不是表缺失**。 |
| **W2** | **Mode2 = Mode1 精度** | 离线封存预测下 Mode2 只改“跑哪些模型”的成本账，**点预测相同**——正文必须写明。 |
| **W3** | **S18 API 成本** | DeepSeek 为云 API；实测 **wall time** 计入条件费用作 **CPU 等价**；**不是本地 GPU 秒**。 |
| **W4** | **S14 组均值无 CI** | 结构组均值仅点估计，无 bootstrap CI。 |
| **W5** | **S15 DM 范围** | 对 ETS/G1 的 DM 以 **MASE/MAE 为主**；MAPE/RMSE 有 R 列、DM 未全四指标展开。 |
| **W6** | **Figure 12c 较简** | 主画 K–MASE；算力节省细节读 S17 `models_run`。 |
| **W7** | **旧 proxy 勿引用** | `llm_judgments.jsonl` / `*_fill.jsonl` / `all_decisions.parquet` 仅对照；正文只用 `*_deepseek.*`（见 README）。 |
| — | **正观测过滤** | 重开 24,897 / 压力 3,018 正观测已在 MAPE 的 `ape` 中执行；压力期有效正观测任务数未单独成表。 |

### C. 仍缺（无法用现有封存数据“假装完成”）
1. **warm_start 内层先验**：无 M5 的 B-fold/`inner` 已完成任务误差库 → `not_run`。  
2. **本地 Qwen3-4B GGUF 推理**：无 NVIDIA 驱动；已用 DeepSeek API 替代。  
3. **若坚持字面 Qwen3-4B**：需 DashScope 等同款模型 API 或修复 GPU 后重刷判断缓存。

## 四、对汇总稿的修正建议（已执行重写）

- 删除“当前为 llm_proxy”的过时表述  
- 明确 LLM=DeepSeek、权威缓存路径  
- 压缩巨型宽表粘贴，改为关键结果 + 指向 CSV  

## 五、写作时推荐引用句

> 第 4.3 节语言模型判断模块通过 DeepSeek API（`deepseek-chat`，temperature=0）实现；共 2,376 次结构化 JSON 调用全部通过校验，零回退。相对纯统计基线 G1，DeepSeek 动态选模（G2/G4）在重新开放期将 MASE 由 1.152 降至约 1.141；单独替换配权（G3）几乎无增益。
