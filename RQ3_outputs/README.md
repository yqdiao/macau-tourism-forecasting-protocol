# RQ3_outputs 使用说明

## 权威文件
- 表：`tables/table_S14`–`table_S18`
- 图：`figures/figure11_*`, `figures/figure12_*`
- DeepSeek 判断：`cache/llm_judgments_deepseek.jsonl`
- DeepSeek 决策：`cache/all_decisions_deepseek.parquet`
- 核对汇总（含 **W1–W7 写作口径标记**）：`4.3_RQ3_实验数据核对汇总稿.md`
- 缺口审计：`logs/rq3_output_gap_audit_final.md`
- 质量检查：`logs/rq3_output_quality_checks.csv`

## 【W7】不要用于正文（旧 proxy 缓存）
- `cache/llm_judgments.jsonl` / `*_fill.jsonl` / `all_decisions.parquet`：早期 llm_proxy  
- 正文、表注、复现说明一律引用 `*_deepseek.*`

## 写作口径速查（非漏跑）
详见汇总稿「【写作口径标记】」与 `logs/rq3_output_gap_audit_final.md` §B：

| 标记 | 一句话 |
|:-----|:-------|
| **W1** | G3≈G1 是实证，不是漏跑 |
| **W2** | Mode2 只改成本，精度=Mode1 |
| **W3** | S18 用云 API wall time 作 CPU 等价，非本地 GPU |
| **W4** | S14 组均值无 bootstrap CI |
| **W5** | S15 DM 以 MASE/MAE 为主 |
| **W6** | Fig12c 简；算力看 S17 `models_run` |
| **W7** | 勿引用旧 proxy 缓存 |

## 复现
```bash
# 统计+proxy 初跑
python RQ3/run_all.py
# 缺口补表
python RQ3/fill_gaps.py
# DeepSeek 真实 LLM（需 RQ3/.env）
python RQ3/run_deepseek_rq3.py
```
