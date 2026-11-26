# RQ3 补全后完成度核对

自动检查 6/6 通过。

## 已补全
- S15：相对 ETS / Granite / G1 / B3FM 的 R 与 DM（MASE、MAE）
- S14：相对 G1 的 R/DM（及 ETS/Granite/B3FM 对齐列）
- S16：大中小市场 + 假日窗口 + 周末窗口 + 假日分层激活率
- S18：`llm_backend` 标注；权重偏差列
- 缺失项清单：`logs/missing_items_manifest_rq3.csv`

## 仍无法补全（环境/数据限制）
- 真实 Qwen3-4B 选模/配权（无 NVIDIA 驱动）
- 冷启动 warm_start 内层先验（仓库无 M5 的 inner completed-task 误差库）
- Figure 11/12 未因本次列增补而重绘（主文图数据源仍有效；S16 假日分层为表级增补）
