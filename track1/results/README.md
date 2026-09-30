# Apple Silicon Mac 运行结果

本目录只记录这一台 Mac 上的运行。它与 Windows 上基于 commit `256ff66` 的 B 组运行是两次独立实验，预测分片不能拼接，也不能用两台机器各完成一部分后共同冻结。

## 代码与环境

- 机器：MacBook Pro（Mac16,1），Apple M4，10 核，16 GB 统一内存
- macOS：26.6.2
- Python：Homebrew Python 3.12.14，ARM64 原生虚拟环境 `.venv`。正式计算使用 CPU，没有启用 MPS
- 基线 commit：`b404117ffd2d29003c6e2f12277d9293898975ca`
- 本次运行 commit：`dfd5359f85f51ad4c3963e9e552a181a9f209177`
- `dfd5359` 相对 `b404117` 只改了两处执行设置：Transformer 使用 `torch.set_num_threads(1)`，并且每个模型单独建立进程池。这是为了避免 LightGBM 与 Transformer 在同一进程里共用 OpenMP 时死锁。搜索空间、协议和模型结构没有改
- B 组 workers：3。外层回测程序本身是单进程
- 硬件记录中的序列号、硬件 UUID 和 Provisioning UDID 已在发表前删除

## 正式结果

### `b_validation/`

B 组候选验证，2026-09-25 16:43 至 2026-09-26 09:14。

- 12,636 / 12,636 个预测分片，失败 0
- 未使用 `--max-tasks` 或 `--allow-dirty-code`
- `run_manifest.json` 中 `code_dirty=false`，`target_artifact_accessed=false`
- 预测阶段只读取 B 组 `task_inputs.parquet`，SHA256 为 `78ea345abd02a128222447d2924352eea32eca6367a1d05a114bf86e8a45e77f`
- 相同配置的失败重试跳过了全部 12,636 个已完成分片，没有新的失败
- 进程内计算时间约 14 小时 16 分。墙钟约 16 小时 31 分，其中约 2 小时为合盖睡眠

### `outer_backtest/`

外层历史回测，2026-09-26 11:03 至 2026-09-30 03:25。

- 范围：`outer_backtest`、历史长度 `all`、种子 42
- 957 个任务，13 个已冻结模型，348,348 行预测
- 未使用 `--max-tasks`
- `target_artifact_accessed=false`
- 输入 `task_inputs.parquet` 的 SHA256 为 `7ab6ebb51e3146b258f60454b45fefb20e84dccec9619362f6651e0ca6a0999e`
- 冻结配置 SHA256 为 `f3db59140eda9d6161431b1cb9d44bc09cd853b70beb854d3d519eb01ec247e1`
- `predictions.parquet` 的 SHA256 为 `f0d7503eb99566b513bde74b5754a9d360d3a93f3095175ebbd276972f01d471`

评价在 `track1/evaluation/`，不在本目录。评价才读取 `shared/data/targets.parquet`。

## 不能当作正式结果

- `mac_m4_pilot/`：第一轮 smoke。LightGBM 完成后进入 Transformer 时发生 OpenMP 死锁，只有 21 个分片，运行被终止
- `mac_m4_pilot_threadfix/`：线程修复后的 smoke，每个模型最多 3 个任务，共 57 个分片
- `outer_backtest_smoke/`：外层命令检查，只有 1 个任务

## 冻结配置

`track1/configs/frozen_baseline_configs.yaml` 来自这次 Mac B 组评价。状态是 `frozen_after_B_group_tuning`。Seasonal Naive 固定为 `SN01`，其 `validation_score` 按协议为空；评价汇总中的任务级 MASE(7) 宏平均是 1.206266。其余 12 个模型按 `mean_MASE7`、`mean_fit_seconds`、`candidate_id` 选出。

外层指标由 `track1/code/evaluate_predictions.py` 计算：先得到 28 天任务级指标，再在同一目的地内对任务等权平均。澳门与香港分开报告，没有合成一个总分。
