# Shared：两个 Track 的共用部分

`shared/` 只保存 Track 1 与 Track 2 必须共同使用的制品：

- `configs/protocol.yaml`：统一的 28 天预测协议和全部滚动起点；
- `data/task_inputs.parquet`：模型可读历史；
- `data/targets.parquet`：仅评估器可读真实目标；
- `data/task_index.parquet` 与 `data/task_manifest.json`：任务索引、数量和 SHA256；
- `data/splits/`：A、B、外层输入分片；
- `requirements.txt`：CPU 数据管道和 baseline 的共用依赖。

模型运行环境不得挂载 `data/targets.parquet`。Track 2 应复用这些任务 ID 和时间边界，不另外生成一套不一致的日期划分。
