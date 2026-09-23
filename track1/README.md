# Track 1：单模型统一回测

Track 1 比较 13 个 baseline 与 7 个时序模型，使用相同的单变量日度输入、28 天 horizon 和滚动起点。

目录说明：

- `code/`：可执行 Python 代码和七模型 adapter；
- `configs/`：baseline 搜索与冻结配置、七模型上游 revision；
- `compose/`、`docker/`：CPU baseline 和 CUDA 模型环境；
- `docs/`：研究协议、审计和服务器手册；
- `notebooks/`：数据管道与统一回测 notebook；
- `scripts/`：B 组长时运行及 GPU 批量脚本；
- `tests/`：接口、28 天输出和目标隔离测试。

所有命令从仓库根目录运行。共用输入位于 `shared/data/`，运行结果写入未跟踪的 `track1/results/` 和 `track1/evaluation/`。
