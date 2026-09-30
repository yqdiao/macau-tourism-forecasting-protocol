# Apple Silicon Mac 评价结果

这些评价读取 `shared/data/targets.parquet`。预测目录本身没有读取该文件。目标文件 SHA256 为 `1ec5dee35f3846c8a51fbe4d5faab9ac536a853944c84d43bbb56dffdc092715`。

对应的预测、日志和运行说明在 `track1/results/README.md`。这是 Mac 运行，commit 为 `dfd5359f85f51ad4c3963e9e552a181a9f209177`，不要与 Windows commit `256ff66` 的结果合并。

## `b_validation/`

B 组选参评价，2026-09-26 10:14 完成。`validation_status.json` 中 `complete=true`，12,636 条路径全部合格。

- 任务级指标：`task_metrics.parquet`，SHA256 `2517cba909f10f4bc374dc0922ff145a0e7f3c8979aff6b885c67e5915a4597a`
- 候选汇总：`candidate_summary.parquet`，SHA256 `fa15917e86654ffa792b763432e5e5fb2d8eeb949fe4f02d261e13ae41773b68`
- 汇总方式：每条 28 天路径先计算任务级 MASE(7)；Transformer 先在同一任务内对种子取平均；再在市场内对折取平均，然后对市场等权

冻结配置 SHA256 为 `f3db59140eda9d6161431b1cb9d44bc09cd853b70beb854d3d519eb01ec247e1`。

## `outer_backtest/`

外层评价，2026-09-30 完成。12,441 条模型—任务路径全部匹配到目标，没有无法计算的 MASE(7)。

- 任务级指标：`task_metrics.parquet`，SHA256 `31e8f6959e0a98859b178d017e4ff8e6abb5448048734377f1226ca4c3f5627c`
- 汇总：`aggregate_metrics.parquet`，SHA256 `545b38f90a6929818fd74aa54341bae993d0926b0396873a1786a5cfa0a85190`
- 逐行评分：`scored_predictions.parquet`，SHA256 `1e19d961c557c50653ddb4740b516829c2de6862497f79a3b2f7305f2f322dd4`
- 汇总方式：28 天任务级指标之后，在同一目的地内对任务等权平均。澳门 891 个任务与香港 66 个任务分开，没有混合成一个总分

澳门外层任务级 MASE(7) 从低到高为：SVR 1.446、Gradient Boosting 1.513、LightGBM 1.515、Random Forest 1.516、Parrot Random Forest 800 1.516、Transformer 1.591、Lasso 1.614、ElasticNet 1.616、Ridge 1.624、Linear Regression 1.626、ETS 1.639、SARIMA 1.751、Seasonal Naive 1.883。
