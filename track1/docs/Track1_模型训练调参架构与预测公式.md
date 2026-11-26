# Track 1 非零样本模型：输入、架构、训练、调参、预测公式与结果

本文只整理**需要训练或需要在 B 组选参**的模型。五个零样本基础模型（Granite / Chronos-2 / TimesFM 3.0 / Moirai / Lag-Llama）按协议不调参、不训练，不在本文展开。

代码以仓库当前实现为准：

- 基线：`track1/code/models.py`
- PatchTST / Informer：`track1/code/foundation_models/adapters/{patchtst,informer,_trainable}.py`
- 轻量家族：`track1/code/light_models.py`
- 搜索与冻结：`track1/configs/{baseline_search_spaces,foundation_search_spaces,light_model_plan,light_model_round2,frozen_*}.yaml`

---

## 0. 覆盖清单

| 家族 | 模型 | 冻结候选 | 训练方式 | 种子 |
|---|---|---|---|---|
| 统计 | 季节性朴素 | SN01 | 不训练，不调参 | 42 |
| 统计 | ETS | ETS03 | 每任务拟合 Holt-Winters | 42 |
| 统计 | SARIMA | SAR03 | 每任务拟合 SARIMAX | 42 |
| 表格直接 | 岭回归 | RID02 | 每任务 28 个直接回归器 | 42 |
| 表格直接 | 普通最小二乘 | LIN01 | 同上 | 42 |
| 表格直接 | Lasso | LAS03 | 同上 | 42 |
| 表格直接 | Elastic Net | EN03 | 同上 | 42 |
| 表格直接 | RBF-SVR | SVR02 | 同上 | 42 |
| 表格直接 | LightGBM | LGB01 | 同上 | 42 |
| 表格直接 | 梯度提升 | GB01 | 同上 | 42 |
| 表格直接 | 随机森林 | RF03 | 同上 | 42 |
| 表格直接 | 随机森林 800 | PRF01 | 同上 | 42 |
| 小型网络 | Transformer | TR03 | 每任务从零训练编码器 | B 组 11/29/42；外层 42 |
| 可训练基础 | PatchTST | PT06 | 每任务从零监督训练 | 11, 29, 42 |
| 可训练基础 | Informer | INF07 | 每任务从零监督训练 | 11, 29, 42 |
| 轻量全局 | 全局 LightGBM | GL03 | 每起点一个模型，27 市场合训 | 42 |
| 轻量全局 | NLinear | NL01 | 每起点一个共享线性图 | 11, 29, 42 |
| 消融（不选参） | GL-noCal / GL-local | GL03 参数 | 去掉日历 / 改为逐市场 | 42 |
| 事后第 2 轮 | 逐市场 LightGBM | LL01 | 每市场一个模型，保留日历 | 42 |

DLinear 只作为 NL03 进入 B 组，未冻结，外层不跑。

---

## 1. 共享协议

### 1.1 任务与切分

每个任务是一条客源市场日度序列，在起点 \(o\) 上预报未来 28 天 \(\hat y_{o+1},\ldots,\hat y_{o+28}\)。主历史设置是起点之前的全部观测（`history=all`）。输入只来自该任务自己的历史计数；香港目的地序列不进入 B 组选参，轻量模型也不跑香港。

| 组别 | 用途 | 澳门任务 |
|---|---|---|
| A01–A13 | 疫情压力，只报告，不选参 | \(27\times 13=351\) |
| B01–B12 | 2023 年恢复期，**唯一选参窗口** | \(27\times 12=324\) |
| O01–O33 | 外层回测 | \(27\times 33=891\)（基线/可训练基础另含香港 2 条，共 957） |

外层起点从 2023-12-31 起每隔 28 天一个。主窗口最后规则起点 2026-06-14，目标截止 2026-07-12。受保护目标 SHA256：`1ec5dee35f3846c8a51fbe4d5faab9ac536a853944c84d43bbb56dffdc092715`。预测器不得读取 `targets.parquet`。

### 1.2 主指标与宏平均

季节性朴素尺度（周期 7）在起点当时的历史上计算，预测器写入 `mase7_scale`，评价器只做除法：

\[
s_7 = \frac{1}{T-7}\sum_{t=8}^{T}\lvert y_t-y_{t-7}\rvert,\qquad
\mathrm{MASE}(7)=\frac{\frac{1}{28}\sum_{h=1}^{28}\lvert\hat y_{o+h}-y_{o+h}\rvert}{s_7}.
\]

\(s_7=0\) 的任务从 MASE 宏平均中剔除，MAE 仍保留。选择与主表都用同一套聚合：多种子先对任务平均 → 市场内各折平均 → 27 个市场等权。并列规则：更低的平均拟合秒数，再取更小的候选编号。**外层、A 组、180/365 敏感性、香港分数一律不得回头改候选。**

### 1.3 共同输出

每个任务写出 28 行预测。基线/可训练基础还写 `prediction_raw`、`was_clipped`、拟合/推理秒数。最终用于评分的点预测都截到 \(\hat y=\max(0,\hat y_{\mathrm{raw}})\)。评价产物：

- 基线 B：`track1/evaluation/b_validation_server_repro/`，冻结 `track1/configs/frozen_baseline_configs.yaml`（2026-09-26）
- PatchTST/Informer B：`track1/evaluation/foundation_b_search/candidate_summary.csv`，冻结 `track1/configs/frozen_foundation_configs.yaml`（2026-10-01）
- 外层主表：`track1/evaluation/outer_main_table/main_table.csv`
- 敏感性：`track1/evaluation/sensitivity/summary.csv`
- 轻量 B / 外层 / 消融 / 第 2 轮：`track1/evaluation/light_models_{b,extra,round2}/`

---

## 2. 季节性朴素（SN01）

**不训练、不调参。** 搜索空间只有 `{"period": 7}`。B 组 repro MASE 1.206，不参与优选。外层澳门 MASE 1.883。

### 2.1 输入

只需最近 7 个已观测日。实现要求整条历史至少 90 天且非负有限，但公式只用最后一周。

### 2.2 预测公式

把起点前最后一周按天循环铺满 28 天，**绝不使用预测块内部的值**：

\[
\hat y_{o+h}=y_{o-6+\bigl((h-1)\bmod 7\bigr)},\qquad h=1,\ldots,28.
\]

代码：`np.resize(y[-7:], 28)`。拟合时间记 0。

---

## 3. ETS（指数平滑）

实现：`statsmodels.tsa.holtwinters.ExponentialSmoothing`，在 \(z_t=\log(1+y_t)\) 上拟合，预报后再 `expm1`。

### 3.1 输入

该市场起点前全部日度计数 \(y_{1:T}\)，\(T\ge 90\)。无协变量。

### 3.2 已登记搜索与 B 组结果

固定：`transform=log1p`，加法周季节，`seasonal_periods=7`，`initialization_method="estimated"`，`optimized=True`。只搜趋势。

| 候选 | 趋势 | 阻尼 | B 组 MASE（repro） | 入选 |
|---|---|---|---:|---|
| ETS01 | 无 | — | 1.051 | 否 |
| ETS02 | 加法 | 否 | 1.055 | 否 |
| ETS03 | 加法 | 是 | 1.046（冻结分 1.047） | **是** |

冻结 2026-09-26：ETS03，B 官方分 1.047266，平均拟合 0.508 秒。B 组最好的基线，成本对比因此固定为 Granite 对 ETS，不用外层才更好的 SVR。

### 3.3 冻结架构（ETS03）

加法阻尼 Holt–Winters，季节周期 \(m=7\)。记水平 \( \ell_t \)、趋势 \( b_t \)、季节 \( s_t \)，阻尼 \(\phi\in(0,1]\)，平滑系数 \(\alpha,\beta,\gamma\) 由极大似然估计：

\[
\begin{aligned}
\ell_t &= \alpha\bigl(z_t-s_{t-7}\bigr)+(1-\alpha)\bigl(\ell_{t-1}+\phi b_{t-1}\bigr),\\
b_t &= \beta\bigl(\ell_t-\ell_{t-1}\bigr)+(1-\beta)\phi b_{t-1},\\
s_t &= \gamma\bigl(z_t-\ell_{t-1}-\phi b_{t-1}\bigr)+(1-\gamma)s_{t-7}.
\end{aligned}
\]

### 3.4 预测公式

\[
\hat z_{T+h}=\ell_T+\Bigl(\sum_{i=1}^{h}\phi^i\Bigr)b_T+s_{T-7+((h-1)\bmod 7)+1},\qquad
\hat y_{T+h}=\mathrm{expm1}(\hat z_{T+h}),\ \hat y\leftarrow\max(0,\hat y).
\]

### 3.5 结果

| 窗口 | 澳门 MASE(7) |
|---|---:|
| B 冻结 | 1.047 |
| 外层 all | 1.639（95% 1.571–1.713） |
| 外层 180 / 365 | 1.116 / 1.140 |
| A 组 all | 0.197 |

外层相对朴素胜率 0.776。隔离计时约 \$0.019 / 1000 任务。

---

## 4. SARIMA

实现：`statsmodels.tsa.statespace.sarimax.SARIMAX`，同样在 \(z=\log1p(y)\) 上拟合，`enforce_stationarity=False`，`enforce_invertibility=False`，`maxiter=50`。

### 4.1 输入

该市场全部历史。无外生回归项。季节周期必须为 7。

### 4.2 搜索与 B 组

| 候选 | \((p,d,q)\) | \((P,D,Q,7)\) | 截距 | B MASE | 入选 |
|---|---|---|---|---:|---|
| SAR01 | (1,0,0) | (1,0,0,7) | c | 1.387 | 否 |
| SAR02 | (1,1,0) | (1,0,0,7) | n | 1.374 | 否 |
| SAR03 | (0,1,1) | (0,1,1,7) | n | 1.201 | **是** |

冻结 SAR03，官方 B 1.200664，平均拟合 1.565 秒。

### 4.3 冻结方程（SAR03）

无截距的航空旅客模型：

\[
(1-B)(1-B^7)z_t=(1+\theta B)(1+\Theta B^7)\varepsilon_t.
\]

展开：

\[
z_t=z_{t-1}+z_{t-7}-z_{t-8}+\varepsilon_t+\theta\varepsilon_{t-1}+\Theta\varepsilon_{t-7}+\theta\Theta\varepsilon_{t-8}.
\]

### 4.4 预测与结果

多步预报用状态空间 Kalman 预测，再 \(\hat y=\max\bigl(0,\mathrm{expm1}(\hat z)\bigr)\)。

| 窗口 | 澳门 MASE(7) |
|---|---:|
| B | 1.201 |
| 外层 all | 1.751 |
| 外层 180 / 365 | 1.229 / 1.203 |
| A 组 | 0.196 |

A 组上 SARIMA 与 ETS、TimesFM 同一档；外层全历史上明显差于 ETS。

---

## 5. 表格直接回归：共享特征与训练

九个 sklearn / LightGBM 模型共用同一套管线 `tabular_direct` + `prepare_tabular_training_sets`。差别只在回归器。

### 5.1 输入（37 维，不含任何未来观测）

先做 \(z_t=\log(1+y_t)\)。对每个提前期 \(h=1,\ldots,28\) 独立建一个回归器 \(f_h\)。锚点 \(t\) 的特征只用 \(z_{\le t}\)：

**28 个滞后** \(z_t,z_{t-1},\ldots,z_{t-27}\)。

**8 个窗口统计**，窗长 \(w\in\{7,14,28,56\}\)（总体标准差，`ddof=0`）：

\[
\bar z_{t,w}=\frac{1}{w}\sum_{i=0}^{w-1}z_{t-i},\qquad
s_{t,w}=\sqrt{\frac{1}{w}\sum_{i=0}^{w-1}(z_{t-i}-\bar z_{t,w})^2}.
\]

**1 个同星期均值**。偏移 \(o_h=(7-h\bmod 7)\bmod 7\)，对应目标日 \(t+h\) 的星期：

\[
\bar w_{t,h}=\frac14\sum_{k=0}^{3}z_{t-o_h-7k}.
\]

于是 \(x_{t,h}\in\mathbb{R}^{37}\)。最短历史：实现要求 \(T\ge 56+28+20=104\)。有效上下文 56 天。无日历、无跨市场特征。

### 5.2 训练样本与损失

对固定 \(h\)，锚点 \(t=55,56,\ldots,T-1-h\)（0 下标下从第 56 天起，为 56 日窗留满样本）。目标：

\[
u_{t,h}=z_{t+h}=\log(1+y_{t+h}).
\]

每个 \(h\) 单独 `fit(X_h, u_{\cdot,h})`。线性族与 SVR 先 `StandardScaler`（训练集均值方差，推理用同一套）。树模型与 LightGBM 吃原始 37 维。种子 42。没有验证集、没有早停。

### 5.3 推理与逆变换

在 \(t=T-1\)（起点当天）构造 \(x_{T-1,h}\)，得到 \(\hat z_h=f_h(x_{T-1,h})\)，再

\[
\hat y_{T+h}=\mathrm{expm1}\bigl(\min(30,\hat z_h)\bigr),\qquad \hat y\leftarrow\max(0,\hat y).
\]

`min(30,·)` 防止 `expm1` 溢出。28 个回归器互不共享参数。

### 5.4 为什么是直接策略

一次拟合只学一个 \(h\)，避免把 \(h=1\) 的预测再喂给 \(h=2\)。代价是 28 套模型，随机森林类拟合很慢。

---

## 6. 线性表格模型

四个模型都是 `StandardScaler` + 线性估计器。记标准化后的特征为 \(\tilde x\)，预测 \(\hat z=w^\top\tilde x+b\)（Lasso / Elastic Net 的 \(b\) 同样估计）。

### 6.1 岭回归 RID02

\[
\min_{w,b}\ \lVert u-Xw-b\rVert_2^2+\alpha\lVert w\rVert_2^2.
\]

| 候选 | \(\alpha\) | B MASE | 入选 |
|---|---:|---:|---|
| RID01 | 1 | 1.162 | 否 |
| RID02 | 10 | 1.093 | **是** |
| RID03 | 100 | 1.125 | 否 |

冻结 B 1.093038，拟合 0.232 秒。外层 1.624；180 天 1.650；365 天 1.192；A 组 0.226。B 上很强，外层全历史掉到线性族中间，短历史 365 天明显回升。

### 6.2 普通最小二乘 LIN01

只有一个候选：`fit_intercept=True`。无惩罚。

\[
\min_{w,b}\ \lVert u-Xw-b\rVert_2^2.
\]

B 1.312（冻结）/ 1.315（repro）。外层 1.626。180 天崩到 3.359（样本少、无惩罚）；365 天 1.234；A 组 0.226。

### 6.3 Lasso LAS03

`max_iter=10000`。

\[
\min_{w,b}\ \frac{1}{2n}\lVert u-Xw-b\rVert_2^2+\alpha\lVert w\rVert_1.
\]

| 候选 | \(\alpha\) | B MASE | 入选 |
|---|---:|---:|---|
| LAS01 | 0.0001 | 1.306 | 否 |
| LAS02 | 0.001 | 1.160 | 否 |
| LAS03 | 0.01 | 1.106 | **是** |

B 组更强的惩罚更好。外层 1.614；180 天 1.696；365 天 1.148；A 组 0.228。

### 6.4 Elastic Net EN03

\[
\min_{w,b}\ \frac{1}{2n}\lVert u-Xw-b\rVert_2^2+\alpha\Bigl(\rho\lVert w\rVert_1+\frac{1-\rho}{2}\lVert w\rVert_2^2\Bigr).
\]

| 候选 | \(\alpha\) | \(\rho\) (l1_ratio) | B MASE | 入选 |
|---|---:|---:|---|---|
| EN01 | 0.001 | 0.3 | 1.123 | 否 |
| EN02 | 0.001 | 0.7 | 1.140 | 否 |
| EN03 | 0.01 | 0.3 | 1.102 | **是** |

与 Lasso 一样，B 组选了更强的 \(\alpha\)。外层 1.616；180 天 2.039；365 天 1.187；A 组 0.227。

---

## 7. SVR（SVR02）

`StandardScaler` + `sklearn.svm.SVR`，RBF 核，\(\varepsilon=0.01\) 固定。对每个 \(h\) 解 \(\varepsilon\)-SVR：

\[
\min_{w,b,\xi,\xi^*}\ \frac12\lVert w\rVert_2^2+C\sum_i(\xi_i+\xi_i^*),\quad
\lvert u_i-w^\top\phi(\tilde x_i)-b\rvert\le\varepsilon+\xi_i/\xi_i^*.
\]

核 \(k(\tilde x,\tilde x')=\exp(-\gamma\lVert\tilde x-\tilde x'\rVert^2)\)，\(\gamma=1/37\)（sklearn `scale`）。

| 候选 | \(C\) | B MASE | 入选 |
|---|---:|---:|---|
| SVR01 | 0.1 | 1.167 | 否 |
| SVR02 | 1.0 | 1.154 | **是** |
| SVR03 | 10 | 1.242 | 否 |

冻结 B 1.154077，拟合 14.3 秒。外层 **1.446，全部需训练模型里最好**（主表第 5，仅次于四个零样本）。180 天 1.133，365 天 1.183，A 组 0.288（疫情窗上偏弱）。隔离计时约 \$0.17 / 1000 任务。

---

## 8. 树模型

都不做特征标准化。`random_state=seed`，基线种子 42。目标仍是 \(\log1p\) 尺度。

### 8.1 随机森林 RF03

`sklearn.ensemble.RandomForestRegressor`，`n_jobs=2`，`min_samples_split=5`。每棵树对 bootstrap 样本做 CART 回归，预测为袋内树平均。

| 候选 | 树数 | 最大深度 | B MASE | 拟合秒 | 入选 |
|---|---:|---:|---:|---:|---|
| RF01 | 100 | 6 | 1.267 | 44 | 否 |
| RF02 | 100 | 10 | 1.265 | 62 | 否 |
| RF03 | 300 | 10 | 1.259 | 189 | **是** |

冻结官方 B 1.258608。外层 1.515；180 天 1.203；365 天 1.156；A 组 0.235。隔离计时约 \$3.77 / 1000 任务。

### 8.2 随机森林 800（PRF01）

对照更大的森林：800 棵，`max_depth=None`（不剪枝），无 `min_samples_split` 覆盖（sklearn 默认 2）。只有 PRF01。B 1.259145，与 RF03 几乎相同，拟合 871 秒。外层 1.516。隔离计时约 \$15.56 / 1000 任务。精度几乎不涨，成本大约四倍。

### 8.3 梯度提升 GB01

`sklearn.ensemble.GradientBoostingRegressor`，平方损失，学习率固定 0.05。第 \(m\) 棵树拟合残差 \(u-F_{m-1}(x)\)，\(F_m=F_{m-1}+\nu\,T_m\)。

| 候选 | 树数 | 深度 | B MASE | 入选 |
|---|---:|---:|---:|---|
| GB01 | 100 | 3 | 1.263 | **是** |
| GB02 | 100 | 6 | 1.287 | 否 |
| GB03 | 300 | 3 | 1.266 | 否 |

B 组浅树更好。外层 1.513（树模型里最好）；180 天 1.228；365 天 1.175；A 组 0.237。

### 8.4 逐市场 LightGBM（LGB01）

`lightgbm.LGBMRegressor`，`objective=regression`（L2），`min_child_samples=10`，`colsample_bytree=0.75`，`subsample=0.75`，`subsample_freq=1`，`n_jobs=2`，`verbosity=-1`。仍是 **每市场、每 \(h\) 一个模型**，与第 12 节全局 LightGBM 不是同一个适配器。

叶子生长：每次选增益最大的叶分裂，直到 `num_leaves` 或 `max_depth`。预测是 \(F(x)=\sum_{m=1}^{M}\nu\,t_m(x)\)。

| 候选 | 树数 | 学习率 | 叶数 | 深度 | B MASE | 入选 |
|---|---:|---:|---:|---:|---:|---|
| LGB01 | 100 | 0.05 | 31 | 8 | 1.254 | **是** |
| LGB02 | 300 | 0.05 | 31 | 8 | 1.277 | 否 |
| LGB03 | 100 | 0.05 | 63 | 10 | 1.257 | 否 |

B 组更少的树更好。外层 1.515；180 天 1.183；365 天 1.155；A 组 0.238。隔离计时约 \$0.17 / 1000 任务。

---

## 9. 小型 Transformer（TR03）

本地编码器回归，**不是**预训练基础模型。实现全部在 `models.py` 的 `DemandTransformer`。CPU，`torch.set_num_threads(2)`，`use_deterministic_algorithms(True)`。

### 9.1 输入与预处理

回看长度 \(L\)（候选 56 或 84）。对整段历史做一次全局标准化：

\[
z_t=\log(1+y_t),\quad
\mu=\mathrm{mean}(z),\ \sigma=\mathrm{std}(z)\ \text{或 }1,\quad
\tilde z_t=(z_t-\mu)/\sigma.
\]

监督窗：对 \(t=L-1,\ldots,T-1-28\)，

\[
X=\tilde z_{t-L+1:t}\in\mathbb{R}^{L\times 1},\qquad
Y=\tilde z_{t+1:t+28}\in\mathbb{R}^{28}.
\]

### 9.2 架构

1. 投影：\(h_i=W_{\mathrm{in}}\tilde z_i+b_{\mathrm{in}}\)，\(W_{\mathrm{in}}\in\mathbb{R}^{1\times d}\)，\(d=32\)。
2. 固定正弦位置编码（不训练）：
   \[
   \mathrm{PE}(p,2j)=\sin(p/10000^{2j/d}),\quad
   \mathrm{PE}(p,2j+1)=\cos(p/10000^{2j/d}).
   \]
3. 一层 `TransformerEncoderLayer`：`nhead=4`，`dim_feedforward=64`，`dropout=0.1`，`batch_first=True`。标准缩放点积多头注意力 + FFN。
4. 取最后一个时间步，线性头 \(W_{\mathrm{out}}\in\mathbb{R}^{d\times 28}\) 一次输出 28 步。

前向：

\[
\hat Y=\mathrm{Linear}\bigl(\mathrm{Enc}(h+\mathrm{PE})_{:,L,:}\bigr).
\]

参数量大约：投影 \(32+32\)，一层注意力与 FFN 约 \(4d^2+2d\cdot 64\)，头 \(32\times 28+28\)，合计约 \(10^4\) 量级。

### 9.3 训练

- 损失：标准化 \(\log1p\) 上的 MSE
- 优化器：Adam
- 批大小 64，打乱种子 = 运行种子
- **固定轮数，无验证、无早停**
- B 组三种子 11/29/42；外层冻结后只用种子 42

| 候选 | 回看 | 轮数 | 学习率 | \(d\) | B MASE（3 种子） | 入选 |
|---|---:|---:|---:|---:|---:|---|
| TR01 | 56 | 8 | 1e-3 | 32 | 1.300 | 否 |
| TR02 | 84 | 8 | 1e-3 | 32 | 1.321 | 否 |
| TR03 | 56 | 16 | 5e-4 | 32 | 1.288 | **是** |

更长回看没有帮助。冻结官方 B 1.288447，拟合 56.2 秒。

### 9.4 预测公式

取最后 \(L\) 步 \(\tilde z_{T-L+1:T}\)，网络输出 \(\hat{\tilde z}_{1:28}\)，再

\[
\hat z=\mathrm{clip}(\hat{\tilde z}\cdot\sigma+\mu,\ -20,\ 30),\qquad
\hat y=\max\bigl(0,\ \mathrm{expm1}(\hat z)\bigr).
\]

### 9.5 结果

外层 1.591（与 PatchTST 1.590 几乎相同）；180 天 1.187；365 天 1.159；A 组 0.246。隔离计时量级低于 PatchTST（CPU 小网），精度停在线性/树与可训练基础之间。

---

## 10. PatchTST（PT06）

上游固定检出：`/opt/upstream/PatchTST_supervised`。每个任务、每个种子从随机初始化训练，权重写到该任务的 `.pt`，**不跨任务、不跨起点复用**。必须 CUDA。

### 10.1 输入

单变量历史。训练至少需要 `context_length + 28` 天。主设置 `history=all`；180 天历史上因窗口不够记为无法训练。无协变量。时间标记不用。

### 10.2 外层标准化（适配器）

\[
z=\log1p(y),\quad
\mu=\mathrm{mean}(z),\ \sigma=\mathrm{std}(z)\ \text{或 }1.
\]

窗口 \(X=\bigl((z_{s:s+C}-\mu)/\sigma\bigr)\in\mathbb{R}^{C\times 1}\)，目标 \(Y\) 为随后 28 步同一变换。最多 4096 个窗；更多则在 \([0,T-C-28]\) 上均匀下采样。尾部 10% 窗（至少 1、至多 20%）做验证，按时间切，不打乱先后。

上游模型内部还有 RevIN（`revin=1, affine=0, subtract_last=0`）：每个窗再减自己的均值、除自己的标准差，输出加回去。因此实际是 **全局 log1p 标准化 + 实例 RevIN**。

### 10.3 冻结架构（PT06）

| 项 | 值 |
|---|---|
| 上下文 \(C\) | 180 |
| patch / stride | 16 / 8，`padding_patch=end` |
| 编码器层 / 头 | 3 / 4 |
| \(d_{\mathrm{model}}\) / \(d_{\mathrm{ff}}\) | 128 / 256 |
| dropout / fc_dropout / head_dropout | 0.2 / 0.2 / 0 |
| 通道 | `enc_in=1`，`individual=0` |
| 分解 | 关（`decomposition=0`） |
| 批大小 / 最大轮数 | 32 / 30 |
| 优化器 / 损失 | AdamW，lr \(10^{-4}\)；MSE |
| 早停耐心 | 10，恢复验证最优 |

patch 数大约 \((180-16)/8+1=21\)，端侧补丁后再加 1。每个 patch 线性映到 \(d=128\)，加位置编码，过 3 层通道独立 Transformer 编码器，展平后线性映到 28 步。

记 \(P\) 为 patch 嵌入序列：

\[
\hat{\tilde z}_{1:28}=\mathrm{Head}\bigl(\mathrm{Encoder}(P)\bigr).
\]

### 10.4 训练循环（与 Informer 共用 `fit_with_tail_validation`）

1. `seed_everything(seed)`：Python / NumPy / CUDA，确定性算法 `warn_only`。
2. DataLoader 打乱种子 = 运行种子（登记时改过，不再固定为 0）。
3. 每轮训练集 MSE；验证集不更新梯度。
4. 验证损失下降超过 \(10^{-8}\) 则存盘并清零耐心，否则耐心 +1，到 10 停。
5. 最多 30 轮。加载最佳状态。

### 10.5 预测公式

推理时适配器**只做全局逆变换**（不再手动逆 RevIN，由上游模块完成）：

\[
\hat y=\mathrm{expm1}\bigl(\hat{\tilde z}\cdot\sigma+\mu\bigr),\qquad \hat y\leftarrow\max(0,\hat y).
\]

### 10.6 B 组 8 候选（每种子 324 任务，共 972 路径，失败 0）

| 候选 | \(C\) | \(d\) / 层 | lr | 耐心 | B MASE | 种子标准差 | 拟合秒 |
|---|---:|---|---:|---:|---:|---:|---:|
| PT01 | 365 | 128 / 3 | 1e-4 | 5 | 1.917 | 0.079 | 9.6 |
| PT02 | 180 | 128 / 3 | 1e-4 | 5 | 1.367 | 0.011 | 10.1 |
| PT03 | 365 | 128 / 3 | 5e-4 | 5 | 2.303 | 0.252 | 8.6 |
| PT04 | 365 | 64 / 2 | 1e-4 | 5 | 1.714 | 0.024 | 9.2 |
| PT05 | 365 | 128 / 3 | 1e-4 | 10 | 1.907 | 0.078 | 15.8 |
| **PT06** | **180** | **128 / 3** | **1e-4** | **10** | **1.364** | **0.020** | **16.3** |
| PT07 | 365 | 128 / 3 | 5e-4 | 10 | 2.293 | 0.242 | 14.9 |
| PT08 | 365 | 64 / 2 | 1e-4 | 10 | 1.718 | 0.030 | 13.7 |

短上下文 180 明显优于 365；学习率 5e-4 最差。PT06 只比 PT02 好 0.002，并列规则仍选 PT06（分更低）。

### 10.7 冻结后结果

外层 1.590（种子标准差 0.004）；365 天 1.302；180 天无法训练；A 组 0.276。隔离计时约 \$3.44 / 1000 任务（含每任务训练）。B 组调参约 3.3 卡时，不摊进单次成本。

---

## 11. Informer（INF07）

上游：`/opt/upstream` 的 `models.model.Informer`。同样每任务从零训练，必须 CUDA。训练循环与 PatchTST 相同，差别在归一化、编解码器和时间标记。

### 11.1 两种归一化（候选参数）

**global**（INF01/INF05）：整段 \(z=\log1p(y)\) 用一个 \(\mu,\sigma\)，窗和目标都 \((z-\mu)/\sigma\)。这是上游 Informer2020 训练集 StandardScaler 的写法。

**instance**（其余 6 个）：每个编码器窗减去**该窗自己的均值** \(c\)，再除以**任务全局** \(\sigma\)。不用窗内标准差，因为疫情平坦窗的窗标准差接近 0。目标用同一个 \(c\)。推理时把**最后上下文窗**的均值加回。

INF07 用 instance。

### 11.2 冻结架构（INF07）

| 项 | 值 |
|---|---|
| 归一化 | instance |
| 上下文 / label | 180 / 28 |
| \(d_{\mathrm{model}}\) / 头 / \(d_{\mathrm{ff}}\) | 128 / 4 / 256 |
| 编码器 / 解码器层 | 2 / 1 |
| ProbSparse `factor` | 5 |
| dropout | 0.1 |
| 注意力 / 激活 | `attn=prob`，GELU |
| 嵌入 / 频率 | `embed=fixed`，`freq=d` |
| distil / mix | True / True |
| 时间标记 | 全零 \(\mathbb{R}^{L\times 4}\)，单变量实验 |
| 批大小 / 轮数 / 耐心 | 16 / 30 / 10 |
| 优化器 | AdamW，lr \(10^{-4}\) |

解码器输入是编码器最后 28 步拼 28 个零（标准 Informer label+pred 拼接）：

\[
D=\bigl[X_{C-\ell_L+1:C};\ 0_{28\times 1}\bigr]\in\mathbb{R}^{56\times 1}.
\]

ProbSparse 注意力只计算 \(O(\ln L)\) 个查询。蒸馏层在编码器层间减半长度。输出 28 步单通道。

### 11.3 预测公式（instance）

\[
c=\mathrm{mean}(z_{T-C+1:T}),\quad
\tilde x=(z_{T-C+1:T}-c)/\sigma,\quad
\hat y=\mathrm{expm1}\bigl(\hat{\tilde z}\cdot\sigma+c\bigr),\ \hat y\leftarrow\max(0,\hat y).
\]

### 11.4 B 组 8 候选（972 路径，失败 0）

| 候选 | 归一化 | \(C\) | lr | 耐心 | B MASE | 种子标准差 | 拟合秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| INF01 | global | 365 | 1e-4 | 5 | 2.737 | 0.096 | 103 |
| INF02 | instance | 365 | 1e-4 | 5 | 2.532 | 0.134 | 82 |
| INF03 | instance | 180 | 1e-4 | 5 | 2.037 | 0.075 | 91 |
| INF04 | instance | 365 | 5e-4 | 5 | 2.324 | 0.063 | 71 |
| INF05 | global | 365 | 1e-4 | 10 | 2.697 | 0.086 | 136 |
| INF06 | instance | 365 | 1e-4 | 10 | 2.521 | 0.105 | 105 |
| **INF07** | **instance** | **180** | **1e-4** | **10** | **2.031** | **0.084** | **122** |
| INF08 | instance | 365 | 5e-4 | 10 | 2.314 | 0.063 | 105 |

全局标准化是 8 个里最差的。短上下文 + instance 最好，但仍远差于 ETS 的 1.047。外层不用 INF01。

### 11.5 冻结后结果

外层 1.672（种子标准差 0.046）；365 天 1.417；180 天无法训练；A 组 0.273。隔离计时约 \$14.78 / 1000 任务。B 组调参约 18.4 卡时。香港 MASE 4.964，是主表最差的香港分数。

---

## 12. 轻量模型共享约定

方案 `light_model_plan.yaml` 在读取敏感性分数之前登记（提交 `17583a4`，2026-10-06）。容差 0.05 事先写死。不跑香港目的地。B 组选参规则与主路径相同。每个家族冻一个候选；B 分更低的家族当轻量冠军。外层、A 组、消融不参与选择。

### 12.1 法定日历（仅全局 / 逐市场 LightGBM）

`shared/data/holiday_calendar.parquet`，SHA256 `9060f62e542a884cf6f56d2feda8e527bb0b37eaebdedfbb5441a4357cf6cadc`，3841 行，2016-12-01 至 2026-12-31。`python-holidays 0.106`。内地名称含「调休 / 补假 / 休息日 / 延长」的行已删。不含学校假期和演出。特征对**目标日** \(\tau=t+h\) 计算，起点日已知：

| 列 | 含义 |
|---|---|
| `dow` | 星期 0–6 |
| `month` | 月 1–12 |
| `macau_holiday` | 澳门法定假日 |
| `mainland_holiday` | 内地法定假日 |
| `source_holiday` | 该客源市场法定假日 |
| `days_to_mainland_holiday` | 到最近内地假日的有符号天数，截断到 \([-7,7]\) |
| `mainland_block_length` | 该日所在内地连休日长度，否则 0 |

NLinear / DLinear **不用**日历。

---

## 13. 全局 LightGBM（GL03）与两个消融

代码：`global_lightgbm`。一个起点、27 个市场、一个模型、一个 \(h\) 特征，覆盖 \(h=1,\ldots,28\)。

### 13.1 目标（残差，不是绝对 \(\log1p\)）

锚点 \(t\) 的 28 日水平：

\[
L_t=\log\bigl(1+\mathrm{mean}(y_{t-27:t})\bigr).
\]

训练目标：

\[
r_{t,h}=z_{t+h}-L_t=\log(1+y_{t+h})-L_t.
\]

逆变换：

\[
\hat y_{o+h}=\mathrm{expm1}\bigl(\hat r_h+L_o\bigr),\qquad \hat y\leftarrow\max(0,\hat y).
\]

### 13.2 数值特征（相对 \(L_t\)）

- 28 个水平残差滞后：\(z_{t-k}-L_t\)，\(k=0,\ldots,27\)
- 窗 \(w\in\{7,14,28,56,365\}\) 的 \(\mathrm{mean}(z)-L_t\) 与 \(\mathrm{std}(z)\)（不足则从序列头截）
- 水平比：\(L_t-\log(1+\mathrm{mean}(y_{t-364:t}))\)
- 4 个同星期残差：对每个 \(h\)，取 \(t+h-7k\le t\) 的最近 4 个 \(k\)，\(z_{\cdot}-L_t\)；越界为 NaN
- 标量 \(h\)
- 类别 `market_id` ∈ {0,…,26}
- 加上 7 列日历（GL-noCal 去掉）

行数上限 1 500 000。超过则最近 365 个锚点全留，更早的均匀抽。最短锚点下标 55。

### 13.3 训练

`LGBMRegressor`：`objective=regression`，`min_child_samples=50`，`subsample=0.8`，`subsample_freq=1`，`colsample_bytree=0.8`，`n_jobs=4`，**无早停**，树数是候选参数。`categorical_feature=["market_id"]`。种子 42。

合训时 27 个市场的锚点行堆在一起。推理时每个市场用自己的最后一天构造 28 行（\(h=1..28\)），同一模型预测。

### 13.4 B 组搜索（324/324，失败 0）

| 候选 | 叶数 | 学习率 | 树数 | 锚点窗 | B MASE | 拟合秒 | 入选 |
|---|---:|---:|---:|---|---:|---:|---|
| GL01 | 31 | 0.05 | 500 | all | 1.133 | 1.66 | 否 |
| GL02 | 31 | 0.05 | 500 | 730 | 1.222 | 0.76 | 否 |
| **GL03** | **63** | **0.03** | **1000** | **all** | **1.115** | **2.96** | **是** |

冻结 2026-10-07。家族内 GL03 胜出，但轻量冠军是 NL01（见下节）。

### 13.5 消融（GL03 参数，不选参）

| 设置 | 做法 | B | 外层 |
|---|---|---:|---:|
| GL03 合训+日历 | 冻结主路径 | 1.115 | 1.338 |
| GL-noCal | 去掉 7 列日历 | 1.185 | 1.435 |
| GL-local | 每市场单独拟合，日历保留 | 1.187 | 1.260 |

外层上去掉日历差约 0.10；改成逐市场从 1.338 降到 1.260。H2（合训解释增益）不成立。A 组合训 GL03 是 0.236；去掉日历是 0.221，逐市场是 0.235。

### 13.6 外层与对照

GL03 外层 1.338，比 Chronos-2 高 0.186，比 SVR 1.446 好，比 NL01 1.627 好。并行运行大约 \$0.14 / 1000 任务（未做隔离计时；按墙钟×4 线程/27 市场分摊）。

---

## 14. NLinear / DLinear（NL01 冠军）

参考 Zeng 等，AAAI 2023。每起点一个共享线性图，27 个市场的窗堆在一起（通道独立、权重共享）。CPU，`threads=2`。不用日历。

### 14.1 输入

\(z=\log1p(y)\)。回看 \(L\in\{56,180\}\)。对每个市场，窗终点 \(e=L-1,\ldots,T-1-28\)：

\[
X=z_{e-L+1:e}\in\mathbb{R}^{L},\qquad Y=z_{e+1:e+28}\in\mathbb{R}^{28}.
\]

全部市场的窗按终点时间稳定排序后，**最后 10%** 做验证（与 PatchTST 同一条尾部规则）。批大小 256，Adam，lr \(10^{-3}\)，MSE，最多 50 轮，耐心 5。种子 11/29/42。

### 14.2 NLinear（NL01 / NL02）

只减最后一天再加回：

\[
\hat z_{1:28}=W\bigl(z_{T-L+1:T}-z_T\mathbf{1}\bigr)+z_T\mathbf{1},\qquad W\in\mathbb{R}^{28\times L}.
\]

实现：`nn.Linear(L, 28)`，偏置默认开启。参数 \(28L+28\)。NL01 的 \(L=56\)，共 1 596 个参数。

### 14.3 DLinear（NL03）

移动平均核 25，两端用首/末值各垫 12 天，再 unfold 取均值得趋势，季节 = 原序列 − 趋势：

\[
\tau=\mathrm{MA}_{25}(z),\quad
\hat z=W_\tau\tau+W_s(z-\tau).
\]

两个 `Linear(L, 28)`。只作为 NL03 进 B 组。

### 14.4 预测公式

\[
\hat y=\mathrm{expm1}(\hat z),\qquad \hat y\leftarrow\max(0,\hat y).
\]

没有 PatchTST 那种全局均值方差缩放；水平完全靠「减最后一点 / 加最后一点」或移动平均。

### 14.5 B 组（每候选 972 路径）

| 候选 | 结构 | \(L\) | B MASE | 种子标准差 | 拟合秒 | 入选 |
|---|---|---:|---:|---:|---:|---|
| **NL01** | **NLinear** | **56** | **1.073** | **0.0013** | **0.29** | **是，轻量冠军** |
| NL02 | NLinear | 180 | 1.101 | 0.0075 | 0.28 | 否 |
| NL03 | DLinear | 180 | 1.171 | 0.0177 | 0.40 | 否 |

B 上 NL01 仅次于 ETS 1.047，优于全部表格基线和 GL03。短回看再次胜过长回看。

### 14.6 冻结后结果

外层 1.627，与 ETS 1.639 持平（差 −0.012），比 Chronos-2 高 **0.475**，远超事先 0.05 容差。A 组 0.209。并行运行大约 \$0.006 / 1000 任务。B 组恢复期选参把冠军给了 NL01，到外层回到 ETS 附近——方案里已写明这是恢复期选参的代价。

H1（轻量冠军相对 Chronos-2 的 95% 区间上界 ≤ 0.05）不成立；点估计已经超出 0.475。

---

## 15. 第 2 轮：逐市场 LightGBM（LL01–LL03）

`light_model_round2.yaml` 状态是 `registered_after_outer_results`（2026-10-09）。第 1 轮 0.05 容差不变。事后把对照条放宽到 0.1 之后，只改消融指出的那一块：逐市场训练、保留日历、特征与残差目标与 GL 相同。**不替换第 1 轮冻结，外层只作探索。** 三个候选用完即停，不再因本轮外层分数加候选。

| 候选 | 叶数 | 学习率 | 树数 | 锚点窗 | B MASE | 外层 MASE | 相对 Chronos-2 |
|---|---:|---:|---:|---|---:|---:|---:|
| **LL01** | 63 | 0.03 | 1000 | all | **1.187** | **1.260** | +0.108 |
| LL02 | 31 | 0.05 | 500 | all | 1.194 | 1.267 | +0.115 |
| LL03 | 63 | 0.03 | 1000 | 730 | 1.499 | 1.821 | +0.668 |

B 组选 LL01。公式、特征、日历与第 13 节完全相同，只是 `pooled=False`：每个市场单独 `fit`。LL01 的 B/外层分数与 GL-local 消融一致（1.187 / 1.260），这是预期——参数与 GL03 相同。0.1 容差仍未越过：+0.108。锚点窗收到 730 天（LL03）明显变差。

---

## 16. 需训练 / 需选参模型的结果总表

零样本五模型不列入。数字是澳门等权 MASE(7)。外层主设置为 `history=all`。

| 模型 | 候选 | B | 外层 | 180 | 365 | A |
|---|---|---:|---:|---:|---:|---:|
| ETS | ETS03 | 1.047 | 1.639 | 1.116 | 1.140 | 0.197 |
| NLinear | NL01 | 1.073 | 1.627 | — | — | 0.209 |
| 岭回归 | RID02 | 1.093 | 1.624 | 1.650 | 1.192 | 0.226 |
| Elastic Net | EN03 | 1.102 | 1.616 | 2.039 | 1.187 | 0.227 |
| Lasso | LAS03 | 1.106 | 1.614 | 1.696 | 1.148 | 0.228 |
| 全局 LightGBM | GL03 | 1.115 | 1.338 | — | — | 0.236 |
| SVR | SVR02 | 1.154 | **1.446** | 1.133 | 1.183 | 0.288 |
| SARIMA | SAR03 | 1.201 | 1.751 | 1.229 | 1.203 | 0.196 |
| 季节性朴素 | SN01 | 1.206 | 1.883 | 1.262 | 1.323 | 0.206 |
| 逐市场 LightGBM（基线） | LGB01 | 1.254 | 1.515 | 1.183 | 1.155 | 0.238 |
| 随机森林 | RF03 | 1.259 | 1.515 | 1.203 | 1.156 | 0.235 |
| 随机森林 800 | PRF01 | 1.259 | 1.516 | 1.202 | 1.155 | 0.235 |
| 梯度提升 | GB01 | 1.263 | 1.513 | 1.228 | 1.175 | 0.237 |
| 逐市场 LGB + 日历（事后） | LL01 | 1.187 | 1.260 | — | — | — |
| GL-local 消融 | GL03 | 1.187 | 1.260 | — | — | 0.235 |
| GL-noCal 消融 | GL03 | 1.185 | 1.435 | — | — | 0.221 |
| Transformer | TR03 | 1.288 | 1.591 | 1.187 | 1.159 | 0.246 |
| 普通最小二乘 | LIN01 | 1.312 | 1.626 | 3.359 | 1.234 | 0.226 |
| PatchTST | PT06 | 1.364 | 1.590 | 无法训练 | 1.302 | 0.276 |
| Informer | INF07 | 2.031 | 1.672 | 无法训练 | 1.417 | 0.273 |

外层上，需训练模型的顺序是：LL01/GL-local（探索）→ GL03 → SVR → 四类树 → PatchTST / 小型 Transformer → 线性族 → NL01 / ETS → Informer → SARIMA → 朴素。B 组顺序几乎反过来：ETS / NL01 最好，Informer 最差。这是恢复期选参与外层分布不同的直接证据。

---

## 17. 预测器写出的字段与路径

### 17.1 基线（`run_backtest` / `run_b_validation`）

每个任务一个 parquet 分片，含 `task_id, fold_id, market, origin_date, history_length, horizon_day, model, candidate_id, seed, prediction_raw, prediction, was_clipped, mase7_scale, rmsse7_scale, high_demand_threshold` 以及拟合/推理秒数。B 组目录形态：`predictions/<model>/<candidate>/<seed>/<task>.parquet`。

### 17.2 PatchTST / Informer

同上，另存 `checkpoint_dir/<model>/<task_id>.pt`：`state_dict`、归一化统计、`epochs_run`、`best_validation_mse`。正式外层在 `track1/results/foundation_outer_frozen/`。

### 17.3 轻量模型

每个起点一个目录：

`<output-root>/<split>/<model>/<candidate>/seed<seed>/<fold>_<origin>/`

内含 `predictions.parquet`、`timings.jsonl`、`run_manifest.json`（计划 SHA、代码 SHA、日历 SHA、`target_artifact_accessed=false`）。默认根目录 `track1/results/light_models/`。第 2 轮同一脚本，`--plan light_model_round2.yaml`。

评价器只读这些预测和受保护目标，按第 1.2 节宏平均。冻结文件是选参的唯一下游接口；外层脚本不得扫描 B 组以外的候选。

---

## 18. 实现对照（公式在代码里的位置）

| 内容 | 位置 |
|---|---|
| 朴素 / ETS / SARIMA / 表格 / Transformer | `track1/code/models.py` |
| 表格 37 维特征 | `_features` 与 `prepare_tabular_training_sets` |
| MASE 尺度 | `run_backtest._scales`：\(\mathrm{mean}\lvert y_t-y_{t-7}\rvert\) |
| 窗与尾部验证 | `foundation_models/adapters/_trainable.py` |
| PatchTST 配置与逆变换 | `adapters/patchtst.py` |
| Informer instance 中心化 | `adapters/informer.py` 的 `encode_context` |
| 日历 7 列 | `light_models.Calendar.features` |
| GL 残差特征 | `light_models._gl_features` |
| NLinear / DLinear | `light_models._linear_module` |
| 基线搜索 | `track1/configs/baseline_search_spaces.yaml`（33 个候选） |
| PT/INF 搜索 | `track1/configs/foundation_search_spaces.yaml`（各 8 个） |
| 轻量第 1 / 2 轮 | `light_model_plan.yaml` / `light_model_round2.yaml` |

文档日期 2026-10-11。上表的零样本模型仍是只看本市场历史的外层分数。2026-10-11 另有一轮输入对齐：Chronos-2 1.129，TimesFM 3.0 1.077，Moirai 2.0 1.562。Granite 与 Lag-Llama 的检查点不能接收日历和其他市场历史。LL01 仍比对齐后的 Chronos-2 高 0.131。说明见 `论文主文_三条结论与文献索引.md` 第 4 节和 `零样本对齐轻量输入后的外层结果.md`。




