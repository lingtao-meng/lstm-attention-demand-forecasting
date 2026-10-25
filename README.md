# M5 周度需求预测：FVA 和 newsvendor

用 M5（Walmart 真实销售数据）做一个 item-store 级的周度需求预测。这个项目不是再比一次模型，主要想回答三个问题：

1. 简单基准到底有多强？
2. LightGBM 相对最强基准的 FVA 是多少？
3. 误差降下来，订货成本能降多少？

数据、代码、图表和结果都在仓库里，可以直接复现。

## 数据

M5 来自 Kaggle 的 Walmart 销售竞赛，包含 10 家店、3,049 个商品、1,941 天的日销量，还有日历事件和每周价格。

这个仓库取了 **100 个 item-store series**：每家店销量最高的 5 个，再随机抽 50 个。这样既不会全是畅销品，也不至于被长尾噪声淹没。

- 日销量聚成周销量（周一到周日）
- 最后 **26 周**做 holdout，前面按时间切分，不随机打乱
- 价格来自 `sell_prices.csv`，按周平均后并到 series 上
- 训练时只用 holdout 之前的数据

数据文件约 320MB，没有进 git。第一次运行先下载：

```bash
python scripts/download_data.py
```

脚本用并行 range 请求从 Hugging Face 的公开镜像拉，单连接会被限速。

## 方法

**基准**

| 模型 | 做法 |
|---|---|
| naive | 上周销量 |
| seasonal naive | 去年同周 |
| MA4 | 过去 4 周均值 |

**LightGBM**

特征包括 lag 1/2/3/4/8/13/26/52 周，4/8/13/26/52 周滚动均值和标准差，13 周零销量占比，8 周 EWM，周内位置的正余弦，月份/年份，事件和 SNAP 标记，价格、相对价格和价格变化。

模型不是直接预测销量，而是**预测相对 naive 的残差**：最终预测 = naive + 残差。原因是树模型在时间外推上比较弱，直接预测在 holdout 上会系统性低预测；锚在 naive 上更稳。这个对比在下面有数。

**评估**

- MAE、RMSE、偏差、每家店的 tracking signal（Σerror / MAD，|TS| > 4 视为有偏）
- FVA = 1 − 模型 MAE / 基准 MAE
- 所有模型都在同一个 26 周 holdout 上算

**Newsvendor**

临界比 CR = Cu / (Cu + Co) = 0.30 / (0.30 + 0.25) = 0.545。每家店用自己 holdout 误差的经验分位数订货，成本和正态近似对比。

## 结果

### 单窗口：26 周 holdout

| 模型 | MAE | RMSE | 偏差 | \|TS\|>4 | FVA vs naive |
|---|---|---|---|---|---|
| **LightGBM（残差）** | **22.77** | **47.24** | −1.90 | 38/100 | **+10.2%** |
| naive | 25.36 | 59.85 | +0.08 | 2/100 | 0 |
| MA4 | 28.08 | 66.74 | −0.18 | 21/100 | −10.7% |
| seasonal naive | 37.38 | 79.37 | +5.66 | 80/100 | −47.4% |

![预测示例](images/forecast_examples.png)

两个和直觉不太一样的地方：

- **naive 很强。** 在这 100 个 series 上，MA4 和 seasonal naive 都没打赢它；seasonal naive 在最后 26 周甚至差了 47%。这不是 seasonal 方法本身不行，是这段窗口的水平和前一年对不上，硬套 52 周滞后会吃亏。
- **直接预测不如残差模型。** 同样特征下，直接预测销量 MAE 23.36、FVA +7.9%；改成预测 naive 残差后 MAE 22.77、FVA +10.2%。树模型外推趋势吃亏，这个差距基本就是它带来的。

![模型对比](images/model_comparison.png)

模型还是有偏：38/100 家店的 |TS| > 4，中位 TS −2.59。

### 滚动回测：+10.2% 是最好的一折

![滚动回测](images/rolling_backtest.png)

把同样的 100 个 series 走 6 个不重叠的 26 周测试块，每次重新训练：

- FVA 平均 **+5.5% ± 4.2%**，范围 **−1.0% 到 +10.2%**
- 6 折里 4 折显著优于 naive，1 折显著更差，1 折不显著（Diebold–Mariano，α=0.05）
- 单窗口的 +10.2% 是乐观值。更可信的说法是“平均 5% 左右，好的窗口能到 10%，差的时候会输”

FVA 本来就不是模型属性，跟窗口、品类、促销节奏都有关系。一个数字不如一个分布诚实。

### 概率预测：P10/P50/P90 + conformal

![概率预测](images/probabilistic_forecast.png)

- P10/P50/P90 都预测 naive 残差；P10/P90 在训练期最后 16 周做 split conformal
- 80% 区间实际覆盖 **81.9%**，平均宽度 64.3 件
- pinball loss：P10 **4.61** / P50 **11.10** / P90 **5.69**

![校准](images/probabilistic_calibration.png)

覆盖率略高于名义值，说明区间偏保守。对库存来说这通常比偏窄安全，但会多备一点货。

### Newsvendor：点预测 vs 概率预测

点预测版（用测试集误差分位，偏乐观）：

| 策略 | 成本（$/series-week） |
|---|---|
| naive | $13.48 |
| LightGBM | **$12.02（−10.9%）** |

概率版（用 16 周校准误差，严格样本外）：

| 策略 | 成本（$/series-week） |
|---|---|
| naive + conformal | $14.81 |
| P50 + conformal | **$13.11（−11.5%）** |
| naive + normal | $13.90 |

![概率版 newsvendor](images/probabilistic_newsvendor.png)

两个口径的绝对成本不能直接比：点预测版用了测试集误差分位，概率版用校准期误差分位，后者更接近上线时能看到的信息。但方向一致：**模型误差降下来，订货成本也降**；概率版里 P50+conformal 比正态近似低 5.7%。

## 几个判断

- FVA 的分母要用**同窗口的最强基准**，不是随手拿 naive。这个项目里 naive 恰好最强，但换个窗口，seasonal 可能翻上来。
- 单窗口 +10.2%，六折平均 +5.5%。报 FVA 最好同时给区间，不然很容易被一个窗口骗了。
- 预测误差降 10%，newsvendor 成本大致也降 10%，因为误差是线性传导的。真正决定省多少钱的，还是缺货/持有成本参数。
- 100 个 series 不是全量 M5。换成全部 30,490 个 series，长尾和零销量会更多，FVA 大概率会变。
- 价格只用了每周均价，没有促销标记；M5 的促销信息藏在价格变化里，这块还能再挖。

## 运行

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analysis.py

#  ablation：直接预测销量 vs 预测 naive 残差
python scripts/run_analysis.py --direct

# 概率预测 + conformal
python scripts/run_probabilistic.py

# 滚动回测 + DM 检验
python scripts/rolling_backtest.py
```

结果写到 `results/`，图写到 `images/`，包括 `model_metrics.csv`、`model_metrics_direct.csv`、`predictions_test.csv`、`newsvendor_summary.csv`、`diagnostics.json`。

## 文件

```
src/m5.py                 数据加载、选 series、周度聚合、特征、基准、切分
src/evaluation.py         MAE/RMSE/bias/tracking signal/FVA、newsvendor
src/probabilistic.py      pinball、区间覆盖、conformal、DM 检验
scripts/download_data.py  并行下载 M5
scripts/run_analysis.py   点预测端到端流程
scripts/run_probabilistic.py  分位数预测 + conformal + 订货
scripts/rolling_backtest.py   六折滚动回测 + DM
tests/                    合成数据单测
results/                  指标、预测、newsvendor、校准和滚动回测结果
images/                   预测示例、概率区间、校准、成本前沿、滚动回测
```

## 数据来源

M5 Forecasting – Accuracy（Kaggle）。这里通过 Hugging Face 的公开镜像下载，文件是 `calendar.csv`、`sales_train_validation.csv`、`sell_prices.csv`。数据版权归原竞赛/沃尔玛所有，本仓库只用于学习和复现。
