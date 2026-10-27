# 需求预测与库存决策：门店级 LSTM + M5 概率预测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![CI](https://github.com/lingtao-meng/lstm-attention-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/lingtao-meng/lstm-attention-demand-forecasting/actions/workflows/ci.yml)


这个仓库有两块内容：

1. **门店级预测**（根目录）：用 Walmart 数据训练 LSTM + Self-Attention，做未来 4 周销售预测，并接到安全库存 / newsvendor。
2. **M5 item-store 预测**（`m5/`）：100 个 item-store series 的周度预测，补了分位数、conformal、滚动回测和分层 reconciliation。

两块都是“预测 → 决策”，不是只比模型指标。根目录的门店级模型还带一个 Streamlit 应用。

## 项目结构

| 模块 | 内容 | 入口 |
|---|---|---|
| 门店级预测 | Walmart 45 店、LSTM+Attention、P10/P50/P90、FVA、newsvendor | `scripts/`、`app/` |
| M5 预测 | 100 个 item-store series、LightGBM 残差、conformal、六折滚动、分层 | [`m5/README.md`](m5/README.md) |


在线应用：[点击试用](https://lstm-attention-demand-forecasting-94s4zjmdvbzoar6sqh5zqx.streamlit.app)

## 数据

Walmart Recruiting - Store Sales Forecasting，Kaggle 公开数据集。45 家门店，143 周的周销售数据。特征包括周销售额、节假日标记、温度、油价、CPI、失业率。

## 模型结构

输入层（过去12周×6个特征）→ 双层双向 LSTM（128维）→ Multi-Head Self-Attention（4头）→ 残差连接 + LayerNorm → 全连接层 → 输出（未来4周×3个分位数 P10/P50/P90）。

训练了 40 个 epoch，早停防止过拟合。总共 83 万参数，在 Mac 上跑的。

## 效果对比

在同一个测试集上对比了六个模型，全部用原始美元空间评估：

| 模型 | MAE | 相对 Naive |
|------|------|:--:|
| Naive（用上周值预测本周） | $56,122 | 基线 |
| ARIMA(2,0,2) | $52,143 | +7.5% |
| XGBoost（时序） | $44,926 | +20.0% |
| LightGBM（时序） | $41,855 | +25.4% |
| Prophet（Meta） | $95,830 | -70.0% |
| **LSTM + Attention（旧口径）** | — | 待重跑；真实四步重跑见下节 **+7.7%** |

树模型（XGBoost +20%, LightGBM +25%）比传统统计方法（ARIMA +7.5%）好不少，但还是打不过 LSTM。Prophet 仍然最差。

LSTM 的优势在于它能学到每家门店的独立时序模式——树模型把 12 周×6 特征拍平成 72 维向量，丢失了时序结构信息。

## 三个额外实验

**消融实验（真实重跑 2026-10-27）：** 同一时间切分、同一随机种子、各 25 epoch，
纯 LSTM 四步 MAE 均值 **0.1059**，LSTM+Attention **0.1062**——纯 LSTM 略好，
但只差 0.3%，**不足以支撑「注意力没用」的强结论**。

> 早期版本写的是「纯 LSTM +44.7% vs Attention +39.1%」，那组数字没有任何训练支撑，
> 对应 notebook 单元只打印了硬编码字符串，已删除。真实量级是 7%–8%，不是 40%+。

**回顾窗口实验：** ⚠️ **未重跑**。原稿「8 周 +42.0%、12 周 +35.2%、16 周 +34.0%」引用了未定义变量 `imp`，
单元独立执行会 `NameError`，数字无来源。请勿引用；如需保留，须用与上面同一套脚本按窗口长度重跑。

**滚动交叉验证（真实重跑 5 折）：** 结果**不稳定，且多数折输给 naive**：

| 折 | 模型 MAE | naive MAE | 相对 naive |
|---|---|---|---|
| 1 | 0.3599 | 0.3487 | −3.2% |
| 2 | 0.3073 | 0.1029 | **−198.8%** |
| 3 | 0.1999 | 0.1974 | −1.3% |
| 4 | 0.1357 | 0.2335 | +41.9% |
| 5 | 0.0944 | 0.1050 | +10.1% |

均值 **−30.2% ± 85.8%**。早期折训练样本太少、模型严重过拟合，第 2 折比"上周销量"差 3 倍。

> 原稿「32%–50%，均值 38.2% ± 6.4%」是硬编码列表 `cv=[49.5,...]` 算出来的。
> 诚实结论：**单次时间切分上比 naive 好约 8%，但滚动回测里不稳定，不能宣称"可复现的稳定优势"。**

## 预测精度值多少钱：从误差到库存

> ⚠️ **本节是教学情景，不是本模型的实测收益。** 下面的 37.9% 是脚本的**输入假设**（先假定误差按 ×0.621 缩小，再套库存公式，所以库存自然同比例下降）；2026-10-27 的真实重跑中，模型相对 naive 的改善约 **8%**（见上），且滚动回测不稳定。因此本节的金额只能说明「误差若下降 37.9% 意味着什么」，**不能当作模型带来的降本**。

MAE 降 37.9% 是个假设值（不是模型实测），然后呢？计划岗真正关心的是这一步——**误差下降能换来多少安全库存和持有成本**。

`scripts/forecast_to_inventory.py` 用 Walmart 数据做了这个换算：

1. 先用 naive 基线（上周销量当本周预测）算出**真实误差分布**——不是拍脑袋的假设；
2. 假设新模型误差按同比例缩小（σ_model = 0.621 × σ_naive）；
3. 按 `SS = z · σ_error · √L`（L = 4 周提前期）反推安全库存与年持有成本。

结果（45 家门店，95% 服务水平，持有成本率 25%/年）：

| 指标 | naive 基线 | LSTM + Attention | 变化 |
|---|---|---|---|
| 周度绝对误差 MAE 合计 | $3,822,095 | $2,373,521 | −37.9% |
| 安全库存（45 店合计） | $22,599,992 | $14,034,595 | **−37.9%** |
| 年持有成本 | $5,649,998 | $3,508,649 | **省 $2.14M** |

服务水平取 90% / 95% / 99% 时，年节省分别是 $1.67M / $2.14M / $3.03M。

> **口径说明**：这是情景推算，不是企业实测，也不是本模型的真实收益。它只回答「如果 MAE 降 37.9%，库存会怎样」；真实重跑相对 naive 约 +7.7%，滚动回测不稳定。安全库存对误差是线性的，但 MAE 降幅不能直接推出标准差和尾部分位数同比例下降。

复现：`python scripts/forecast_to_inventory.py --improvement 0.379`

## 预测值不值：FVA 与 newsvendor

MAE 降 37.9% 是对 naive 讲的。但 naive 不是最强基准：在同一 holdout 窗口上，
4 周移动平均（MA4）和 seasonal naive 都可能更强。所以我先做了两件事。

### 1. FVA：基准选错，结论会虚高

![FVA](images/fva_tracking.png)

固定最后 26 周（45 店 × 26 周），只用可复现的美元空间基准：

| 模型 | MAE | FVA vs naive | 偏差 | 中位跟踪信号 | \|TS\|>4 |
|---|---|---|---|---|---|
| **MA4** | **47,554** | **+6.5%** | +0.01% | 0.31 | 7/45 |
| naive | 50,865 | 0 | +0.15% | 0.63 | 0/45 |
| seasonal naive | 54,031 | −6.2% | +2.03% | 18.07 | 43/45 |

- **MA4 才是最强简单基准**；seasonal naive 在这 26 周反而比 naive 差 6.2%，而且 43/45 家店
  \|TS\| > 4，说明它是系统性有偏的（全窗口上它更好，是窗口效应，不能直接搬到部署）。
- 旧稿里的 37.9% 没有真实训练支撑，已废弃；真实四步重跑相对 naive 约 **+7.7%**，如果拿它直接当「对最强基准的 FVA」，
  会高估。README 里近似的 $34,852 若只作量级参考，对 MA4 的 FVA 约 26.7%——但该美元值不可直接比，
  正式引用请用真实重跑数字，不要再引用 37.9%。
- 所以 FVA 的分母该用 MA4，不是 naive。另外，报告 FVA 的时候得把评估窗口和基准一起写出来，不然数字没法比。

### 2. Newsvendor：误差分布形状也值钱

![newsvendor](images/newsvendor.png)

给定缺货成本 Cu = 0.30、持有成本 Co = 0.25，临界比 CR = Cu/(Cu+Co) = 0.545；
最优订货量 = 预测 + 误差分布的 CR 分位数。

| 方法 | 期望成本（$/店-周） | 相对经验分位 |
|---|---|---|
| 经验分位数（MA4 误差） | **$12,666** | — |
| 正态近似 | $12,975 | +2.4% |
| 误差缩减 37.9%（假设情景） | **$7,866** | **−37.9%** |

- 误差分布厚尾/偏斜，**经验分位数比正态近似更稳**（正态多花 2.4%）。
- 服务水平从 80% 提到 95%，成本从 $15.9k 涨到 $23.0k；0.8 之后边际成本陡增。
- 预测误差下降会**线性**传导到 newsvendor 成本——这和安全库存的口径一致，但 newsvendor
  把缺货成本显式写出来了。

```bash
python scripts/forecast_diagnostics.py
python scripts/forecast_diagnostics.py --holdout 12 --underage 0.4
```

## M5 模块

`m5/` 是同一主题下的扩展：M5 真实 Walmart 销售，100 个 item-store series，最后 26 周 holdout。

- 点预测：LightGBM 残差，单窗口 FVA +9.8%；六折滚动 +6.1% ± 4.5%
- 概率预测：P10/P50/P90 + conformal，80% 区间覆盖 78.5%
- 订货：P50+conformal 的 newsvendor 成本 $12.96，比 naive+conformal 低 12.5%
- 分层：114 个节点，bottom-up 赢 MinT-OLS

详细结果、图和复现命令见 [`m5/README.md`](m5/README.md)。

## 一些踩过的坑

最初打算用 Google 的 Temporal Fusion Transformer（TFT），结果 pytorch-forecasting 和 pytorch-lightning 的版本冲突搞了很久没搞定。最后放弃了，直接用 PyTorch 从零写 LSTM+Attention。回头看这个决定反而是对的——手写一遍对底层理解深了很多。

**关于 MAE 的口径（重要）**：LSTM 那一行的美元值（调优前 $43,504、调优后 ~$35,000）是从标准化空间反推的近似值；Naive / ARIMA / Prophet 是在原始美元空间直接测的。**旧稿的 22.8% → 37.9% 没有真实训练支撑；真实四步重跑约 +7.7%，滚动回测不稳定，不要再用 37.9%。**

## 运行

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn streamlit
jupyter notebook lstm_demand_forecast.ipynb

# 预测诊断：FVA + tracking signal + newsvendor
python scripts/forecast_diagnostics.py

# M5 模块（数据约 320MB，先下载）
cd m5
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analysis.py
python scripts/run_probabilistic.py
python scripts/rolling_backtest.py
python scripts/hierarchical.py
cd ..

# Streamlit app
cd app && streamlit run app.py
```

## License

MIT
