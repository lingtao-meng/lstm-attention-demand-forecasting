# LSTM + Attention 需求预测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)


用 Walmart 零售数据训练了一个 LSTM 加 Self-Attention 的深度学习模型，做未来 4 周的销售预测。纯 PyTorch 手写的，没用高层 API。

另外还部署了一个 Streamlit 网页应用，可以直接选门店看预测结果。

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
| **LSTM + Attention（Optuna调优）** | **~$34,852** | **+37.9%** |

树模型（XGBoost +20%, LightGBM +25%）比传统统计方法（ARIMA +7.5%）好不少，但还是打不过 LSTM。Prophet 仍然最差。

LSTM 的优势在于它能学到每家门店的独立时序模式——树模型把 12 周×6 特征拍平成 72 维向量，丢失了时序结构信息。

## 三个额外实验

**消融实验：** 对比纯 LSTM vs LSTM+Attention，发现纯 LSTM（+44.7%）反而比加 Attention（+39.1%）更好。不是所有花哨技术都有效——在这个数据量和任务上，简单的结构更不容易过拟合。

**回顾窗口实验：** 试了 4/8/12/16 周的输入窗口。8 周是最优的（+42.0%），12 周（+35.2%）和 16 周（+34.0%）反而下降——太长的历史信息引入了噪声。

**滚动交叉验证：** 在不同时间区间上重复实验 5 次，LSTM 相对 Naive 的提升稳定在 32%-50% 之间（均值 38.2% ± 6.4%）。说明模型的优势不是某一次随机种子运气好，而是可复现的。

## 预测精度值多少钱：从误差到库存

MAE 降了 37.9%，然后呢？计划岗真正关心的是这一步——**误差下降能换来多少安全库存和持有成本**。

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

> **口径说明**：这是情景推算，不是企业实测。它回答的是「MAE 降 37.9% 意味着什么」，而不是「某公司省了多少钱」。关键点在于安全库存对预测误差是**线性**的，所以误差降幅会等比例传导到库存——这也是预测项目的价值最容易被低估的地方。

复现：`python scripts/forecast_to_inventory.py --improvement 0.379`

## 一些踩过的坑

最初打算用 Google 的 Temporal Fusion Transformer（TFT），结果 pytorch-forecasting 和 pytorch-lightning 的版本冲突搞了很久没搞定。最后放弃了，直接用 PyTorch 从零写 LSTM+Attention。回头看这个决定反而是对的——手写一遍对底层理解深了很多。

**关于 MAE 的口径（重要）**：LSTM 那一行的美元值（调优前 $43,504、调优后 ~$35,000）是从标准化空间反推的近似值；Naive / ARIMA / Prophet 是在原始美元空间直接测的。**真正可比的只有标准化空间的提升比例（22.8% → 37.9%）**，所以对外引用时请用百分比（−37.9%），不要用美元绝对值。

## 运行

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn streamlit
jupyter notebook lstm_demand_forecast.ipynb

# Streamlit app
cd app && streamlit run app.py
```

## License

MIT
