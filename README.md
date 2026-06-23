# LSTM + Attention 需求预测

用 Walmart 零售数据训练了一个 LSTM 加 Self-Attention 的深度学习模型，做未来 4 周的销售预测。纯 PyTorch 手写的，没用高层 API。

另外还部署了一个 Streamlit 网页应用，可以直接选门店看预测结果。

在线应用：[点击试用](https://lstm-attention-demand-forecasting-bxj7ppl5rqqf9wbe8uoney.streamlit.app)

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

## 一些踩过的坑

最初打算用 Google 的 Temporal Fusion Transformer（TFT），结果 pytorch-forecasting 和 pytorch-lightning 的版本冲突搞了很久没搞定。最后放弃了，直接用 PyTorch 从零写 LSTM+Attention。回头看这个决定反而是对的——手写一遍对底层理解深了很多。

另外注意，LSTM 的 MAE（包括调优前后的 $43,504 和 ~$35,000）都是从标准化空间反推的估算值。Naive、ARIMA 和 Prophet 的 MAE 则是在原始美元空间直接测的。标准化空间的提升比例（22.8%→37.9%）是准确可比的，但美元数值是近似。

## 运行

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn streamlit
jupyter notebook lstm_demand_forecast.ipynb

# Streamlit app
cd app && streamlit run app.py
```

## License

MIT
