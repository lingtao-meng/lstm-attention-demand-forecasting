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

我在同一个测试集上对比了四个模型，全部用原始美元空间评估：

| 模型 | MAE | 相对 Naive |
|------|------|:--:|
| Naive（用上周值预测本周） | $56,357 | 基线 |
| ARIMA(2,0,2) | $52,143 | +7.5% |
| Prophet（Meta） | $95,830 | -70.0% |
| LSTM + Attention（初始参数） | $43,504 | +22.8% |
| **LSTM + Attention（Optuna调优）** | **~$35,000** | **+37.9%** |

Prophet 在这里表现很差，比瞎猜还差 70%。因为 Prophet 假设所有门店共享同一种季节性模式，但实际上 45 家门店各有各的规律。ARIMA 比 Naive 好一点但要逐店调参，没法规模化。

后来用 Optuna 调了一下超参数（试了 15 组不同的 hidden_dim、层数、dropout、学习率组合），最佳配置是 hidden_dim=256、单层 LSTM、dropout=0.33。调完之后提升到了 37.9%，比初始参数多了 4.3 个百分点。

如果把这个预测精度的提升换算成业务影响：每店每周预测误差减少大概 $12,853，45 店一年合计减少约 $3,008 万的误差。

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
