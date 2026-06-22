# LSTM + Attention 需求预测模型

基于 **Walmart 零售数据** 的深度学习需求预测项目。使用纯 PyTorch 实现双层 LSTM + Self-Attention 架构，对 45 家门店进行多步销售预测。

## 🧠 模型架构

```
输入 (过去12周) → 双层双向LSTM → Multi-Head Self-Attention → 残差连接 + LayerNorm → 全连接 → 分位数输出 (未来4周)
                                                                                          ├─ P10 (保守)
                                                                                          ├─ P50 (中位)
                                                                                          └─ P90 (乐观)
```

**为什么选这个架构：**
- **LSTM** 捕捉销售数据中的长期依赖和季节性模式
- **Self-Attention** 自动学习不同历史时间步的重要性（哪些周对预测最关键）
- **分位数输出** P10/P50/P90 三个分位数，量化预测不确定性
- **纯 PyTorch 手写**，不依赖高层时序库，展示对底层原理的理解

## 📊 数据

- **来源：** Walmart Recruiting - Store Sales Forecasting (Kaggle)
- **规模：** 45 家门店 × 143 周 = 6,435 条记录
- **特征：** 周销售额、节假日标记、温度、油价、CPI、失业率
- **任务：** 给定过去 12 周数据，预测未来 4 周销售额

## 🚀 结果

| 模型 | MAE ↓ | RMSE ↓ | vs 基线 |
|------|-------|--------|---------|
| **LSTM + Attention** | **$43,504** | **$60,505** | **4.5%** |
| Naive（上周值） | $56,357 | $79,011 | 5.3% |
| ARIMA(2,0,2) | $52,143 | $74,871 | 4.9% |
| Prophet（Meta） | $95,830 | $145,211 | 8.8% |
| **LSTM vs Naive** | **+22.8%** | **+23.4%** | — |
| **LSTM vs ARIMA** | **+16.6%** | **+19.2%** | — |
| **LSTM vs Prophet** | **+54.6%** | **+58.3%** | — |

> 所有模型均在**原始美元空间**（非标准化）公平评估，使用相同的训练/测试切分。LSTM 是唯一在所有指标上显著优于所有基线的模型。

### 业务价值

| 指标 | 数值 |
|------|------|
| 每店每周预测误差减少 | **$12,853** |
| 单店年度误差减少 | **$668,356** |
| 45店年度误差减少 | **约 $3,008万** |
| 等效于避免的缺货/积压损失 | **约29周库存量/年** |

> 以上数字全部基于原始美元空间的实测 MAE，非估算值。

### 关键发现
1. **四模型全面对比，LSTM 唯一全胜：** Naive → ARIMA → Prophet → LSTM 四级基准测试中，LSTM 以 +22.8% 领先 Naive、+16.6% 领先 ARIMA、+54.6% 领先 Prophet。是所有指标上唯一全面优于基线的模型
2. **传统方法在此场景失效：** Prophet 比 Naive 差 70%（全局季节性假设在多门店异构数据上崩溃），ARIMA 仅小幅提升 7.5%（且需逐店调参，不可规模化）
3. **注意力机制揭示业务规律：** 模型自动关注 W-5/W-6/W-7 周的信息，与零售补货周期吻合
4. **分位数输出直接指导决策：** P10→安全库存下限 | P50→预期需求 | P90→避免过度采购上限

## 🛠 技术栈

- **PyTorch** — 深度学习框架
- **NumPy / Pandas** — 数据处理
- **Matplotlib / Seaborn** — 可视化
- **Scikit-learn** — 数据标准化

## 📁 项目结构

```
tft-demand-forecasting/
├── README.md
├── data/
│   └── Walmart.csv                    # 原始数据集
├── notebooks/
│   ├── 01_eda.ipynb                   # 数据探索（5张图）
│   └── lstm_demand_forecast.ipynb     # LSTM+Attention 模型（完整训练+评估）
├── images/
│   ├── eda_overview.png               # 销售趋势、分布、节假日效应
│   ├── eda_external_features.png      # 外部特征分析
│   ├── eda_correlation.png            # 特征相关性热力图
│   ├── lstm_forecast_results.png      # 训练曲线、预测对比、误差分布
│   ├── lstm_attention_weights.png     # 注意力权重可视化
│   └── lstm_forecast_results.png       # 预测结果对比
└── requirements.txt

> 运行 notebook 后会自动生成 `models/` 目录，包含训练好的模型权重（best_lstm_attention.pt）。
```

## 🔧 快速开始

```bash
# 1. 安装依赖
pip install torch pandas numpy matplotlib seaborn scikit-learn jupyter

# 2. 运行 Notebook
cd notebooks
jupyter notebook lstm_demand_forecast.ipynb
```

## 📝 许可

MIT
