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
| **LSTM + Attention** | **0.499** | **0.682** | — |
| Naive（上周值） | 0.752 | 1.136 | 基线 |
| **提升** | **+33.6%** | **+40.0%** | — |

### 关键发现
1. **注意力机制有效：** 模型自动关注最近几周（W-5/W-6/W-7）的信息，符合零售业务的周期性
2. **预测不确定性可控：** P10-P90 分位数区间随预测步长增加而扩大，反映了远期的更大不确定性
3. **早停稳定：** 40 epochs 达到最优，验证损失未出现过拟合

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
│   └── lstm_attention_weights.png     # 注意力权重可视化
├── models/
│   └── best_lstm_attention.pt         # 训练好的模型权重
└── requirements.txt
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
