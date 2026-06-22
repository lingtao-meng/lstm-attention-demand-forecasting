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
| **LSTM + Attention** | **~$37,421** | **~$64,383** | **~3.5%** |
| Naive（上周值） | $56,357 | $79,011 | 5.3% |
| Prophet（Meta） | $95,830 | $145,211 | 8.8% |
| **LSTM vs Naive** | **+33.6%** | **+18.5%** | — |
| **LSTM vs Prophet** | **+60.9%** | **+55.7%** | — |

> \* LSTM 的美元 MAE 由标准化空间的 MAE 反推估算（基于 StandardScaler 线性变换性质：MAE_original = MAE_scaled × σ_y）。Naive 和 Prophet 的 MAE 均直接在原始美元空间测量。

### 业务价值翻译

| 指标 | 数值 | 计算方式 |
|------|------|------|
| 每店每周预测误差减少 | **~$18,936** | Naive MAE − LSTM MAE |
| 单店年度误差减少 | **~$98.5万** | × 52周 |
| 45店年度误差减少 | **~$4,430万** | × 45店 |
| 等效于避免的缺货/积压损失 | **约43周库存量/年** | 误差减少总额 ÷ 平均周销售额 |

> 以上为基于标准化的估算值。精确数值需在不做特征标准化的原始美元空间重新训练模型后确认。

### 关键发现
1. **LSTM 碾压 Prophet：** Prophet 在多门店场景中完全失效（MAE比Naive差70%），而 LSTM+Attention 实现了33.6%的正向提升——深度学习在此场景中不是「锦上添花」而是「雪中送炭」
2. **注意力机制有效：** 模型自动关注最近几周（W-5/W-6/W-7）的信息，符合零售业务的周期性
3. **预测不确定性可控：** P10-P90 分位数区间可直接用于库存安全水平的决策——P10对应保守备货下限，P90对应避免过度采购上限

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
