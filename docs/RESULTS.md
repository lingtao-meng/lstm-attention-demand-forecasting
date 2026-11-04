# 需求预测结果核对

由 `python scripts/verify_results.py --write` 生成。
直接复算已保存预测与订货量；本命令不执行训练，也不验证旧模型文件的完整训练来源。

## 45店四步预测

- 45店四步预测：1,440 个测试起点，Attention MAE **0.1062**，相对四步 naive 改善 **7.71%**；MA4 为 **0.1011**。
- 五折回测相对 naive 的改善均值 **8.32%**，折间标准差 **20.20 个百分点**。
- M5 概率订货：同样使用独立校准，P50成本 **$13.28**，naive成本 **$14.65** / 商品门店周。

| 方法 | 标准化四步MAE |
|---|---:|
| naive | 0.115110 |
| ma4 | 0.101142 |
| lstm_pure | 0.105856 |
| lstm_attention | 0.106235 |

## M5

- 100条商品门店序列 × 26周，共 **2,600** 条测试记录；按周更新已观测历史，做一步预测。
- LightGBM 残差 MAE **24.2834件**，相对 naive 改善 **6.40%**。
- 六折改善均值 **4.25%**，折间样本标准差 **2.85 个百分点**，范围 **-0.90% 至 7.14%**。
- 名义80%区间：样本外经验覆盖 **80.81%**，平均宽度 **66.20件**。
- 给定缺货/积压成本，P50+校准的单周期成本 **$13.2804**，较同样校准的 naive **$14.6468** 低 **9.33%**。

当前三分位输出中有 94/2600 条不满足 P10≤P50≤P90；覆盖与宽度按实际上下界计算。
区间调整和P50订货校准各自计算；区间变宽不意味着订货量自动增加。

## 核对范围

重建45店四步目标与训练期缩放，复算保存预测的MAE；检查M5点预测、零售周日期、样本键、区间覆盖和宽度、非负订货量、三个订货基准成本及各折FVA。
滚动回测本命令只核对汇总表的算式；训练、调参、校准分数生成及分层实验需分别运行正式入口。
标准差沿用各实验原记录：45店五折使用ddof=0，M5六折使用ddof=1；两者均描述折间波动，不是均值的置信区间。

## 输入文件 SHA-256

- `data/Walmart.csv`：`0846b6941217dd938ceba6ae04650ccbbd01f4877a5f88bfa4dbecda54806c8f`
- `results/lstm_split_experiment.json`：`7ab193d899b40fabf52c6c050371ec9675b1617f68763a689214690890f9436c`
- `results/lstm_pred_lstm_attention.npy`：`396d26d202808af86aa7dae1ca0b461c942be1d7c60ae80d31fa984c26a5e85c`
- `results/lstm_pred_lstm_pure.npy`：`f4b87960a4a9c802b30fe12a438c1c1765594bbce4cee8b1688d4eaa3595c1d3`
- `m5/results/predictions_test.csv`：`9bf64fc28aebfc0e81d18f529d59cb2e8d50cc4f7cb8b4fc14a196c8f27ceb4e`
- `m5/results/model_metrics.csv`：`f1fc70d4ffef4bbd4f68eac50a76979c6ca2708aebef6eac553ce6c5b5d49d11`
- `m5/results/diagnostics.json`：`18a70af38e1ad18713b71f89c06e6a38f377f1b3b443174b3685277eb34373be`
- `m5/results/quantile_predictions.csv`：`166ab4cb271e2071dc9dc150a5db438837a0bcdb76bb83266963d53c27468473`
- `m5/results/probabilistic.json`：`796cb4123278c996b8a2e54ab756f158cc87c4824d5105e2a327babff4e6de5c`
- `m5/results/probabilistic_newsvendor.csv`：`31b34b02e59c79d9cc5a5901e80817cd8809d1653ed9e9d174ce30d3bf983054`
- `m5/results/rolling_backtest.csv`：`0003f32d142e8ddc8718bda17461b0d2bd80a2f606889205434c1f9ae68e8beb`
