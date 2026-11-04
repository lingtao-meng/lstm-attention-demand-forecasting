# 多随机种子与训练期选样稳健性

复算入口：`python scripts/verify_robustness.py --write`。实验覆盖五随机种子对照和训练期选样对照，主实验与补充实验分别报告。

## 45店：五种子完整训练

预先固定种子7、17、42、73、101；每种子分别训练Attention与纯LSTM，共10次，每次25个epoch。时间切分、12周输入、四步输出、默认结构和验证集选检查点规则相同。
所有种子都报告，不根据测试表现挑选种子；种子重复衡量训练随机性，不等同于新的时间回测或独立数据集。

目标与简单基准先从公开CSV独立重建核对，指标再使用冻结评估数组并以float64累计，避免不同平台的float32汇总误差被FVA比值放大。可用 `python scripts/recompute_seed_metrics.py --write` 复算指标，无需重新训练。

| 种子 | Attention MAE | 纯LSTM MAE | Attention相对naive | 纯LSTM相对naive |
|---|---:|---:|---:|---:|
| 7 | 0.099500 | 0.100899 | 13.56% | 12.35% |
| 17 | 0.106946 | 0.097270 | 7.09% | 15.50% |
| 42 | 0.106235 | 0.105856 | 7.71% | 8.04% |
| 73 | 0.099052 | 0.098491 | 13.95% | 14.44% |
| 101 | 0.105428 | 0.099460 | 8.41% | 13.60% |

| 模型 | MAE均值±种子间样本标准差 | 相对naive改善均值±样本标准差 | 优于naive/MA4的次数 |
|---|---:|---:|---:|
| lstm_attention | 0.103432 ± 0.003835 | 10.15% ± 3.33个百分点 | 5/5；2/5 |
| lstm_pure | 0.100395 ± 0.003330 | 12.78% ± 2.89个百分点 | 5/5；4/5 |

naive MAE为0.115110，MA4为0.101142。相同种子下Attention优于纯LSTM为1/5次；当前证据不支持Attention有稳定额外增益。
种子间标准差描述训练波动，不是均值的置信区间。时间泛化仍需结合原五折回测，不能用本表替代。

## M5：选样时点对照

两套方案均使用100条序列、相同公开数据与正式配置：点模型最多3000棵树并按8周验证早停100轮；三个分位模型各700棵树；16周独立校准；26周测试；6折滚动回测。
全时期方案按完整销量筛选，作为回顾性基准。训练期方案只用校准期之前的销量排名和判定非零资格，六折分别重新筛选；测试区间和训练预算不缩减。

主窗口执行16周概率校准；六折执行点预测回测，选样时采用16周历史隔离界限，点模型仍使用原8周验证。逐折元数据中的calibration_start在此表示选样隔离段起点，不表示每折额外训练了概率模型。

| 选样方案 | 主窗口选样截止 | 与原100条重合 | 点预测相对naive改善 | 六折均值±样本标准差 |
|---|---|---:|---:|---:|
| full_period | 2016-04-24 | 100 | 6.40% | 4.25% ± 2.85个百分点 |
| train_period | 2015-07-03 | 49 | 6.30% | 3.88% ± 2.83个百分点 |

| 选样方案 | naive MAE | 残差模型MAE | direct模型MAE |
|---|---:|---:|---:|
| full_period | 25.944231 | 24.283391 | 24.875743 |
| train_period | 24.836538 | 23.272928 | 23.067270 |

direct在训练期选样的主窗口中优于残差模型，原全时期样本中则相反；六折仍按预先指定的残差模型评估，不根据测试结果切换模型。

| 选样方案 | 80%区间覆盖/宽度 | naive+校准成本 | P50+校准成本 | 同样本内成本降幅 |
|---|---:|---:|---:|---:|
| full_period | 80.81% / 66.20件 | $14.6468 | $13.2804 | 9.33% |
| train_period | 80.73% / 62.38件 | $11.9009 | $10.5263 | 11.55% |

订货成本单位为美元/商品门店周，缺货/积压系数为价格的0.30/0.25倍。每行策略比较使用相同测试样本和独立校准，Q统一截为非负。
选样后商品组合发生变化，两个方案的绝对MAE和费用不构成同一总体的配对提升；不能把两行绝对费用之差写成模型降本。

## 六折覆盖与缺失记录

每折选中100条序列、26周原始测试面板为2600行。沿用原项目的缺失特征过滤，不额外缩样；下表列明实际评估数量。训练期方案每折仅按当时历史重选样本。

| 方案 | 折 | 选样截止 | 测试开始 | 原始/评估记录 | 缺失过滤 | FVA vs naive |
|---|---:|---|---|---:|---:|---:|
| full_period | 1 | 2016-04-24 | 2015-10-24 | 2600/2600 | 0 | 6.40% |
| full_period | 2 | 2016-04-24 | 2015-04-25 | 2600/2595 | 5 | 5.02% |
| full_period | 3 | 2016-04-24 | 2014-10-25 | 2600/2537 | 63 | 3.65% |
| full_period | 4 | 2016-04-24 | 2014-04-26 | 2600/2488 | 112 | 4.21% |
| full_period | 5 | 2016-04-24 | 2013-10-26 | 2600/2311 | 289 | -0.90% |
| full_period | 6 | 2016-04-24 | 2013-04-27 | 2600/2219 | 381 | 7.14% |
| train_period | 1 | 2015-07-03 | 2015-10-24 | 2600/2600 | 0 | 6.30% |
| train_period | 2 | 2015-01-02 | 2015-04-25 | 2600/2600 | 0 | 4.20% |
| train_period | 3 | 2014-07-04 | 2014-10-25 | 2600/2600 | 0 | 1.26% |
| train_period | 4 | 2014-01-03 | 2014-04-26 | 2600/2600 | 0 | 5.83% |
| train_period | 5 | 2013-07-05 | 2013-10-26 | 2600/2600 | 0 | -0.42% |
| train_period | 6 | 2013-01-04 | 2013-04-27 | 2600/2600 | 0 | 6.08% |

逐折缺失字段计数、实际序列数、筛选截止与训练来源在各方案的 `folds.json` 中；第一折与主窗口数据、参数和种子相同，共享一次主残差模型训练，其他折分别重训。

## 复现与证据

从仓库根目录执行：

```bash
python scripts/lstm_seed_robustness.py --seeds 7 17 42 73 101 --epochs 25 --threads 2 --models-dir /tmp/lstm_robustness_models
python m5/scripts/selection_robustness.py --data-dir m5/data --series 100 --seed 42 --holdout 26 --folds 6 --models-dir /tmp/m5_robustness_models
python scripts/verify_robustness.py --write
python scripts/verify_robustness.py
```

两条训练命令输出到独立robustness目录，不覆盖主实验。目录含配置、环境、原始数据与训练源码指纹、每次预测、校准预测、订货量和逐折样本清单；模型检查点写到指定models-dir。
核对命令不训练模型：从公开CSV重建45店目标，并独立复算全部保存预测的误差、种子统计、M5区间调整、每条序列残差分位数、三套Q、费用及六折指标。

选样函数另有未来销量扰动测试；研究结论限定于公开数据、所选样本和设定费用。

原始数据选样核验：使用独立前缀求和、排名与抽样重建14份选样清单。复现命令为 `python scripts/verify_selection_data.py --data-dir m5/data`，记录位于 `m5/results/robustness/selection/raw_selection_verification.json`。

### 预测证据 SHA-256

- `m5/results/robustness/selection/full_period/fold_1_predictions.npz`：`bc9c117db2246fa470b5f6f530605b6f543b76099f2abf1a873495b2e8453a17`
- `m5/results/robustness/selection/full_period/fold_2_predictions.npz`：`52c3bab7bf3dc7d8524bd75ae6de368dc5ce39177ab9f3df92cd9de6b5cd6e3a`
- `m5/results/robustness/selection/full_period/fold_3_predictions.npz`：`4d5f9feebaffa1ae9b07d12f3cc47349adf7868c70737124f49189a5f0cffbb9`
- `m5/results/robustness/selection/full_period/fold_4_predictions.npz`：`1195e9c211daf4c1414540b2f256dd630740ea9653df47555503a3afbf33405b`
- `m5/results/robustness/selection/full_period/fold_5_predictions.npz`：`d0edb9988710c44a508ef7d39105ab1a6c89034588ae0c8c5d7a16c024d5a066`
- `m5/results/robustness/selection/full_period/fold_6_predictions.npz`：`18a3c4b7e3cfd3d269ef7b25a08ace39adbb635082782a2d06d9baa026fb9acf`
- `m5/results/robustness/selection/full_period/main_predictions.npz`：`fef8a277c81e9b8f23edd80bf5c9f5116305892bae2ec0a63c769cc7d303c362`
- `m5/results/robustness/selection/train_period/fold_1_predictions.npz`：`2b356189363b06facaf1fd77f94fd3a15bd21fdd42a7773dad7ec2b2424b0b8f`
- `m5/results/robustness/selection/train_period/fold_2_predictions.npz`：`93020003dfca9c92e0686cbfe3904de9474587e2b52dbfa6019313824d06d039`
- `m5/results/robustness/selection/train_period/fold_3_predictions.npz`：`ef0536034a6bb1cc7a582fe3c3de707f5353e394b479873f477ee32f4a8cf92e`
- `m5/results/robustness/selection/train_period/fold_4_predictions.npz`：`419cafce447d3af479b7366f1f051d79ae6b2abe39a25ce21338672c11c8ae08`
- `m5/results/robustness/selection/train_period/fold_5_predictions.npz`：`5dc3f34df4c5ccdf34e703b63d6ac15021bb5ba5a65dc44d97a6fe25407021cb`
- `m5/results/robustness/selection/train_period/fold_6_predictions.npz`：`2bea4dbf92af5a3c7cb4031c3ed89ce091a4adf0797af40222c35a0f0aa2a187`
- `m5/results/robustness/selection/train_period/main_predictions.npz`：`f1be524e12ee7e5da0faa4ce7c91f52f343d38727c5eebcc2c7bb3ad6e763372`
- `results/robustness/lstm_seeds/predictions.npz`：`17ec1626d2de8ddf979b1869afb375158d7e6cb1d2df3fc695cbacb2ec5c858f`
