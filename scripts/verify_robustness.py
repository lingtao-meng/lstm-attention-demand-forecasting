"""Independently recompute seed and sample-selection robustness results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_results import walmart_targets  # noqa: E402


def near(value, expected, label, atol=2e-6):
    if not np.isfinite(value) or not np.isfinite(expected) or not np.isclose(value, expected, atol=atol, rtol=1e-7):
        raise ValueError(f"{label}: {value} != {expected}")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if any(value.dtype.kind in 'fiu' and not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f'Non-finite numeric prediction evidence: {path}')
    return result


def ids(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def ordered_quantile(values, level):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 < level < 1:
        raise ValueError('A finite nonempty calibration sample and level in (0,1) are required')
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * level)))
    return float(np.sort(values)[rank - 1])


def point_metrics(a, summary):
    expected = summary["mae"]
    actual = a["sales"]
    for name in ["naive", "ma4", "seasonal", "residual"]:
        pred = a["point"] if name == "residual" else a[name]
        if pred.shape != actual.shape:
            raise ValueError('Point prediction shape does not match targets')
        near(float(np.abs(actual - pred).mean()), expected[name], f"{name} MAE")
    if "direct" in a:
        near(float(np.abs(actual - a["direct"]).mean()), expected["direct"], "direct MAE")
    near(1 - expected["residual"] / expected["naive"], summary["fva_naive"], "point FVA")


def check(root=ROOT):
    seed_dir = root / "results/robustness/lstm_seeds"
    config, summary, runs = load(seed_dir / "config.json"), load(seed_dir / "summary.json"), load(seed_dir / "runs.json")
    if config["epochs_per_model"] != 25 or config["seeds"] != [7, 17, 42, 73, 101]:
        raise ValueError("The committed experiment must retain its prespecified five seeds and 25 epochs")
    if hashlib.sha256((root / "data/Walmart.csv").read_bytes()).hexdigest() != config["data_sha256"]:
        raise ValueError("Walmart data fingerprint changed")
    if hashlib.sha256((root / "scripts/train_lstm_forecast.py").read_bytes()).hexdigest() != config["training_source_sha256"]:
        raise ValueError("LSTM training source fingerprint changed")
    a = arrays(seed_dir / "predictions.npz")
    truth, naive, ma4 = walmart_targets(root, load(root / "results/lstm_split_experiment.json"))
    for key, value in [("truth", truth), ("naive", naive), ("ma4", ma4)]:
        np.testing.assert_allclose(a[key], value, atol=2e-6, rtol=0)
    if len(runs) != 10 or {(r['seed'], r['model']) for r in runs} != {(s, m) for s in config['seeds'] for m in ['lstm_attention', 'lstm_pure']}:
        raise ValueError("All ten prespecified runs must be present exactly once")
    if summary['completed_runs'] != len(runs):
        raise ValueError('Seed completion count differs from saved runs')
    with (seed_dir / 'metrics.csv').open(encoding='utf-8', newline='') as f:
        metric_rows = list(csv.DictReader(f))
    if len(metric_rows) != len(runs):
        raise ValueError('Seed CSV/JSON run counts differ')
    metric_map = {(int(row['seed']), row['model']): row for row in metric_rows}
    naive_mae, ma4_mae = float(np.abs(truth - naive).mean()), float(np.abs(truth - ma4).mean())
    near(naive_mae, summary['naive_mae'], 'seed naive MAE')
    near(ma4_mae, summary['ma4_mae'], 'seed MA4 MAE')
    for r in runs:
        pred = a[r['array_key']]
        if pred.shape != truth.shape or not np.isfinite(pred).all() or r['epochs'] != 25:
            raise ValueError('Invalid saved LSTM run')
        error = np.abs(pred - truth)
        near(float(error.mean()), r['mae'], 'seed run MAE')
        near(float(metric_map[r['seed'], r['model']]['mae']), r['mae'], 'seed CSV/JSON MAE')
        near((1 - float(error.mean()) / naive_mae) * 100, r['fva_naive_pct'], 'seed run FVA')
        np.testing.assert_allclose(error.mean(axis=0), r['step_mae'], atol=2e-6, rtol=0)
    for model in summary['models']:
        values = [r for r in runs if r['model'] == model['model']]
        near(np.mean([r['mae'] for r in values]), model['mae_mean'], 'seed mean')
        near(np.std([r['mae'] for r in values], ddof=1), model['mae_std_ddof1'], 'seed standard deviation')
        near(np.mean([r['fva_naive_pct'] for r in values]), model['fva_naive_mean_pct'], 'seed FVA mean')
        near(np.std([r['fva_naive_pct'] for r in values], ddof=1), model['fva_naive_std_ddof1_pct'], 'seed FVA standard deviation')
        if sum(r['mae'] < naive_mae for r in values) != model['wins_vs_naive'] or sum(r['mae'] < ma4_mae for r in values) != model['wins_vs_ma4']:
            raise ValueError('Incorrect seed win count')
    by_seed = {(r['seed'], r['model']): r['mae'] for r in runs}
    deltas = [by_seed[s, 'lstm_attention'] - by_seed[s, 'lstm_pure'] for s in config['seeds']]
    near(np.mean(deltas), summary['attention_minus_pure_mae_mean'], 'paired seed difference')
    if sum(d < 0 for d in deltas) != summary['attention_wins_paired_seeds']:
        raise ValueError('Incorrect paired-seed winner count')
    m5_dir = root / 'm5/results/robustness/selection'
    m5_config, selection = load(m5_dir / 'config.json'), load(m5_dir / 'summary.json')
    for name, sha in m5_config['source_sha256'].items():
        if hashlib.sha256((root / 'm5' / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'M5 formal training source changed: {name}')
    if (m5_config['series'], m5_config['rolling_folds'], m5_config['quantile_trees'], m5_config['point_max_trees']) != (100, 6, 700, 3000):
        raise ValueError('M5 formal sample/fold/training settings changed')
    full_ids = ids(m5_dir / 'full_period/selected_main.csv')
    for policy in ['full_period', 'train_period']:
        base, s = m5_dir / policy, selection[policy]
        if load(base / 'summary.json') != s:
            raise ValueError('Main and policy summaries differ')
        selected = ids(base / 'selected_main.csv')
        if len(selected) != 100 or len(selected & full_ids) != s['selection']['overlap_with_full_period_sample']:
            raise ValueError('Wrong selected population or overlap')
        m = arrays(base / 'main_predictions.npz')
        if len(m['sales']) != s['probability']['test_rows'] or not set(m['ids']) <= selected:
            raise ValueError('Prediction records are not from the declared sample')
        point_metrics(m, s['point'])
        p = s['probability']
        if max(m['cal_week_start']) >= min(m['week_start']) or len(set(m['cal_week_start'])) != 16:
            raise ValueError('Calibration is not a separate 16-week historical block')
        if policy == 'train_period' and not s['selection']['selection_end'] < min(m['cal_week_start']):
            raise ValueError('Sample selection used calibration/test sales')
        scores = np.maximum(m['cal_q10'] - m['cal_actual'], m['cal_actual'] - m['cal_q90'])
        adjustment = ordered_quantile(scores, 0.8)
        near(adjustment, p['conformal_adjustment'], 'interval adjustment')
        np.testing.assert_allclose(m['lower80'], m['q10'] - adjustment, atol=2e-6, rtol=0)
        np.testing.assert_allclose(m['upper80'], m['q90'] + adjustment, atol=2e-6, rtol=0)
        near(np.mean((m['sales'] >= m['lower80']) & (m['sales'] <= m['upper80'])), p['coverage_80'], 'coverage')
        near(np.mean(m['upper80'] - m['lower80']), p['width_80'], 'width')
        tau = m5_config['underage'] / (m5_config['underage'] + m5_config['overage'])
        expected = {name: np.empty(len(m['sales'])) for name in ['naive_conformal', 'p50_conformal', 'naive_normal']}
        for item in set(m['ids']):
            c, t = m['cal_ids'] == item, m['ids'] == item
            if c.sum() != 16:
                raise ValueError('Per-series calibration does not contain 16 records')
            base_resid = m['cal_actual'][c] - m['cal_naive'][c]
            model_resid = m['cal_actual'][c] - m['cal_q50'][c]
            expected['naive_conformal'][t] = np.maximum(0, m['naive'][t] + ordered_quantile(base_resid, tau))
            expected['p50_conformal'][t] = np.maximum(0, m['q50'][t] + ordered_quantile(model_resid, tau))
            expected['naive_normal'][t] = np.maximum(0, m['naive'][t] + np.mean(base_resid) + NormalDist().inv_cdf(tau) * np.std(base_resid, ddof=1))
        for name, order in expected.items():
            np.testing.assert_allclose(order, m['order_' + name], atol=2e-6, rtol=0)
            cost = np.mean(m['price'] * (0.30 * np.maximum(m['sales'] - order, 0) + 0.25 * np.maximum(order - m['sales'], 0)))
            near(cost, p['costs'][name], 'recomputed ordering cost')
        near(1 - p['costs']['p50_conformal'] / p['costs']['naive_conformal'], p['cost_reduction'], 'ordering cost reduction')
        fvas, seen_weeks = [], set()
        for fold in s['folds']:
            f = arrays(base / f"fold_{fold['fold']}_predictions.npz")
            point_metrics(f, fold)
            selected_fold = ids(base / f"selected_fold_{fold['fold']}.csv")
            if len(selected_fold) != 100 or not set(f['ids']) <= selected_fold:
                raise ValueError('Wrong rolling sample')
            if len(f['sales']) != fold['test_rows'] or fold['raw_test_rows'] != 2600 or fold['dropped_test_rows'] != 2600 - len(f['sales']):
                raise ValueError('Dropped test records were not reported correctly')
            current_weeks = set(f['week_start'])
            if current_weeks & seen_weeks:
                raise ValueError('Rolling test windows overlap')
            seen_weeks |= current_weeks
            if policy == 'train_period' and not fold['selection']['selection_end'] < fold['selection']['calibration_start'] < fold['test_start']:
                raise ValueError('Fold-specific sample selection crossed its historical cutoff')
            fvas.append(fold['fva_naive'])
        if len(fvas) != 6:
            raise ValueError('All six formal rolling folds are required')
        near(np.mean(fvas), s['rolling']['fva_mean'], 'rolling mean')
        near(np.std(fvas, ddof=1), s['rolling']['fva_std_ddof1'], 'rolling standard deviation')
    return config, summary, runs, selection


def report(root=ROOT):
    config, seed_summary, runs, selection = check(root)
    lines = ['# 多随机种子与训练期选样稳健性', '',
             '复算入口：`python scripts/verify_robustness.py --write`。实验覆盖五随机种子对照和训练期选样对照，主实验与补充实验分别报告。', '',
             '## 45店：五种子完整训练', '',
             '预先固定种子7、17、42、73、101；每种子分别训练Attention与纯LSTM，共10次，每次25个epoch。时间切分、12周输入、四步输出、默认结构和验证集选检查点规则相同。',
             '所有种子都报告，不根据测试表现挑选种子；种子重复衡量训练随机性，不等同于新的时间回测或独立数据集。', '',
             '| 种子 | Attention MAE | 纯LSTM MAE | Attention相对naive | 纯LSTM相对naive |', '|---|---:|---:|---:|---:|']
    for seed in config['seeds']:
        a = next(r for r in runs if r['seed'] == seed and r['model'] == 'lstm_attention')
        p = next(r for r in runs if r['seed'] == seed and r['model'] == 'lstm_pure')
        lines.append(f"| {seed} | {a['mae']:.6f} | {p['mae']:.6f} | {a['fva_naive_pct']:.2f}% | {p['fva_naive_pct']:.2f}% |")
    lines += ['', '| 模型 | MAE均值±种子间样本标准差 | 相对naive改善均值±样本标准差 | 优于naive/MA4的次数 |', '|---|---:|---:|---:|']
    for row in seed_summary['models']:
        lines.append(f"| {row['model']} | {row['mae_mean']:.6f} ± {row['mae_std_ddof1']:.6f} | {row['fva_naive_mean_pct']:.2f}% ± {row['fva_naive_std_ddof1_pct']:.2f}个百分点 | {row['wins_vs_naive']}/5；{row['wins_vs_ma4']}/5 |")
    lines += ['', f"naive MAE为{seed_summary['naive_mae']:.6f}，MA4为{seed_summary['ma4_mae']:.6f}。相同种子下Attention优于纯LSTM为{seed_summary['attention_wins_paired_seeds']}/5次；当前证据不支持Attention有稳定额外增益。",
              '种子间标准差描述训练波动，不是均值的置信区间。时间泛化仍需结合原五折回测，不能用本表替代。', '',
              '## M5：选样时点对照', '',
              '两套方案均使用100条序列、相同公开数据与正式配置：点模型最多3000棵树并按8周验证早停100轮；三个分位模型各700棵树；16周独立校准；26周测试；6折滚动回测。',
              '全时期方案按完整销量筛选，作为回顾性基准。训练期方案只用校准期之前的销量排名和判定非零资格，六折分别重新筛选；测试区间和训练预算不缩减。', '',
             '主窗口执行16周概率校准；六折执行点预测回测，选样时采用16周历史隔离界限，点模型仍使用原8周验证。逐折元数据中的calibration_start在此表示选样隔离段起点，不表示每折额外训练了概率模型。', '',
              '| 选样方案 | 主窗口选样截止 | 与原100条重合 | 点预测相对naive改善 | 六折均值±样本标准差 |', '|---|---|---:|---:|---:|']
    for name in ['full_period', 'train_period']:
        s = selection[name]
        lines.append(f"| {name} | {s['selection']['selection_end']} | {s['selection']['overlap_with_full_period_sample']} | {s['point']['fva_naive']:.2%} | {s['rolling']['fva_mean']:.2%} ± {s['rolling']['fva_std_ddof1'] * 100:.2f}个百分点 |")
    lines += ['', '| 选样方案 | naive MAE | 残差模型MAE | direct模型MAE |', '|---|---:|---:|---:|']
    for name in ['full_period', 'train_period']:
        metric = selection[name]['point']['mae']
        lines.append(f"| {name} | {metric['naive']:.6f} | {metric['residual']:.6f} | {metric['direct']:.6f} |")
    lines += ['', 'direct在训练期选样的主窗口中优于残差模型，原全时期样本中则相反；六折仍按预先指定的残差模型评估，不根据测试结果切换模型。']
    lines += ['', '| 选样方案 | 80%区间覆盖/宽度 | naive+校准成本 | P50+校准成本 | 同样本内成本降幅 |', '|---|---:|---:|---:|---:|']
    for name in ['full_period', 'train_period']:
        p = selection[name]['probability']
        lines.append(f"| {name} | {p['coverage_80']:.2%} / {p['width_80']:.2f}件 | ${p['costs']['naive_conformal']:.4f} | ${p['costs']['p50_conformal']:.4f} | {p['cost_reduction']:.2%} |")
    lines += ['', '订货成本单位为美元/商品门店周，缺货/积压系数为价格的0.30/0.25倍。每行策略比较使用相同测试样本和独立校准，Q统一截为非负。',
              '选样后商品组合发生变化，两个方案的绝对MAE和费用不构成同一总体的配对提升；不能把两行绝对费用之差写成模型降本。', '',
              '## 六折覆盖与缺失记录', '',
              '每折选中100条序列、26周原始测试面板为2600行。沿用原项目的缺失特征过滤，不额外缩样；下表列明实际评估数量。训练期方案每折仅按当时历史重选样本。', '',
              '| 方案 | 折 | 选样截止 | 测试开始 | 原始/评估记录 | 缺失过滤 | FVA vs naive |', '|---|---:|---|---|---:|---:|---:|']
    for name in ['full_period', 'train_period']:
        for f in selection[name]['folds']:
            lines.append(f"| {name} | {f['fold']} | {f['selection']['selection_end']} | {f['test_start']} | {f['raw_test_rows']}/{f['test_rows']} | {f['dropped_test_rows']} | {f['fva_naive']:.2%} |")
    lines += ['', '逐折缺失字段计数、实际序列数、筛选截止与训练来源在各方案的 `folds.json` 中；第一折与主窗口数据、参数和种子相同，共享一次主残差模型训练，其他折分别重训。', '',
              '## 复现与证据', '', '从仓库根目录执行：', '', '```bash',
              'python scripts/lstm_seed_robustness.py --seeds 7 17 42 73 101 --epochs 25 --threads 2 --models-dir /tmp/lstm_robustness_models',
              'python m5/scripts/selection_robustness.py --data-dir m5/data --series 100 --seed 42 --holdout 26 --folds 6 --models-dir /tmp/m5_robustness_models',
              'python scripts/verify_robustness.py --write', 'python scripts/verify_robustness.py', '```', '',
              '两条训练命令输出到独立robustness目录，不覆盖主实验。目录含配置、环境、原始数据与训练源码指纹、每次预测、校准预测、订货量和逐折样本清单；模型检查点写到指定models-dir。',
              '核对命令不训练模型：从公开CSV重建45店目标，并独立复算全部保存预测的误差、种子统计、M5区间调整、每条序列残差分位数、三套Q、费用及六折指标。', '',
              '选样函数另有未来销量扰动测试；研究结论限定于公开数据、所选样本和设定费用。', '']
    lines += ['原始数据选样核验：使用独立前缀求和、排名与抽样重建14份选样清单。复现命令为 `python scripts/verify_selection_data.py --data-dir m5/data`，记录位于 `m5/results/robustness/selection/raw_selection_verification.json`。', '']
    lines += ['### 预测证据 SHA-256', '']
    for path in sorted(list((root / 'results/robustness').rglob('*.npz')) + list((root / 'm5/results/robustness').rglob('*.npz'))):
        lines.append(f"- `{path.relative_to(root)}`：`{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true', help='Update docs/ROBUSTNESS.md after successful verification')
    args = ap.parse_args()
    text = report()
    output = ROOT / 'docs/ROBUSTNESS.md'
    if args.write:
        output.write_text(text, encoding='utf-8')
    elif not output.exists() or output.read_text(encoding='utf-8') != text:
        raise SystemExit('Robustness report is stale. Run python scripts/verify_robustness.py --write')
    print('All 10 seed runs, two selection policies, calibrated quantities and 12 rolling folds verified.')


if __name__ == '__main__':
    main()
