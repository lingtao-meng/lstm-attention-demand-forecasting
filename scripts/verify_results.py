#!/usr/bin/env python3
"""Recompute saved forecast metrics and check the published result summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from statistics import fmean, pstdev, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def close(actual: float, expected: float, label: str, atol: float = 1e-7) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=atol):
        raise ValueError(f"{label}: {actual} != {expected}")


def rows(root: Path, path: str) -> list[dict]:
    with (root / path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def data(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


def walmart_targets(root: Path, experiment: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild four-step targets directly from the public CSV, without PyTorch."""
    features = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    stores: dict[str, list] = {}
    for row in rows(root, "data/Walmart.csv"):
        stores.setdefault(row["Store"], []).append(row)
    inputs, targets, dates = [], [], []
    for store in sorted(stores, key=int):
        group = sorted(stores[store], key=lambda r: datetime.strptime(r["Date"], "%d-%m-%Y"))
        values = np.asarray([[float(r[c]) for c in features] for r in group], dtype=np.float32)
        days = np.asarray([np.datetime64(datetime.strptime(r["Date"], "%d-%m-%Y")) for r in group])
        for end in range(12, len(group) - 4 + 1):
            inputs.append(values[end - 12:end])
            targets.append(values[end:end + 4, 0])
            dates.append(days[end:end + 4])
    x, y, d = np.asarray(inputs), np.asarray(targets), np.asarray(dates)
    valid_start = np.datetime64(experiment["date_ranges"]["valid"][0])
    test_start = np.datetime64(experiment["date_ranges"]["test"][0])
    train = d[:, -1] < valid_start
    valid = (d[:, 0] >= valid_start) & (d[:, -1] < test_start)
    test = d[:, 0] >= test_start
    for name, mask in [("train", train), ("valid", valid), ("test", test)]:
        close(int(mask.sum()), experiment["split"][name], f"Walmart {name} sample count")
    mean, scale = y[train].mean(), y[train].std() + 1e-8
    truth = (y[test] - mean) / scale
    naive = (np.repeat(x[test, -1, 0][:, None], 4, axis=1) - mean) / scale
    ma4 = (np.repeat(x[test, -4:, 0].mean(axis=1)[:, None], 4, axis=1) - mean) / scale
    return truth, naive, ma4


def build(root: Path = ROOT) -> tuple[str, str, str]:
    paths = ["data/Walmart.csv", "results/lstm_split_experiment.json", "results/lstm_pred_lstm_attention.npy",
             "results/lstm_pred_lstm_pure.npy", "m5/results/predictions_test.csv", "m5/results/model_metrics.csv",
             "m5/results/diagnostics.json", "m5/results/quantile_predictions.csv", "m5/results/probabilistic.json",
             "m5/results/probabilistic_newsvendor.csv", "m5/results/rolling_backtest.csv"]
    experiment = data(root, paths[1])
    truth, naive, ma4 = walmart_targets(root, experiment)
    metrics = {"naive": float(np.abs(naive - truth).mean()), "ma4": float(np.abs(ma4 - truth).mean())}
    close(metrics["naive"], experiment["naive_mean_mae"], "Four-step naive MAE", 2e-6)
    close(metrics["ma4"], experiment["ma4"]["mean_mae"], "Four-step MA4 MAE", 2e-6)
    for name in ("lstm_attention", "lstm_pure"):
        pred = np.load(root / f"results/lstm_pred_{name}.npy", allow_pickle=False)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name}: invalid prediction shape or non-finite prediction")
        error = np.abs(pred - truth)
        metrics[name] = float(error.mean())
        close(metrics[name], experiment[name]["mean_mae"], f"{name} MAE", 2e-6)
        for i, value in enumerate(error.mean(axis=0)):
            close(float(value), experiment[name]["step_mae"][i], f"{name} step {i + 1}", 2e-6)
    point = rows(root, paths[4])
    point_metrics = {r['model']: r for r in rows(root, paths[5])}
    for model in ("lgbm", "naive", "ma4", "seasonal"):
        mae = fmean(abs(float(r[model]) - float(r["sales"])) for r in point)
        close(mae, float(point_metrics[model]["mae"]), f"M5 {model} MAE")
    diag = data(root, paths[6])
    quantile = rows(root, paths[7])
    prob = data(root, paths[8])
    n = len(quantile)
    n_series = len({r['id'] for r in quantile})
    n_weeks = len({r['week_start'] for r in quantile})
    if not n or len(point) != n or len({(r['id'], r['week_start']) for r in quantile}) != n:
        raise ValueError("M5 prediction rows must have unique, matching item-store weeks")
    close(n_series, diag['series'], "M5 series count")
    close(n_weeks, diag['holdout_weeks'], "M5 holdout weeks")
    close(n, n_series * n_weeks, "M5 balanced test panel")
    close(prob['calibration_records'], n_series * prob['calibration_weeks'], "M5 calibration sample count")
    point_map = {(r['id'], r['week_start']): float(r['sales']) for r in point}
    for r in quantile:
        close(float(r['actual']), point_map[(r['id'], r['week_start'])], "M5 point/quantile actual")
        if date.fromisoformat(r['week_start']).weekday() != 5:
            raise ValueError("M5 retail weeks must start on Saturday")
    coverage = fmean(float(r['lower80']) <= float(r['actual']) <= float(r['upper80']) for r in quantile)
    width = fmean(float(r['upper80']) - float(r['lower80']) for r in quantile)
    close(coverage, prob['coverage_80'], "M5 coverage")
    close(width, prob['interval_width_80'], "M5 interval width")
    crossings = sum(not float(r['q10']) <= float(r['q50']) <= float(r['q90']) for r in quantile)
    cost_table = {r['policy']: float(r['cost']) for r in rows(root, paths[9])}
    cu, co = prob['underage_price_ratio'], prob['overage_price_ratio']
    if cu <= 0 or co <= 0:
        raise ValueError("Cost coefficients must be positive")
    close(prob['critical_ratio'], cu / (cu + co), "Newsvendor critical ratio")
    for policy, field in [("naive + conformal", "order_base"), ("P50 + conformal", "order_model"), ("naive + normal", "order_normal")]:
        orders = [float(r[field]) for r in quantile]
        if any(not math.isfinite(q) or q < 0 for q in orders):
            raise ValueError(f"{policy}: invalid order quantity")
        cost = fmean(float(r['price']) * (cu * max(float(r['actual']) - q, 0) + co * max(q - float(r['actual']), 0)) for r, q in zip(quantile, orders))
        close(cost, cost_table[policy], f"{policy}: recomputed cost")
        close(cost, next(r['cost'] for r in prob['newsvendor'] if r['policy'] == policy), f"{policy}: JSON cost")
    close(n, prob['test_records'], "M5 test records")
    folds = rows(root, paths[10])
    fvas = []
    for row in folds:
        fva = 1 - float(row['mae_lgbm']) / float(row['mae_naive'])
        close(fva, float(row['fva_naive']), f"M5 fold {row['fold']} FVA")
        fvas.append(fva * 100)
    cv = experiment['rolling_cv']
    for row in cv:
        close((1 - row['model_mae'] / row['naive_mae']) * 100, row['improvement_pct'], f"Walmart fold {row['fold']} FVA", 2e-5)
    improvement = (1 - metrics['lstm_attention'] / metrics['naive']) * 100
    summary = (f"- 45店四步预测：{experiment['split']['test']:,} 个测试起点，Attention MAE **{metrics['lstm_attention']:.4f}**，"
               f"相对四步 naive 改善 **{improvement:.2f}%**；MA4 为 **{metrics['ma4']:.4f}**。\n"
               f"- 五折回测相对 naive 的改善均值 **{fmean(r['improvement_pct'] for r in cv):.2f}%**，"
               f"折间标准差 **{pstdev(r['improvement_pct'] for r in cv):.2f} 个百分点**。\n"
               f"- M5 概率订货：同样使用独立校准，P50成本 **${cost_table['P50 + conformal']:.2f}**，"
               f"naive成本 **${cost_table['naive + conformal']:.2f}** / 商品门店周。")
    reduction = (1 - cost_table['P50 + conformal'] / cost_table['naive + conformal']) * 100
    m5_summary = (f"- {n_series}条商品门店序列 × {n_weeks}周，共 **{n:,}** 条测试记录；按周更新已观测历史，做一步预测。\n"
                  f"- LightGBM 残差 MAE **{float(point_metrics['lgbm']['mae']):.4f}件**，相对 naive 改善 **{float(next(r['fva_vs_naive'] for r in diag['models'] if r['model']=='lgbm')) * 100:.2f}%**。\n"
                  f"- 六折改善均值 **{fmean(fvas):.2f}%**，折间样本标准差 **{stdev(fvas):.2f} 个百分点**，范围 **{min(fvas):.2f}% 至 {max(fvas):.2f}%**。\n"
                  f"- 名义80%区间：样本外经验覆盖 **{coverage:.2%}**，平均宽度 **{width:.2f}件**。\n"
                  f"- 给定缺货/积压成本，P50+校准的单周期成本 **${cost_table['P50 + conformal']:.4f}**，"
                  f"较同样校准的 naive **${cost_table['naive + conformal']:.4f}** 低 **{reduction:.2f}%**。")
    lines = ["# 需求预测结果核对", "", "由 `python scripts/verify_results.py --write` 生成。",
             "直接复算已保存预测与订货量；本命令不执行训练，也不验证旧模型文件的完整训练来源。", "",
             "## 45店四步预测", "", summary, "", "| 方法 | 标准化四步MAE |", "|---|---:|"]
    for name in ['naive', 'ma4', 'lstm_pure', 'lstm_attention']:
        lines.append(f"| {name} | {metrics[name]:.6f} |")
    lines += ["", "## M5", "", m5_summary, "", f"当前三分位输出中有 {crossings}/{n} 条不满足 P10≤P50≤P90；覆盖与宽度按实际上下界计算。",
              "区间调整和P50订货校准各自计算；区间变宽不意味着订货量自动增加。",
              "", "## 核对范围", "", "重建45店四步目标与训练期缩放，复算保存预测的MAE；检查M5点预测、零售周日期、样本键、区间覆盖和宽度、非负订货量、三个订货基准成本及各折FVA。",
              "滚动回测本命令只核对汇总表的算式；训练、调参、校准分数生成及分层实验需分别运行正式入口。",
              "标准差沿用各实验原记录：45店五折使用ddof=0，M5六折使用ddof=1；两者均描述折间波动，不是均值的置信区间。",
              "", "## 输入文件 SHA-256", ""]
    for path in paths:
        lines.append(f"- `{path}`：`{hashlib.sha256((root / path).read_bytes()).hexdigest()}`")
    return summary, m5_summary, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update README result blocks and docs/RESULTS.md")
    args = parser.parse_args()
    main_summary, m5_summary, report = build()
    outputs = {ROOT / 'docs/RESULTS.md': report}
    for path, summary in [('README.md', main_summary), ('m5/README.md', m5_summary)]:
        source = (ROOT / path).read_text(encoding='utf-8')
        before, rest = source.split(START, 1)
        _, after = rest.split(END, 1)
        outputs[ROOT / path] = before + START + '\n' + summary + '\n' + END + after
    for path, text in outputs.items():
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text, encoding='utf-8')
        elif not path.exists() or path.read_text(encoding='utf-8') != text:
            raise SystemExit('Result summary is stale. Run python scripts/verify_results.py --write')
    print('Saved predictions, order costs and reports are consistent.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
