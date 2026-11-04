"""Recompute seed metrics in float64 from frozen prediction evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def recompute(directory):
    runs = json.loads((directory / 'runs.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    with np.load(directory / 'predictions.npz', allow_pickle=False) as a:
        truth = a['truth'].astype(np.float64)
        base = float(np.abs(a['naive'].astype(np.float64) - truth).mean())
        ma4 = float(np.abs(a['ma4'].astype(np.float64) - truth).mean())
        for run in runs:
            error = np.abs(a[run['array_key']].astype(np.float64) - truth)
            run.update(mae=float(error.mean()), step_mae=error.mean(axis=0).tolist(),
                       fva_naive_pct=(1 - float(error.mean()) / base) * 100,
                       fva_ma4_pct=(1 - float(error.mean()) / ma4) * 100)
    table = pd.DataFrame([{k: v for k, v in run.items() if k != 'step_mae'} for run in runs])
    models = []
    for name, group in table.groupby('model'):
        models.append({'model': name, 'runs': len(group), 'mae_mean': float(group['mae'].mean()),
                       'mae_std_ddof1': float(group['mae'].std(ddof=1)),
                       'fva_naive_mean_pct': float(group['fva_naive_pct'].mean()),
                       'fva_naive_std_ddof1_pct': float(group['fva_naive_pct'].std(ddof=1)),
                       'wins_vs_naive': int((group['mae'] < base).sum()),
                       'wins_vs_ma4': int((group['mae'] < ma4).sum())})
    paired = table.pivot(index='seed', columns='model', values='mae')
    delta = paired['lstm_attention'] - paired['lstm_pure']
    summary.update(naive_mae=base, ma4_mae=ma4, models=models,
                   attention_minus_pure_mae_mean=float(delta.mean()),
                   attention_minus_pure_mae_std_ddof1=float(delta.std(ddof=1)),
                   attention_wins_paired_seeds=int((delta < 0).sum()))
    config['metric_accumulator'] = 'float64'
    return config, runs, table, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, default=ROOT / 'results/robustness/lstm_seeds')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    config, runs, table, summary = recompute(args.directory)
    if args.write:
        for name, value in [('config.json', config), ('runs.json', runs), ('summary.json', summary)]:
            (args.directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        table.to_csv(args.directory / 'metrics.csv', index=False)
    print(json.dumps(summary['models'], ensure_ascii=False))


if __name__ == '__main__':
    main()
