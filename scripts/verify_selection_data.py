"""Reconstruct all sample lists directly from original public daily sales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir, root=ROOT):
    output = root / 'm5/results/robustness/selection'
    config = json.loads((output / 'config.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for name, sha in config['data_sha256'].items():
        if fingerprint(data_dir / name) != sha:
            raise ValueError(f'Public raw data fingerprint changed: {name}')
    header = pd.read_csv(data_dir / 'sales_train_validation.csv', nrows=0)
    types = {c: 'int16' if c.startswith('d_') else 'category' for c in header.columns}
    sales = pd.read_csv(data_dir / 'sales_train_validation.csv', dtype=types)
    calendar = pd.read_csv(data_dir / 'calendar.csv', parse_dates=['date'])
    cache, checks = {}, []
    for policy in ['full_period', 'train_period']:
        for case in ['main'] + [f'fold_{i}' for i in range(1, 7)]:
            info = summary[policy]['selection'] if case == 'main' else summary[policy]['folds'][int(case.split('_')[-1]) - 1]['selection']
            key = policy, info['selection_end']
            if key not in cache:
                days = calendar.loc[(calendar['date'] <= info['selection_end']) & calendar['d'].isin(sales.columns), 'd'].tolist()
                candidates = sales.assign(history_total=sales[days].sum(axis=1))
                if policy == 'train_period':
                    candidates = candidates[candidates['history_total'] > 0]
                top = candidates.sort_values('history_total', ascending=False).groupby('store_id', observed=True).head(5)
                remaining = candidates.drop(index=top.index)
                remaining = remaining[remaining['history_total'] > 0]
                picked = pd.concat([top, remaining.sample(n=100 - len(top), random_state=42)], ignore_index=True)
                cache[key] = set(picked['id'].astype(str)), len(days)
            expected, count = cache[key]
            saved = pd.read_csv(output / policy / f'selected_{case}.csv')
            if expected != set(saved['id']):
                raise ValueError(f'Sample ranking/eligibility mismatch: {policy}/{case}')
            if policy == 'train_period':
                if not info['selection_end'] < info['calibration_start'] < info['test_start'] or count != info['selection_days']:
                    raise ValueError(f'Historical cutoff mismatch: {case}')
            checks.append({'policy': policy, 'case': case, 'selection_end': info['selection_end'],
                           'selection_days': count, 'series': len(saved), 'independent_ranking_match': True})
    return {'checks': checks, 'data_sha256': config['data_sha256'], 'data_hashes_verified': True,
            'method': 'Independent prefix sums, nonzero eligibility, per-store top-five and seeded random sampling; no select_series/selection_history call.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'm5/data')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    result = verify(args.data_dir)
    if args.write:
        (ROOT / 'm5/results/robustness/selection/raw_selection_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Original daily sales reproduce all {len(result['checks'])} saved sample lists and their declared cutoffs.")


if __name__ == '__main__':
    main()
