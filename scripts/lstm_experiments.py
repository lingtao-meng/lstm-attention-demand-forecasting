#!/usr/bin/env python3
"""Optuna search and input-window sweep for the reproducible LSTM experiment.

Both experiments use the same target-date split as train_lstm_forecast.py.
They are intentionally small (few trials / short epochs) so the result files are
reproducible on a laptop; increase --epochs for a final run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lstm_forecast as base  # noqa: E402


def prepare(panel, input_weeks: int):
    X, Y, stores, tdates = base.build_windows(panel, input_weeks)
    tr, va, te = base.split_by_target_dates(tdates)
    flat = X[tr].reshape(-1, len(base.FEATURES))
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    Xs = (X - mu) / sd
    y_mu, y_sd = Y[tr].mean(), Y[tr].std() + 1e-8
    Ys = (Y - y_mu) / y_sd
    # 显式用原始“最后一周销量”构造 naive，再套同一套 y 标准化
    naive = np.repeat(((X[:, -1, 0] - y_mu) / y_sd)[:, None], base.HORIZON, axis=1)
    return Xs, Ys, naive, tr, va, te


def run_one(panel, input_weeks: int, epochs: int, seed: int = 42):
    Xs, Ys, naive_all, tr, va, te = prepare(panel, input_weeks)
    model = base.train_model(Xs[tr], Ys[tr], Xs[va], Ys[va], len(base.FEATURES), True, epochs, seed)
    import torch
    with torch.no_grad():
        pred = model(torch.from_numpy(Xs[te])).numpy()
    naive = naive_all[te]
    mae = float(np.abs(pred - Ys[te]).mean())
    nmae = float(np.abs(naive - Ys[te]).mean())
    return {"input_weeks": input_weeks, "model_mae": mae, "naive_mae": nmae,
            "improvement_pct": float((nmae - mae) / nmae * 100)}


def window_sweep(panel, epochs: int):
    rows = [run_one(panel, window, epochs) for window in (4, 8, 12, 16)]
    (ROOT / "results" / "lstm_window_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"window {row['input_weeks']:2d}: {row['improvement_pct']:+.2f}%")
    return rows


def optuna_search(panel, epochs: int, trials: int):
    import optuna

    Xs, Ys, _, tr, va, te = prepare(panel, 12)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [32, 64, 96])
        layers = trial.suggest_categorical("layers", [1, 2])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        model = base.train_model(Xs[tr], Ys[tr], Xs[va], Ys[va], len(base.FEATURES), True,
                                 epochs, 42, lr=lr, hidden=hidden, layers=layers, dropout=dropout)
        import torch
        with torch.no_grad():
            pred = model(torch.from_numpy(Xs[va])).numpy()
        return float(np.abs(pred - Ys[va]).mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    rows = [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]
    (ROOT / "results" / "lstm_optuna.json").write_text(
        json.dumps({"best_params": study.best_params, "best_value": study.best_value,
                    "trials": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best", study.best_params, "value", study.best_value)
    return study


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["window", "optuna", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    panel = base.load_panel(ROOT / "data" / "Walmart.csv")
    if args.mode in ("window", "both"):
        window_sweep(panel, args.epochs)
    if args.mode in ("optuna", "both"):
        optuna_search(panel, args.epochs, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
