"""Repeat the unchanged four-step LSTM ablation over prespecified seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_lstm_forecast import FEATURES, build_windows, load_panel, split_by_target_dates, train_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/Walmart.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42, 73, 101])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--outdir", type=Path, default=ROOT / "results/robustness/lstm_seeds")
    ap.add_argument("--models-dir", type=Path, required=True, help="Independent directory for fitted checkpoints")
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.epochs < 1:
        raise ValueError("Seeds must be unique and epochs must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    x, y, stores, target_dates = build_windows(load_panel(args.data))
    tr, va, te = split_by_target_dates(target_dates)
    flat = x[tr].reshape(-1, len(FEATURES))
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    y_mu, y_sd = y[tr].mean(), y[tr].std() + 1e-8
    xs, ys = (x - mu) / sd, (y - y_mu) / y_sd
    naive = (np.repeat(x[te, -1, 0][:, None], 4, axis=1) - y_mu) / y_sd
    ma4 = (np.repeat(x[te, -4:, 0].mean(axis=1)[:, None], 4, axis=1) - y_mu) / y_sd
    truth = ys[te]
    base = float(np.abs(naive - truth).mean())
    ma4_error = float(np.abs(ma4 - truth).mean())
    arrays = {"truth": truth, "naive": naive, "ma4": ma4,
              "store_ids": stores[te], "target_dates": target_dates[te].astype("datetime64[D]").astype(str)}
    config = {
        "seeds": args.seeds, "epochs_per_model": args.epochs, "threads": args.threads,
        "input_weeks": 12, "horizon": 4, "features": FEATURES, "hidden": 64, "layers": 2,
        "dropout": 0.2, "learning_rate": 0.001, "optimizer": "AdamW", "weight_decay": 0.0001,
        "loss": "SmoothL1Loss", "checkpoint_selection": "minimum validation SmoothL1Loss",
        "split": {"train": len(tr), "valid": len(va), "test": len(te)},
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "training_source_sha256": hashlib.sha256((ROOT / "scripts/train_lstm_forecast.py").read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "device": "CPU", "deterministic_algorithms": True},
        "no_seed_selection": True,
    }
    (args.outdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for seed in args.seeds:
        for name, attention in [("lstm_attention", True), ("lstm_pure", False)]:
            print(f"start seed={seed} model={name} epochs={args.epochs}", flush=True)
            run_start = time.perf_counter()
            model = train_model(xs[tr], ys[tr], xs[va], ys[va], len(FEATURES), attention, args.epochs, seed)
            with torch.no_grad():
                pred = model(torch.from_numpy(xs[te])).numpy()
            array_key = f"seed_{seed}_{name}"
            arrays[array_key] = pred
            error = np.abs(pred - truth)
            mean_error = float(error.mean())
            row = {"seed": seed, "model": name, "mae": mean_error,
                   "fva_naive_pct": (1 - mean_error / base) * 100,
                   "fva_ma4_pct": (1 - mean_error / ma4_error) * 100,
                   "step_mae": error.mean(axis=0).tolist(), "parameters": sum(p.numel() for p in model.parameters()),
                   "epochs": args.epochs, "elapsed_seconds": time.perf_counter() - run_start, "array_key": array_key}
            rows.append(row)
            torch.save({"state_dict": model.state_dict(), "seed": seed, "use_attention": attention,
                        "config": config, "x_mean": mu.tolist(), "x_std": sd.tolist(),
                        "y_mean": float(y_mu), "y_std": float(y_sd)}, args.models_dir / f"{array_key}.pt")
            (args.outdir / "runs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"done seed={seed} model={name} mae={mean_error:.6f} FVA={row['fva_naive_pct']:+.2f}% seconds={row['elapsed_seconds']:.1f}", flush=True)
    np.savez_compressed(args.outdir / "predictions.npz", **arrays)
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "step_mae"} for row in rows])
    table.to_csv(args.outdir / "metrics.csv", index=False)
    summaries = []
    for name, group in table.groupby("model"):
        summaries.append({"model": name, "runs": len(group), "mae_mean": float(group["mae"].mean()),
                          "mae_std_ddof1": float(group["mae"].std(ddof=1)),
                          "fva_naive_mean_pct": float(group["fva_naive_pct"].mean()),
                          "fva_naive_std_ddof1_pct": float(group["fva_naive_pct"].std(ddof=1)),
                          "wins_vs_naive": int((group["mae"] < base).sum()),
                          "wins_vs_ma4": int((group["mae"] < ma4_error).sum())})
    paired = table.pivot(index="seed", columns="model", values="mae")
    delta = paired["lstm_attention"] - paired["lstm_pure"]
    summary = {"completed_runs": len(rows), "seeds": args.seeds, "naive_mae": base, "ma4_mae": ma4_error,
               "models": summaries, "attention_minus_pure_mae_mean": float(delta.mean()),
               "attention_minus_pure_mae_std_ddof1": float(delta.std(ddof=1)),
               "attention_wins_paired_seeds": int((delta < 0).sum()),
               "elapsed_seconds": time.perf_counter() - started,
               "scope": "Fixed four-step holdout, 25 epochs per run; seeds are initialization/training repeats, not new temporal folds or independent datasets."}
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
