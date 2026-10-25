#!/usr/bin/env python3
"""Rolling-origin backtest and Diebold-Mariano test.

A single 26-week holdout can flatter or punish a model.  This script walks the
same 100 series through six non-overlapping 26-week test blocks, refits the
residual LightGBM each time, and reports the FVA distribution plus a paired
Diebold-Mariano test against naive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import MODELS  # noqa: E402
from src.m5 import (  # noqa: E402
    CAT_COLS,
    add_baselines,
    add_features,
    feature_columns,
    load_sales,
    select_series,
    weekly_panel,
)
from src.probabilistic import dm_test  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def fit_model(train: pd.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    weeks = sorted(train["week"].unique())
    valid_start = weeks[-8]
    fit = train[train["week"] < valid_start]
    valid = train[train["week"] >= valid_start]
    fit = fit.assign(target=fit["sales"] - fit["naive"])
    valid = valid.assign(target=valid["sales"] - valid["naive"])
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        fit[features],
        fit["target"],
        eval_set=[(valid[features], valid["target"])],
        eval_metric="l1",
        categorical_feature=[c for c in CAT_COLS if c in features],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--block", type=int, default=26)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    sales = load_sales(Path(args.data_dir) / "sales_train_validation.csv")
    calendar = pd.read_csv(Path(args.data_dir) / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(Path(args.data_dir) / "sell_prices.csv")
    selected = select_series(sales, args.series, args.seed)
    df = add_baselines(add_features(weekly_panel(selected, calendar, prices)))
    features = feature_columns(df)
    weeks = sorted(df["week"].unique())

    rows = []
    for fold in range(1, args.folds + 1):
        test_end = len(weeks) - args.block * (fold - 1)
        test_start = test_end - args.block
        test_weeks = weeks[test_start:test_end]
        train = df[df["week"] < test_weeks[0]].dropna(subset=features + ["sales"])
        test = df[df["week"].isin(test_weeks)].dropna(subset=features)
        if len(train) < 1000 or len(test) == 0:
            continue
        model = fit_model(train, features)
        test = test.copy()
        test["lgbm"] = np.maximum(0, test["naive"].to_numpy() + model.predict(test[features]))
        result = {"fold": fold, "test_start": str(test_weeks[0]), "test_end": str(test_weeks[-1])}
        for name in MODELS:
            error = (test["sales"] - test[name]).to_numpy()
            result[f"mae_{name}"] = float(np.abs(error).mean())
        best_baseline = min(result[f"mae_{m}"] for m in ("naive", "seasonal", "ma4"))
        result["fva_naive"] = 1 - result["mae_lgbm"] / result["mae_naive"]
        result["fva_best"] = 1 - result["mae_lgbm"] / best_baseline
        statistic, p_value = dm_test(
            (test["sales"] - test["lgbm"]).to_numpy(),
            (test["sales"] - test["naive"]).to_numpy(),
        )
        result["dm_stat"] = statistic
        result["dm_p"] = p_value
        rows.append(result)
        print(f"fold {fold} {test_weeks[0]}..{test_weeks[-1]} "
              f"FVA {result['fva_naive']:+.1%}, DM p={p_value:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "rolling_backtest.csv", index=False)
    summary = {
        "folds": len(table),
        "fva_mean": float(table["fva_naive"].mean()),
        "fva_std": float(table["fva_naive"].std(ddof=1)),
        "fva_min": float(table["fva_naive"].min()),
        "fva_max": float(table["fva_naive"].max()),
        "dm_significant_folds": int((table["dm_p"] < 0.05).sum()),
        "dm_better_folds": int(((table["dm_stat"] < 0) & (table["dm_p"] < 0.05)).sum()),
        "dm_worse_folds": int(((table["dm_stat"] > 0) & (table["dm_p"] < 0.05)).sum()),
        "table": table.to_dict(orient="records"),
    }
    (RESULTS / "rolling_backtest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"FVA mean {summary['fva_mean']:+.1%} ± {summary['fva_std']:.1%}, "
          f"range {summary['fva_min']:+.1%}..{summary['fva_max']:+.1%}, "
          f"DM significant {summary['dm_significant_folds']}/{len(table)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    ax = axes[0]
    ax.bar(table["fold"], table["fva_naive"] * 100, color="#1f6f78", width=0.55)
    ax.axhline(0, color="#6b6b6b", linewidth=0.8)
    ax.axhline(summary["fva_mean"] * 100, color="#b23a48", linestyle="--", linewidth=1,
               label=f"mean {summary['fva_mean']:+.1%}")
    ax.set_xticks(table["fold"])
    ax.set_xlabel("fold (1 = latest test block)")
    ax.set_ylabel("FVA vs naive (%)")
    ax.set_title("Rolling-origin FVA", loc="left")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for model, color, label in [("naive", "#c9a227", "naive"), ("ma4", "#9aa6ad", "MA4"),
                                ("seasonal", "#b8c4c9", "seasonal"), ("lgbm", "#1f6f78", "LightGBM")]:
        ax2.plot(table["fold"], table[f"mae_{model}"], marker="o", linewidth=1.8, color=color, label=label)
    ax2.set_xticks(table["fold"])
    ax2.set_xlabel("fold (1 = latest test block)")
    ax2.set_ylabel("MAE (weekly units)")
    ax2.set_title("MAE by fold", loc="left")
    ax2.legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "rolling_backtest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {IMAGES / 'rolling_backtest.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
