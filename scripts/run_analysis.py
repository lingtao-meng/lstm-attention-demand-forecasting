#!/usr/bin/env python3
"""M5 weekly demand forecasting and newsvendor evaluation.

The pipeline:

1. pick 100 item-store series (top 5 per store + 50 random);
2. aggregate daily sales to weeks and build lag/rolling/price/calendar features;
3. forecast the last 26 weeks with naive, seasonal naive, MA4 and LightGBM;
4. compare MAE / FVA and translate the errors into newsvendor cost.

The data is not committed.  Run ``python scripts/download_data.py`` first, or
put ``calendar.csv``, ``sales_train_validation.csv`` and ``sell_prices.csv`` in
``data/``.
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

from src.evaluation import evaluate, newsvendor, per_series  # noqa: E402
from src.m5 import (  # noqa: E402
    CAT_COLS,
    add_baselines,
    add_features,
    feature_columns,
    load_sales,
    select_series,
    time_split,
    weekly_panel,
)

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def load_inputs(data_dir: Path):
    sales = load_sales(data_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices_path = data_dir / "sell_prices.csv"
    prices = pd.read_csv(prices_path) if prices_path.exists() else None
    return sales, calendar, prices


def train_model(train: pd.DataFrame, features: list[str], direct: bool = False) -> lgb.LGBMRegressor:
    weeks = sorted(train["week"].unique())
    valid_start = weeks[-8] if len(weeks) > 8 else weeks[0]
    fit = train[train["week"] < valid_start]
    valid = train[train["week"] >= valid_start]
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
    if direct:
        fit = fit.assign(target=fit["sales"])
        valid = valid.assign(target=valid["sales"])
    else:
        fit = fit.assign(target=fit["sales"] - fit["naive"])
        valid = valid.assign(target=valid["sales"] - valid["naive"])
    model.fit(
        fit[features],
        fit["target"],
        eval_set=[(valid[features], valid["target"])],
        eval_metric="l1",
        categorical_feature=[c for c in CAT_COLS if c in features],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def plot_examples(preds: pd.DataFrame, weekly: pd.DataFrame, holdout_start, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = preds.groupby("id", observed=True)["sales"].sum().sort_values(ascending=False).head(3).index
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, series in zip(axes, top):
        history = weekly[weekly["id"] == series].sort_values("week").tail(40)
        forecast = preds[preds["id"] == series].sort_values("week")
        history_x = history["week"].dt.start_time.dt.strftime("%Y-%m-%d")
        forecast_x = forecast["week"].dt.start_time.dt.strftime("%Y-%m-%d")
        ax.plot(history_x, history["sales"], color="#333", linewidth=1.2, label="actual")
        ax.plot(forecast_x, forecast["naive"], color="#d99a2b", linestyle="--", linewidth=1, label="naive")
        ax.plot(forecast_x, forecast["lgbm"], color="#1f6f78", linewidth=1.5, label="LightGBM")
        ax.axvline(len(history) - len(forecast) - 0.5, color="#9b3b3b", linestyle=":", linewidth=1)
        ax.set_title(series, fontsize=9)
        ax.set_xticks(range(0, len(history), 4))
        ax.set_xticklabels(history_x.iloc[::4], rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("weekly units")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(images / "forecast_examples.png", dpi=150)
    plt.close(fig)


def plot_models(models: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    ax = axes[0]
    colors = {"naive": "#9b3b3b", "seasonal": "#d99a2b", "ma4": "#2f5d8c", "lgbm": "#1f6f78"}
    ax.bar(models["model"], models["mae"], color=[colors.get(m, "#7c8a99") for m in models["model"]])
    for i, row in enumerate(models.itertuples()):
        ax.text(i, row.mae * 1.02, f"FVA {row.fva_vs_naive:+.1%}", ha="center", fontsize=8)
    ax.set_ylabel("MAE (weekly units / series)")
    ax.set_title("Same 26-week holdout: baselines vs LightGBM")

    ax2 = axes[1]
    ax2.bar(models["model"], models["bias"], color=[colors.get(m, "#7c8a99") for m in models["model"]])
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.set_ylabel("mean error (actual − forecast)")
    ax2.set_title("Bias")
    fig.tight_layout()
    fig.savefig(images / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_newsvendor(frontier: pd.DataFrame, summary: dict, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model, label, color in [
        (summary["baseline"], "best baseline", "#d99a2b"),
        ("lgbm", "LightGBM", "#1f6f78"),
    ]:
        view = frontier[frontier["model"] == model]
        ax.plot(view["service_level"] * 100, view["expected_cost"], marker="o", color=color, label=label)
    ax.axvline(summary["critical_ratio"] * 100, color="#9b3b3b", linestyle="--", linewidth=1, label="critical ratio")
    ax.set_xlabel("cycle service level (%)")
    ax.set_ylabel("expected newsvendor cost (USD / series-week)")
    ax.set_title("Cost-service frontier on the same holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(images / "newsvendor.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direct", action="store_true", help="predict sales directly instead of the naive residual")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    sales, calendar, prices = load_inputs(Path(args.data_dir))
    selected = select_series(sales, args.series, args.seed)
    selected[["id", "item_id", "store_id", "cat_id"]].to_csv(RESULTS / "selected_series.csv", index=False)
    print(f"selected {len(selected)} series")

    weekly = weekly_panel(selected, calendar, prices)
    df = add_baselines(add_features(weekly))
    features = feature_columns(df)
    train, test, holdout_start = time_split(df, args.holdout)
    train = train.dropna(subset=features + ["sales"])
    test = test.dropna(subset=features)
    print(f"train {len(train):,} rows, test {len(test):,} rows, holdout starts {holdout_start}")

    model = train_model(train, features, direct=args.direct)
    preds = test[["id", "week", "sales", "price", "naive", "seasonal", "ma4"]].copy()
    prediction = model.predict(test[features])
    if args.direct:
        preds["lgbm"] = np.maximum(0, prediction)
    else:
        preds["lgbm"] = np.maximum(0, test["naive"].to_numpy() + prediction)
    output = preds.copy()
    output["week"] = output["week"].astype(str)
    output.to_csv(RESULTS / "predictions_test.csv", index=False)

    models = evaluate(preds)
    detail = per_series(preds)
    metrics_name = "model_metrics_direct.csv" if args.direct else "model_metrics.csv"
    models.to_csv(RESULTS / metrics_name, index=False)
    detail.to_csv(RESULTS / "per_series.csv", index=False)

    baseline = models[models["model"] != "lgbm"].sort_values("mae").iloc[0]["model"]
    base_detail, base_frontier = newsvendor(preds, baseline)
    lgbm_detail, lgbm_frontier = newsvendor(preds, "lgbm")
    base_frontier["model"] = baseline
    lgbm_frontier["model"] = "lgbm"
    frontier = pd.concat([base_frontier, lgbm_frontier], ignore_index=True)
    frontier.to_csv(RESULTS / "newsvendor_frontier.csv", index=False)

    summary = {
        "series": int(preds["id"].nunique()),
        "holdout_weeks": args.holdout,
        "holdout_start": str(holdout_start),
        "baseline": baseline,
        "critical_ratio": float(base_detail["critical_ratio"].iloc[0]),
        "models": models.to_dict(orient="records"),
        "newsvendor": {
            "baseline_cost": float(base_detail["cost_empirical"].mean()),
            "lgbm_cost": float(lgbm_detail["cost_empirical"].mean()),
            "cost_reduction": float(1 - lgbm_detail["cost_empirical"].mean() / base_detail["cost_empirical"].mean()),
            "normal_premium": float(base_detail["cost_normal"].mean() / base_detail["cost_empirical"].mean() - 1),
        },
    }
    (RESULTS / "diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"policy": baseline, "cost": summary["newsvendor"]["baseline_cost"]},
            {"policy": "lgbm", "cost": summary["newsvendor"]["lgbm_cost"]},
        ]
    ).to_csv(RESULTS / "newsvendor_summary.csv", index=False)

    print(models.to_string(index=False))
    print(f"newsvendor: {baseline} ${summary['newsvendor']['baseline_cost']:,.0f}, "
          f"lgbm ${summary['newsvendor']['lgbm_cost']:,.0f} "
          f"({-summary['newsvendor']['cost_reduction']:.1%})")

    plot_examples(preds, df, holdout_start, IMAGES)
    plot_models(models, IMAGES)
    plot_newsvendor(frontier, summary, IMAGES)
    print(f"wrote {IMAGES / 'forecast_examples.png'}, {IMAGES / 'model_comparison.png'}, {IMAGES / 'newsvendor.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
