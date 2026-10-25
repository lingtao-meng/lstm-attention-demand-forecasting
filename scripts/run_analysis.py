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


INK = "#2b2b2b"
TEAL = "#1f6f78"
AMBER = "#c9a227"
RED = "#b23a48"
MUTED = "#9aa6ad"
GRID = "#e8e8e8"


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbf8",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d9d9d9",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
        }
    )


def plot_examples(preds: pd.DataFrame, weekly: pd.DataFrame, holdout_start, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    top = preds.groupby("id", observed=True)["sales"].sum().sort_values(ascending=False).head(3).index
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for idx, (ax, series) in enumerate(zip(axes, top)):
        history = weekly[weekly["id"] == series].sort_values("week").tail(40)
        forecast = preds[preds["id"] == series].sort_values("week")
        n, h = len(history), len(forecast)
        x = np.arange(n)
        dates = history["week"].dt.start_time
        ax.axvspan(n - h - 0.5, n - 0.5, color="#f6efe3", zorder=0)
        ax.plot(x, history["sales"], color=INK, linewidth=1.8, label="actual")
        ax.plot(x[-h:], forecast["naive"], color=AMBER, linewidth=1.4, linestyle="--", label="naive")
        ax.plot(x[-h:], forecast["lgbm"], color=TEAL, linewidth=2.2, label="LightGBM")
        ax.set_title(series.replace("_validation", ""), fontsize=9, loc="left")
        step = 4
        ax.set_xticks(x[::step])
        ax.set_xticklabels(dates.dt.strftime("%b %d").iloc[::step], fontsize=7)
        ax.set_ylim(0, max(history["sales"].max(), forecast[["naive", "lgbm"]].max().max()) * 1.1)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if idx == 0:
            ax.set_ylabel("weekly units")
        if idx == 2:
            ax.text(n - h, ax.get_ylim()[1] * 0.93, "  holdout", color="#8a6d3b", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(images / "forecast_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_models(models: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    order = models.sort_values("mae", ascending=True).reset_index(drop=True)
    display = {"lgbm": "LightGBM", "naive": "naive", "ma4": "MA4", "seasonal": "seasonal"}
    labels = [display.get(m, m) for m in order["model"]]
    y = np.arange(len(order))
    colors = [TEAL if m == "lgbm" else (AMBER if m == "seasonal" else "#a9b4ba") for m in order["model"]]
    xmax = order["mae"].max()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.barh(y, order["mae"], color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for i, row in order.iterrows():
        ax.text(row["mae"] + xmax * 0.015, i, f"{row['mae']:.1f}", va="center", fontsize=8, color=INK)
        ax.text(row["mae"] + xmax * 0.13, i, f"FVA {row['fva_vs_naive']:+.1%}", va="center", fontsize=8, color="#5f6b70")
    ax.set_xlim(0, xmax * 1.38)
    ax.set_xlabel("MAE (weekly units, lower is better)")
    ax.set_title("Same 26-week holdout", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = axes[1]
    ax2.axvline(0, color="#6b6b6b", linewidth=0.8)
    for i, row in order.iterrows():
        color = TEAL if row["model"] == "lgbm" else (AMBER if row["model"] == "seasonal" else MUTED)
        ax2.hlines(i, 0, row["bias"], color=color, linewidth=2.5)
        ax2.scatter(row["bias"], i, color=color, s=28, zorder=3)
        offset = 0.18 if row["bias"] >= 0 else -0.18
        ax2.text(row["bias"] + offset, i, f"{row['bias']:+.2f}", va="center", fontsize=8,
                 ha="left" if row["bias"] >= 0 else "right", color="#5f6b70")
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(-3.2, 7.2)
    ax2.set_xlabel("mean error (actual − forecast)")
    ax2.set_title("Bias", loc="left")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(images / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_newsvendor(frontier: pd.DataFrame, summary: dict, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    _style()
    base = frontier[frontier["model"] == summary["baseline"]].sort_values("service_level")
    lgbm = frontier[frontier["model"] == "lgbm"].sort_values("service_level")
    x = base["service_level"] * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.fill_between(x, lgbm["expected_cost"], base["expected_cost"], color=TEAL, alpha=0.12, label="savings")
    ax.plot(x, base["expected_cost"], color=AMBER, marker="o", markersize=4, linewidth=2, label="best baseline")
    ax.plot(x, lgbm["expected_cost"], color=TEAL, marker="o", markersize=4, linewidth=2, label="LightGBM")
    ax.axvline(summary["critical_ratio"] * 100, color=RED, linestyle="--", linewidth=1.2)
    ax.text(summary["critical_ratio"] * 100 + 1.5, ax.get_ylim()[1] * 0.83,
            f"CR = {summary['critical_ratio']:.2f}", color=RED, fontsize=8)
    base_cost = summary["newsvendor"]["baseline_cost"]
    lgbm_cost = summary["newsvendor"]["lgbm_cost"]
    ax.annotate(
        f"\\${base_cost:.2f} → \\${lgbm_cost:.2f}\n(−{summary['newsvendor']['cost_reduction']:.1%})",
        xy=(50, base_cost),
        xytext=(55, base_cost + 4),
        fontsize=8,
        color="#4a4a4a",
        arrowprops={"arrowstyle": "-", "color": "#9aa6ad", "lw": 0.8},
    )
    ax.set_xlabel("cycle service level (%)")
    ax.set_ylabel("expected cost (USD / series-week)")
    ax.set_title("Cost-service frontier on the same holdout", loc="left")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.set_xticks([50, 60, 70, 80, 90, 95, 99])
    ax.grid(axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(images / "newsvendor.png", dpi=160, bbox_inches="tight")
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
