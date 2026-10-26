#!/usr/bin/env python3
"""Quantile LightGBM + conformal calibration + newsvendor.

Three quantile models (P10/P50/P90) are trained on the naive residual.  The
P10/P90 interval is widened by a split-conformal adjustment on the last 8
weeks of the training period; the P50 residual quantile at the newsvendor
critical ratio is used to place the order directly.

The point-forecast pipeline lives in ``run_analysis.py``; this script is the
probabilistic extension.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import newsvendor  # noqa: E402
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
from src.probabilistic import (  # noqa: E402
    conformal_adjustment,
    conformal_quantile,
    interval_coverage,
    interval_width,
    pinball_loss,
)

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
INK = "#2b2b2b"
TEAL = "#1f6f78"
AMBER = "#c9a227"
RED = "#b23a48"
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


def fit_quantile(train: pd.DataFrame, features: list[str], alpha: float) -> lgb.LGBMRegressor:
    target = train["sales"] - train["naive"]
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train[features],
        target,
        categorical_feature=[c for c in CAT_COLS if c in features],
    )
    return model


def plot_forecast(preds: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    top = preds.groupby("id", observed=True)["actual"].sum().sort_values(ascending=False).head(2).index
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, series in zip(axes, top):
        view = preds[preds["id"] == series].sort_values("week").tail(30)
        x = np.arange(len(view))
        labels = view["week"].dt.start_time.dt.strftime("%b %d")
        ax.fill_between(x, view["lower80"], view["upper80"], color=TEAL, alpha=0.18, label="80% interval")
        ax.plot(x, view["actual"], color=INK, linewidth=1.8, label="actual")
        ax.plot(x, view["q50"], color=TEAL, linewidth=2.0, label="P50")
        ax.plot(x, view["naive"], color=AMBER, linewidth=1.2, linestyle="--", label="naive")
        ax.set_title(series.replace("_validation", ""), fontsize=9, loc="left")
        ax.set_xticks(x[::4])
        ax.set_xticklabels(labels.iloc[::4], fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("weekly units")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(images / "probabilistic_forecast.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(metrics: pd.DataFrame, coverage: float, nominal: float, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.bar(metrics["quantile"].astype(str), metrics["pinball"], color=[AMBER, TEAL, AMBER])
    for i, value in enumerate(metrics["pinball"]):
        ax.text(i, value * 1.02, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("pinball loss (lower is better)")
    ax.set_title("Quantile loss", loc="left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = axes[1]
    bars = ax2.bar(["nominal", "actual"], [nominal * 100, coverage * 100], color=["#a9b4ba", TEAL], width=0.55)
    for bar, value in zip(bars, [nominal * 100, coverage * 100]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("80% interval coverage")
    ax2.set_title("Conformal calibration", loc="left")
    ax2.grid(axis="y")
    ax2.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(images / "probabilistic_calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_newsvendor(summary: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colors = [AMBER, TEAL, "#a9b4ba"]
    bars = ax.bar(summary["policy"], summary["cost"], color=colors, width=0.55)
    for bar, value in zip(bars, summary["cost"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"${value:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("expected newsvendor cost (USD / series-week)")
    ax.set_title("Order at the 0.545 quantile", loc="left")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.set_ylim(0, summary["cost"].max() * 1.18)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(images / "probabilistic_newsvendor.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--underage", type=float, default=0.30)
    ap.add_argument("--overage", type=float, default=0.25)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    sales = load_sales(Path(args.data_dir) / "sales_train_validation.csv")
    calendar = pd.read_csv(Path(args.data_dir) / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(Path(args.data_dir) / "sell_prices.csv")
    selected = select_series(sales, args.series, args.seed)
    weekly = weekly_panel(selected, calendar, prices)
    df = add_baselines(add_features(weekly))
    features = feature_columns(df)
    train, test, holdout_start = time_split(df, args.holdout)
    train = train.dropna(subset=features + ["sales"])
    test = test.dropna(subset=features)
    print(f"train {len(train):,}, test {len(test):,}, holdout {holdout_start}")

    weeks = sorted(train["week"].unique())
    cal_start = weeks[-16]
    proper = train[train["week"] < cal_start]
    calibration = train[train["week"] >= cal_start]

    alphas = [0.1, 0.5, 0.9]
    models = {alpha: fit_quantile(proper, features, alpha) for alpha in alphas}
    quantiles = {}
    for alpha, model in models.items():
        quantiles[alpha] = {
            "cal": calibration["naive"].to_numpy() + model.predict(calibration[features]),
            "test": test["naive"].to_numpy() + model.predict(test[features]),
        }

    lower = quantiles[0.1]["test"]
    upper = quantiles[0.9]["test"]
    scores = np.maximum(
        quantiles[0.1]["cal"] - calibration["sales"].to_numpy(),
        calibration["sales"].to_numpy() - quantiles[0.9]["cal"],
    )
    adjustment = conformal_adjustment(scores, 0.20)
    lower = lower - adjustment
    upper = upper + adjustment

    coverage = interval_coverage(test["sales"].to_numpy(), lower, upper)
    width = interval_width(lower, upper)
    metrics = pd.DataFrame(
        [
            {"quantile": alpha, "pinball": pinball_loss(test["sales"].to_numpy(), quantiles[alpha]["test"], alpha)}
            for alpha in alphas
        ]
    )

    preds = test[["id", "week", "sales", "price", "naive"]].copy()
    preds["actual"] = preds["sales"]
    preds["q10"] = quantiles[0.1]["test"]
    preds["q50"] = quantiles[0.5]["test"]
    preds["q90"] = quantiles[0.9]["test"]
    preds["lower80"] = lower
    preds["upper80"] = upper
    preds.to_csv(RESULTS / "quantile_predictions.csv", index=False)

    cr = args.underage / (args.underage + args.overage)
    cal_frame = calibration[["id", "sales", "naive"]].copy()
    cal_frame["q50"] = quantiles[0.5]["cal"]
    cal_frame["error_base"] = cal_frame["sales"] - cal_frame["naive"]
    cal_frame["resid_model"] = cal_frame["sales"] - cal_frame["q50"]

    base_map = cal_frame.groupby("id", observed=True)["error_base"].apply(
        lambda values: conformal_quantile(values.to_numpy(), cr)
    )
    model_map = cal_frame.groupby("id", observed=True)["resid_model"].apply(
        lambda values: conformal_quantile(values.to_numpy(), cr)
    )
    normal_map = cal_frame.groupby("id", observed=True)["error_base"].apply(
        lambda values: values.mean() + stats.norm.ppf(cr) * values.std(ddof=1)
    )
    preds["order_base"] = preds["naive"] + preds["id"].map(base_map)
    preds["order_model"] = preds["q50"] + preds["id"].map(model_map)
    preds["order_normal"] = preds["naive"] + preds["id"].map(normal_map)

    def cost(order: np.ndarray) -> float:
        actual = preds["actual"].to_numpy()
        price = preds["price"].to_numpy()
        return float(np.mean(np.where(actual > order, args.underage * price * (actual - order),
                                      args.overage * price * (order - actual))))

    summary = pd.DataFrame(
        [
            {"policy": "naive + conformal", "cost": cost(preds["order_base"].to_numpy())},
            {"policy": "P50 + conformal", "cost": cost(preds["order_model"].to_numpy())},
            {"policy": "naive + normal", "cost": cost(preds["order_normal"].to_numpy())},
        ]
    )
    summary.to_csv(RESULTS / "probabilistic_newsvendor.csv", index=False)
    metrics.to_csv(RESULTS / "probabilistic_metrics.csv", index=False)

    payload = {
        "holdout_start": str(holdout_start),
        "critical_ratio": cr,
        "conformal_adjustment": adjustment,
        "coverage_80": coverage,
        "interval_width_80": width,
        "metrics": metrics.to_dict(orient="records"),
        "newsvendor": summary.to_dict(orient="records"),
    }
    (RESULTS / "probabilistic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    print(metrics.to_string(index=False))
    print(f"80% coverage {coverage:.1%}, width {width:.1f}")
    print(summary.to_string(index=False))

    plot_forecast(preds, IMAGES)
    plot_calibration(metrics, coverage, 0.80, IMAGES)
    plot_newsvendor(summary, IMAGES)
    print(f"wrote probabilistic charts to {IMAGES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
