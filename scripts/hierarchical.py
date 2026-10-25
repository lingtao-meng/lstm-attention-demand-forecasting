#!/usr/bin/env python3
"""Hierarchical forecasting and reconciliation on the M5 panel.

The 100 item-store series roll up to 10 stores, 3 states and one total.  Four
forecast combinations are compared on the same 26-week holdout:

* base: each level forecast independently (not coherent);
* bottom-up: sum the bottom forecasts;
* top-down: forecast the total and split it by historical shares;
* MinT-OLS: project the base forecasts onto the coherent subspace.

The point is the trade-off: reconciliation makes the levels add up, but it can
move accuracy away from the bottom level.
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

from src.hierarchy import Hierarchy, bottom_up, build_hierarchy, min_t_ols, top_down  # noqa: E402
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


def fit_residual(train: pd.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    weeks = sorted(train["week"].unique())
    valid_start = weeks[-8]
    fit = train[train["week"] < valid_start]
    valid = train[train["week"] >= valid_start]
    fit = fit.assign(target=fit["sales"] - fit["naive"])
    valid = valid.assign(target=valid["sales"] - valid["naive"])
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.04,
        num_leaves=31,
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


def forecast_level(frame: pd.DataFrame, holdout: int) -> pd.DataFrame:
    df = add_baselines(add_features(frame))
    features = feature_columns(df)
    train, test, _ = time_split(df, holdout)
    train = train.dropna(subset=features + ["sales"])
    test = test.dropna(subset=features).copy()
    model = fit_residual(train, features)
    test["pred"] = np.maximum(0, test["naive"].to_numpy() + model.predict(test[features]))
    return test[["id", "week", "sales", "pred"]]


def make_frames(weekly: pd.DataFrame):
    bottom = weekly[["id", "store_id", "state_id", "week", "sales", "event", "snap", "price"]].copy()
    store = (
        weekly.groupby(["store_id", "week"], observed=True, as_index=False)
        .agg(sales=("sales", "sum"), event=("event", "max"), snap=("snap", "max"))
        .rename(columns={"store_id": "id"})
    )
    store["store_id"] = store["id"]
    state = (
        weekly.groupby(["state_id", "week"], observed=True, as_index=False)
        .agg(sales=("sales", "sum"), event=("event", "max"), snap=("snap", "max"))
        .rename(columns={"state_id": "id"})
    )
    state["store_id"] = state["id"]
    total = (
        weekly.groupby("week", as_index=False)
        .agg(sales=("sales", "sum"), event=("event", "max"), snap=("snap", "max"))
    )
    total["id"] = "total"
    total["store_id"] = "total"
    return bottom, store, state, total


def to_matrix(frame: pd.DataFrame, columns: list[str], weeks: pd.Index) -> np.ndarray:
    pivot = frame.pivot(index="week", columns="id", values="sales").reindex(weeks)
    return pivot.reindex(columns=columns).to_numpy(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    sales = load_sales(Path(args.data_dir) / "sales_train_validation.csv")
    calendar = pd.read_csv(Path(args.data_dir) / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(Path(args.data_dir) / "sell_prices.csv")
    selected = select_series(sales, args.series, args.seed)
    weekly = weekly_panel(selected, calendar, prices)
    bottom, store, state, total = make_frames(weekly)
    holdout_start = sorted(weekly["week"].unique())[-args.holdout]
    hierarchy = build_hierarchy(bottom[bottom["week"] < holdout_start])

    forecasts = {
        "bottom": forecast_level(bottom, args.holdout),
        "store": forecast_level(store, args.holdout),
        "state": forecast_level(state, args.holdout),
        "total": forecast_level(total, args.holdout),
    }
    print("forecasts done")

    weeks = sorted(forecasts["total"]["week"].unique())
    state_ids = [label.split(":", 1)[1] for label in hierarchy.labels if label.startswith("state:")]
    store_ids = [label.split(":", 1)[1] for label in hierarchy.labels if label.startswith("store:")]
    actual_rows, base_rows = [], []
    for frame, ids in [
        (forecasts["total"], ["total"]),
        (forecasts["state"], state_ids),
        (forecasts["store"], store_ids),
        (forecasts["bottom"], hierarchy.bottom_ids),
    ]:
        actual_pivot = frame.pivot(index="week", columns="id", values="sales").reindex(weeks)
        base_pivot = frame.pivot(index="week", columns="id", values="pred").reindex(weeks)
        for node_id in ids:
            actual_rows.append(actual_pivot[node_id].to_numpy(dtype=float))
            base_rows.append(base_pivot[node_id].to_numpy(dtype=float))
    actual = np.vstack(actual_rows)
    base = np.vstack(base_rows)

    level_array = np.array(hierarchy.levels)
    methods = {
        "base": base,
        "bottom-up": bottom_up(hierarchy, base[level_array == "bottom"]),
        "top-down": top_down(hierarchy, base[0]),
        "MinT-OLS": min_t_ols(hierarchy, base),
    }

    level_index = np.array(hierarchy.levels)
    rows = []
    node_rows = []
    for method, forecast in methods.items():
        mae = np.abs(forecast - actual).mean(axis=1)
        gap = float(np.abs(forecast[0] - forecast[-len(hierarchy.bottom_ids):].sum(axis=0)).max())
        rows.append(
            {
                "method": method,
                "bottom_mae": float(mae[level_index == "bottom"].mean()),
                "store_mae": float(mae[level_index == "store"].mean()),
                "state_mae": float(mae[level_index == "state"].mean()),
                "total_mae": float(mae[level_index == "total"].mean()),
                "coherence_gap": gap,
            }
        )
        for label, value in zip(hierarchy.labels, mae):
            node_rows.append({"method": method, "node": label, "level": level_index[hierarchy.labels.index(label)], "mae": float(value)})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "hierarchical_mae.csv", index=False)
    pd.DataFrame(node_rows).to_csv(RESULTS / "hierarchical_nodes.csv", index=False)

    summary = {
        "nodes": len(hierarchy.labels),
        "bottom_nodes": len(hierarchy.bottom_ids),
        "holdout_start": str(weeks[0]),
        "results": table.to_dict(orient="records"),
    }
    (RESULTS / "hierarchical.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(table.to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    levels = ["bottom", "store", "state", "total"]
    colors = {"base": "#9aa6ad", "bottom-up": "#c9a227", "top-down": "#b8c4c9", "MinT-OLS": "#1f6f78"}
    width = 0.2
    x = np.arange(len(levels))
    for i, method in enumerate(methods):
        values = [table.loc[table["method"] == method, f"{level}_mae"].iloc[0] for level in levels]
        means = [actual[level_index == level].mean() for level in levels]
        relative = [value / mean * 100 for value, mean in zip(values, means)]
        ax.bar(x + (i - 1.5) * width, relative, width=width, color=colors[method], label=method)
    ax.set_xticks(x, ["item-store", "store", "state", "total"])
    ax.set_ylabel("MAE / mean sales (%)")
    ax.set_title("Reconciliation by level", loc="left")
    ax.legend(fontsize=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = axes[1]
    x_weeks = np.arange(len(weeks))
    ax2.plot(x_weeks, actual[0], color="#2b2b2b", linewidth=2.0, label="actual")
    ax2.plot(x_weeks, base[0], color="#9aa6ad", linewidth=1.3, linestyle="--", label="base total")
    ax2.plot(x_weeks, methods["bottom-up"][0], color="#c9a227", linewidth=1.8, label="bottom-up")
    ax2.plot(x_weeks, methods["MinT-OLS"][0], color="#1f6f78", linewidth=1.8, label="MinT-OLS")
    ax2.set_xticks(x_weeks[::4])
    ax2.set_xticklabels([str(week)[:10] for week in weeks][::4], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("weekly units")
    ax2.set_title("Total forecast on the holdout", loc="left")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(IMAGES / "hierarchical.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {IMAGES / 'hierarchical.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
