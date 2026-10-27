#!/usr/bin/env python3
"""Forecast diagnostics: FVA, tracking signal, and newsvendor ordering.

The project reports a 7.7% MAE improvement over a naive forecast.  Two
questions follow that a planning team should ask:

1. **Forecast value added (FVA)** — is the benchmark the right one?  A naive
   lag-1 forecast is a low bar; seasonal naive and a 4-week moving average are
   stronger.  FVA must be measured on the *same* holdout window and against the
   *best* simple benchmark.
2. **What is the forecast worth in inventory terms?**  A newsvendor orders the
   ``critical ratio`` quantile of the forecast-error distribution; the cost of
   ordering too much or too little is explicit.

This script uses only the committed Walmart data and reproducible benchmarks.
The LSTM's dollar MAE is **not** compared directly: the README itself notes it
is back-transformed from standardised space, so the model's value is expressed
as a scenario — a 7.7% reduction applied to the best benchmark's errors.

Outputs: results/fva_by_model.csv, fva_tracking_by_store.csv,
newsvendor_summary.csv, newsvendor_frontier.csv, forecast_diagnostics.json,
images/fva_tracking.png, images/newsvendor.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def load_weekly_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def forecast_errors(df: pd.DataFrame, holdout: int = 26) -> pd.DataFrame:
    """Long table of actual / forecast / error for three simple benchmarks."""
    frames = []
    for store, group in df.groupby("Store"):
        group = group.sort_values("Date").reset_index(drop=True)
        sales = group["Weekly_Sales"]
        predictions = {
            "naive": sales.shift(1),
            "seasonal": sales.shift(52),
            "ma4": sales.rolling(4).mean().shift(1),
        }
        mask = group.index >= len(group) - holdout
        for model, prediction in predictions.items():
            frame = pd.DataFrame(
                {
                    "store": store,
                    "model": model,
                    "actual": sales[mask].to_numpy(),
                    "forecast": prediction[mask].to_numpy(),
                }
            )
            frames.append(frame)
    errors = pd.concat(frames, ignore_index=True).dropna()
    errors["error"] = errors["actual"] - errors["forecast"]
    return errors


def tracking_signal(errors: np.ndarray) -> float:
    """Cumulative error / MAD; |TS| > 4 is the usual bias alarm."""
    errors = np.asarray(errors, dtype=float)
    mad = np.abs(errors).mean()
    return float(errors.sum() / mad) if mad > 0 else 0.0


def fva(mae_model: float, mae_benchmark: float) -> float:
    """Forecast value added: 1 − model MAE / benchmark MAE."""
    return float(1 - mae_model / mae_benchmark)


def critical_ratio(underage: float, overage: float) -> float:
    return float(underage / (underage + overage))


def newsvendor_cost(errors: np.ndarray, order: float, underage: float, overage: float) -> float:
    errors = np.asarray(errors, dtype=float)
    return float(
        np.mean(
            np.where(
                errors > order,
                underage * (errors - order),
                overage * (order - errors),
            )
        )
    )


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in errors.groupby("model", sort=False):
        per_store = group.groupby("store")["error"]
        ts = per_store.apply(tracking_signal)
        mae = float(group["error"].abs().mean())
        rows.append(
            {
                "model": model,
                "mae": mae,
                "rmse": float(np.sqrt((group["error"] ** 2).mean())),
                "bias": float(group["error"].mean()),
                "bias_pct": float(group["error"].mean() / group["actual"].mean()),
                "median_ts": float(ts.median()),
                "biased_stores": int((ts.abs() > 4).sum()),
                "stores": int(group["store"].nunique()),
            }
        )
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = table.iloc[0]
    naive = table.loc[table["model"] == "naive"].iloc[0]
    table["fva_vs_naive"] = [fva(m, naive["mae"]) for m in table["mae"]]
    table["fva_vs_best"] = [fva(m, best["mae"]) for m in table["mae"]]
    return table


def newsvendor(errors: pd.DataFrame, benchmark: str, improvement: float, underage: float, overage: float):
    cr = critical_ratio(underage, overage)
    group = errors[errors["model"] == benchmark]
    rows = []
    for store, chunk in group.groupby("store"):
        e = chunk["error"].to_numpy(dtype=float)
        q_empirical = float(np.quantile(e, cr))
        q_normal = float(e.mean() + stats.norm.ppf(cr) * e.std(ddof=1))
        model_errors = e * (1 - improvement)
        q_model = float(np.quantile(model_errors, cr))
        rows.append(
            {
                "store": store,
                "critical_ratio": cr,
                "q_empirical": q_empirical,
                "q_normal": q_normal,
                "q_model": q_model,
                "cost_empirical": newsvendor_cost(e, q_empirical, underage, overage),
                "cost_normal": newsvendor_cost(e, q_normal, underage, overage),
                "cost_model": newsvendor_cost(model_errors, q_model, underage, overage),
            }
        )
    detail = pd.DataFrame(rows)

    frontier = []
    for service in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        costs = []
        for store, chunk in group.groupby("store"):
            e = chunk["error"].to_numpy(dtype=float)
            q = float(np.quantile(e, service))
            costs.append(newsvendor_cost(e, q, underage, overage))
        frontier.append({"service_level": service, "expected_cost": float(np.mean(costs))})
    return cr, detail, pd.DataFrame(frontier)


def plot(models: pd.DataFrame, errors: pd.DataFrame, nv_detail: pd.DataFrame, frontier: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    colors = {"naive": "#9b3b3b", "seasonal": "#d99a2b", "ma4": "#1f6f78"}
    ax.bar(models["model"], models["mae"], color=[colors.get(m, "#7c8a99") for m in models["model"]])
    for i, row in enumerate(models.itertuples()):
        ax.text(i, row.mae * 1.02, f"FVA {row.fva_vs_naive:+.1%}", ha="center", fontsize=8)
    ax.set_ylabel("MAE (USD / store-week)")
    ax.set_title("Same holdout window: which benchmark is actually strongest?")
    ax.set_ylim(0, models["mae"].max() * 1.18)

    ax2 = axes[1]
    jitter = {"naive": 0, "seasonal": 1, "ma4": 2}
    for model, group in errors.groupby("model"):
        ts = group.groupby("store")["error"].apply(tracking_signal).to_numpy()
        x = jitter[model] + np.random.default_rng(1).uniform(-0.15, 0.15, len(ts))
        ax2.scatter(x, ts, s=16, alpha=0.7, color=colors.get(model, "#7c8a99"), label=model)
    ax2.axhline(4, color="#9b3b3b", linestyle="--", linewidth=1)
    ax2.axhline(-4, color="#9b3b3b", linestyle="--", linewidth=1)
    ax2.set_xticks([0, 1, 2], ["naive", "seasonal", "ma4"])
    ax2.set_ylabel("tracking signal (Σ error / MAD)")
    ax2.set_title("Bias: |TS| > 4 per store")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(images / "fva_tracking.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    model_errors = errors[errors["model"] == "ma4"]["error"].to_numpy()
    ax.hist(model_errors, bins=60, color="#2f5d8c", alpha=0.75)
    cr = float(nv_detail["critical_ratio"].iloc[0])
    q_emp = float(np.quantile(model_errors, cr))
    q_norm = float(model_errors.mean() + stats.norm.ppf(cr) * model_errors.std(ddof=1))
    ax.axvline(q_emp, color="#1f6f78", linewidth=2, label=f"empirical q (CR={cr:.2f})")
    ax.axvline(q_norm, color="#d99a2b", linewidth=2, linestyle="--", label="normal q")
    ax.set_xlabel("forecast error (actual − forecast, USD)")
    ax.set_ylabel("store-weeks")
    ax.set_title("MA4 errors: the empirical quantile is not normal")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(frontier["service_level"] * 100, frontier["expected_cost"], marker="o", color="#1f6f78")
    ax2.axvline(cr * 100, color="#9b3b3b", linestyle="--", linewidth=1, label="critical ratio")
    ax2.set_xlabel("cycle service level (%)")
    ax2.set_ylabel("expected newsvendor cost (USD / store-week)")
    ax2.set_title("Cost-service frontier for the MA4 benchmark")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(images / "newsvendor.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "Walmart.csv"))
    ap.add_argument("--holdout", type=int, default=26, help="weeks at the end of each store")
    ap.add_argument("--underage", type=float, default=0.30, help="margin lost per dollar of unmet demand")
    ap.add_argument("--overage", type=float, default=0.25, help="holding cost per dollar left over")
    ap.add_argument("--improvement", type=float, default=0.0771, help="scenario error reduction vs benchmark")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    df = load_weekly_sales(Path(args.data))
    errors = forecast_errors(df, args.holdout)
    models = summarise(errors)
    best = str(models.iloc[0]["model"])

    tracking = (
        errors.groupby(["model", "store"])["error"]
        .apply(tracking_signal)
        .rename("tracking_signal")
        .reset_index()
    )

    cr, nv_detail, frontier = newsvendor(
        errors, best, args.improvement, args.underage, args.overage
    )
    summary = {
        "holdout_weeks": args.holdout,
        "stores": int(errors["store"].nunique()),
        "best_benchmark": best,
        "critical_ratio": cr,
        "underage": args.underage,
        "overage": args.overage,
        "improvement_scenario": args.improvement,
        "models": models.to_dict(orient="records"),
        "newsvendor": {
            "cost_empirical": float(nv_detail["cost_empirical"].mean()),
            "cost_normal": float(nv_detail["cost_normal"].mean()),
            "cost_model": float(nv_detail["cost_model"].mean()),
            "normal_over_empirical": float(
                nv_detail["cost_normal"].mean() / nv_detail["cost_empirical"].mean() - 1
            ),
            "model_saving_vs_empirical": float(
                1 - nv_detail["cost_model"].mean() / nv_detail["cost_empirical"].mean()
            ),
        },
        "frontier": frontier.to_dict(orient="records"),
    }

    models.to_csv(RESULTS / "fva_by_model.csv", index=False)
    tracking.to_csv(RESULTS / "fva_tracking_by_store.csv", index=False)
    nv_detail.to_csv(RESULTS / "newsvendor_summary.csv", index=False)
    frontier.to_csv(RESULTS / "newsvendor_frontier.csv", index=False)
    (RESULTS / "forecast_diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    print(f"holdout: last {args.holdout} weeks, {summary['stores']} stores")
    print(models.to_string(index=False))
    print(f"\nbest benchmark: {best}")
    print(f"newsvendor CR={cr:.3f}: empirical ${summary['newsvendor']['cost_empirical']:,.0f}, "
          f"normal ${summary['newsvendor']['cost_normal']:,.0f} "
          f"({summary['newsvendor']['normal_over_empirical']:+.1%}), "
          f"model scenario ${summary['newsvendor']['cost_model']:,.0f} "
          f"({-summary['newsvendor']['model_saving_vs_empirical']:.1%})")

    plot(models, errors, nv_detail, frontier, IMAGES)
    print(f"wrote {IMAGES / 'fva_tracking.png'} and {IMAGES / 'newsvendor.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
