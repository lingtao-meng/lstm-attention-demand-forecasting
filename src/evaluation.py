"""Forecast metrics, FVA and newsvendor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["naive", "seasonal", "ma4", "lgbm"]


def tracking_signal(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=float)
    mad = np.abs(errors).mean()
    return float(errors.sum() / mad) if mad > 0 else 0.0


def fva(model_mae: float, benchmark_mae: float) -> float:
    return float(1 - model_mae / benchmark_mae) if benchmark_mae > 0 else 0.0


def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics and FVA for each forecast column."""
    rows = []
    for model in MODELS:
        valid = preds.dropna(subset=["sales", model])
        error = valid["sales"] - valid[model]
        ts = valid.assign(error=error).groupby("id", observed=True)["error"].apply(tracking_signal)
        rows.append(
            {
                "model": model,
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "bias": float(error.mean()),
                "bias_pct": float(error.mean() / valid["sales"].mean()),
                "median_ts": float(ts.median()),
                "biased_series": int((ts.abs() > 4).sum()),
                "series": int(valid["id"].nunique()),
            }
        )
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = table.iloc[0]
    naive = table.loc[table["model"] == "naive"].iloc[0]
    table["fva_vs_naive"] = [fva(m, naive["mae"]) for m in table["mae"]]
    table["fva_vs_best"] = [fva(m, best["mae"]) for m in table["mae"]]
    return table


def per_series(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        valid = preds.dropna(subset=["sales", model])
        for series, chunk in valid.groupby("id", observed=True):
            error = chunk["sales"] - chunk[model]
            rows.append(
                {
                    "id": series,
                    "model": model,
                    "mae": float(error.abs().mean()),
                    "bias": float(error.mean()),
                    "ts": tracking_signal(error),
                }
            )
    return pd.DataFrame(rows)


def newsvendor(preds: pd.DataFrame, model: str, underage: float = 0.30, overage: float = 0.25):
    """Per-series empirical newsvendor, with a normal-quantile comparison."""
    cr = underage / (underage + overage)
    rows = []
    for series, chunk in preds.dropna(subset=["sales", model]).groupby("id", observed=True):
        error = (chunk["sales"] - chunk[model]).to_numpy(dtype=float)
        price = float(chunk["price"].mean()) if "price" in chunk.columns else 1.0
        if not np.isfinite(price) or price <= 0:
            price = 1.0
        q_emp = float(np.quantile(error, cr))
        q_norm = float(error.mean() + stats.norm.ppf(cr) * error.std(ddof=1))
        co, cu = overage * price, underage * price
        rows.append(
            {
                "id": series,
                "critical_ratio": cr,
                "price": price,
                "q_empirical": q_emp,
                "q_normal": q_norm,
                "cost_empirical": _cost(error, q_emp, cu, co),
                "cost_normal": _cost(error, q_norm, cu, co),
            }
        )
    detail = pd.DataFrame(rows)
    frontier = []
    for service in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        costs = []
        for series, chunk in preds.dropna(subset=["sales", model]).groupby("id", observed=True):
            error = (chunk["sales"] - chunk[model]).to_numpy(dtype=float)
            price = float(chunk["price"].mean()) if "price" in chunk.columns else 1.0
            if not np.isfinite(price) or price <= 0:
                price = 1.0
            q = float(np.quantile(error, service))
            costs.append(_cost(error, q, underage * price, overage * price))
        frontier.append({"service_level": service, "expected_cost": float(np.mean(costs))})
    return detail, pd.DataFrame(frontier)


def _cost(errors: np.ndarray, order: float, underage_cost: float, overage_cost: float) -> float:
    return float(
        np.mean(
            np.where(
                errors > order,
                underage_cost * (errors - order),
                overage_cost * (order - errors),
            )
        )
    )
