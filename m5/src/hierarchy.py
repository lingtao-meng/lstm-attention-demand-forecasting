"""Hierarchical aggregation and reconciliation for M5.

Node order is: total, states, stores, bottom item-store series.  The summing
matrix S maps the 100 bottom forecasts to all 114 nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Hierarchy:
    labels: list[str]
    levels: list[str]
    bottom_ids: list[str]
    matrix: np.ndarray  # (nodes, bottom)
    shares: np.ndarray  # (bottom,) historical share of total sales


def build_hierarchy(bottom: pd.DataFrame) -> Hierarchy:
    """`bottom` needs id, store_id, state_id, sales, week."""
    total_sales = bottom.groupby("id", observed=True)["sales"].sum()
    store_state = bottom[["store_id", "state_id"]].drop_duplicates().sort_values(["state_id", "store_id"])
    stores = store_state["store_id"].astype(str).tolist()
    states = sorted(bottom["state_id"].astype(str).unique())
    bottom_ids = total_sales.sort_values(ascending=False).index.astype(str).tolist()
    store_of = dict(zip(bottom["id"].astype(str), bottom["store_id"].astype(str)))
    state_of = dict(zip(bottom["id"].astype(str), bottom["state_id"].astype(str)))

    labels = ["total"]
    levels = ["total"]
    rows = [np.ones(len(bottom_ids))]
    for state in states:
        labels.append(f"state:{state}")
        levels.append("state")
        rows.append(np.array([1.0 if state_of[i] == state else 0.0 for i in bottom_ids]))
    for store in stores:
        labels.append(f"store:{store}")
        levels.append("store")
        rows.append(np.array([1.0 if store_of[i] == store else 0.0 for i in bottom_ids]))
    for i in bottom_ids:
        labels.append(f"item:{i}")
        levels.append("bottom")
        row = np.zeros(len(bottom_ids))
        row[bottom_ids.index(i)] = 1.0
        rows.append(row)

    total = float(total_sales.sum())
    shares = np.array([float(total_sales.get(i, 0.0)) / total for i in bottom_ids])
    return Hierarchy(labels, levels, bottom_ids, np.vstack(rows), shares)


def bottom_up(hierarchy: Hierarchy, bottom_forecast: np.ndarray) -> np.ndarray:
    return hierarchy.matrix @ bottom_forecast


def top_down(hierarchy: Hierarchy, total_forecast: np.ndarray) -> np.ndarray:
    bottom = hierarchy.shares[:, None] * total_forecast[None, :]
    return hierarchy.matrix @ bottom


def min_t_ols(hierarchy: Hierarchy, base: np.ndarray) -> np.ndarray:
    """MinT with W = I (OLS reconciliation), the standard simple benchmark."""
    S = hierarchy.matrix
    coef = np.linalg.solve(S.T @ S, S.T @ base)
    return S @ coef


def min_t_wls(hierarchy: Hierarchy, base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """MinT with a diagonal W; weights are inverse error variances per node."""
    S = hierarchy.matrix
    W_inv = np.diag(np.asarray(weights, dtype=float))
    coef = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv @ base)
    return S @ coef
