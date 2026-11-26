"""Periodic-review, lost-sales inventory with scheduled supplier receipts."""

from __future__ import annotations

import numpy as np
from src.probabilistic import conformal_quantile


def protection_offsets(
    actual: np.ndarray, forecast: np.ndarray, protection: int, tau: float,
) -> np.ndarray:
    """Calibrate sum-of-demand residuals from complete historical windows."""
    if actual.ndim != 2 or actual.shape != forecast.shape or not 1 <= protection <= len(actual):
        raise ValueError("Expected paired historical week-by-series matrices and a complete horizon")
    cumulative = np.concatenate((np.zeros((1, actual.shape[1])), np.cumsum(actual, axis=0)))
    totals = cumulative[protection:] - cumulative[:-protection]
    residuals = totals - protection * np.maximum(forecast[:len(totals)], 0)
    return np.apply_along_axis(conformal_quantile, 0, residuals, tau)


def simulate_inventory(
    demand: np.ndarray, target: np.ndarray, initial: np.ndarray, lead: int, case_pack: int = 1,
    *, extra_delay: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Receive, order, fulfill, then record closing stock; delays are indexed by order week."""
    if (demand.ndim != 2 or demand.shape != target.shape or initial.shape != (demand.shape[1],)
            or isinstance(lead, bool) or not isinstance(lead, (int, np.integer)) or lead < 1):
        raise ValueError("Invalid inventory dimensions or lead time")
    for values in (demand, target, initial):
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Demand, target and initial stock must be finite nonnegative whole units")

    if isinstance(case_pack, bool) or not isinstance(case_pack, (int, np.integer)) or case_pack < 1:
        raise ValueError("Case pack must be a positive integer")
    weeks, series = demand.shape
    if extra_delay is None:
        extra_delay = np.zeros(demand.shape, dtype=np.int64)
    else:
        extra_delay = np.asarray(extra_delay)
        if (extra_delay.shape != demand.shape or extra_delay.dtype.kind not in "iu"
                or (extra_delay < 0).any()):
            raise ValueError("Expected one nonnegative integer delay per order week and item")
    receipts = np.zeros((weeks + lead + int(extra_delay.max(initial=0)), series), dtype=np.int64)
    items = np.arange(series)
    columns = ("opening_stock", "arrivals", "pipeline_before_order", "order", "fulfilled",
               "lost", "closing_stock", "closing_pipeline")
    records = {name: np.zeros(demand.shape, dtype=np.int64) for name in columns}
    closing = initial.astype(np.int64)
    on_order = np.zeros(series, dtype=np.int64)
    for step, (need, stock_target) in enumerate(zip(demand, target, strict=True)):
        opening = closing
        arrivals = receipts[step]
        available = opening + arrivals
        open_orders = on_order - arrivals
        shortfall = np.maximum(stock_target.astype(np.int64) - (available + open_orders), 0)
        order = ((shortfall + case_pack - 1) // case_pack) * case_pack
        receipts[step + lead + extra_delay[step], items] += order
        fulfilled = np.minimum(available, need)
        closing = available - fulfilled
        on_order = open_orders + order
        values = (opening, arrivals, open_orders, order, fulfilled, need - fulfilled, closing, on_order)
        for name, value in zip(columns, values, strict=True):
            records[name][step] = value
    return records
