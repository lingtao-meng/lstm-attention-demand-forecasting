"""Replay fixed M5 inventory targets with alternative whole-case purchasing units."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inventory import simulate_inventory
from inventory_replay import FIXED_TAU, WARMUP, PROFILES


def replay(trace):
    rows = []
    if set(trace["tau"]) != {FIXED_TAU}:
        raise ValueError("Expected fixed-policy trace")
    for (fold, lead, forecast), group in trace.groupby(["fold", "lead", "forecast"]):
        panel = group.set_index(["week_start", "id"]).sort_index()
        def matrix(column):
            return panel[column].unstack().to_numpy()
        demand, target, price = (matrix(col) for col in ("demand", "order_up_to", "price"))
        if demand.shape != (26, 100) or not np.isfinite(price).all():
            raise ValueError("Incomplete test panel")
        mask = ~matrix("warmup")
        expected = np.broadcast_to((np.arange(26) >= WARMUP)[:, None], demand.shape)
        np.testing.assert_array_equal(mask, expected)
        initial = matrix("opening_stock")[0]
        for pack in (1, 6, 12):
            state = simulate_inventory(demand, target, initial, lead, case_pack=pack)
            if pack == 1:
                for key, values in state.items():
                    np.testing.assert_array_equal(values, matrix(key))
            stock, lost = state["closing_stock"][mask], state["lost"][mask]
            holding, shortage = PROFILES["balanced"]
            rows.append(dict(fold=fold, lead=lead, forecast=forecast, case_pack=pack,
                             weeks_items=int(mask.sum()), demand=int(demand[mask].sum()),
                             fulfilled=int(state["fulfilled"][mask].sum()), stock=int(stock.sum()),
                             stockouts=int((lost > 0).sum()),
                             cost=float(np.sum(price[mask]*(holding*stock + shortage*lost)))))
    if len(rows) != 6 * 2 * 2 * 3:
        raise ValueError("Incomplete policy or fold coverage")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    folder = ROOT / "results" / "inventory_replay"
    folds = replay(pd.read_csv(folder / "trace_fixed.csv.gz"))
    folds.to_csv(folder / "case_pack_by_fold.csv", index=False)
    totals = folds.groupby(["lead", "forecast", "case_pack"]).sum(numeric_only=True)
    pooled = pd.DataFrame(dict(fill_rate=totals.fulfilled/totals.demand,
                              mean_stock=totals.stock/totals.weeks_items,
                              stockout_rate=totals.stockouts/totals.weeks_items,
                              mean_cost=totals.cost/totals.weeks_items))
    pooled.to_csv(folder / "case_pack_summary.csv", float_format="%.8f")
