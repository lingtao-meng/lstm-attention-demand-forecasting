"""Hierarchy construction and reconciliation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchy import build_hierarchy, bottom_up, min_t_ols, top_down  # noqa: E402


class TestHierarchy(unittest.TestCase):
    def setUp(self):
        weeks = pd.period_range("2024-01-01", periods=2, freq="W")
        rows = []
        for series, store, state, values in [
            ("A_CA_1", "CA_1", "CA", [1, 2]),
            ("B_CA_1", "CA_1", "CA", [3, 4]),
            ("C_TX_1", "TX_1", "TX", [5, 6]),
        ]:
            for week, value in zip(weeks, values):
                rows.append({"id": series, "store_id": store, "state_id": state, "week": week, "sales": value})
        self.bottom = pd.DataFrame(rows)

    def test_matrix_shape_and_levels(self):
        h = build_hierarchy(self.bottom)
        self.assertEqual(h.matrix.shape, (8, 3))  # total + 2 states + 2 stores + 3 bottom
        self.assertEqual(h.levels.count("bottom"), 3)

    def test_bottom_up_matches_total(self):
        h = build_hierarchy(self.bottom)
        bottom = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        reconciled = bottom_up(h, bottom)
        top = reconciled[0]
        self.assertTrue(np.allclose(top, bottom.sum(axis=0)))

    def test_top_down_is_coherent(self):
        h = build_hierarchy(self.bottom)
        total = np.array([100.0, 200.0])
        reconciled = top_down(h, total)
        self.assertTrue(np.allclose(reconciled[0], total))

    def test_ols_is_coherent(self):
        h = build_hierarchy(self.bottom)
        rng = np.random.default_rng(0)
        base = rng.uniform(1, 10, size=(8, 4))
        reconciled = min_t_ols(h, base)
        self.assertTrue(np.allclose(reconciled[0], reconciled[-3:].sum(axis=0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
