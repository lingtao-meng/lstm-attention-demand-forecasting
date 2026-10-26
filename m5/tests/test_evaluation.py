"""Metric, FVA and newsvendor tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate, newsvendor, per_series, tracking_signal  # noqa: E402


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.preds = pd.DataFrame(
            {
                "id": ["A", "A", "B", "B"],
                "sales": [10.0, 12.0, 5.0, 7.0],
                "naive": [9.0, 11.0, 6.0, 6.0],
                "seasonal": [8.0, 10.0, 4.0, 5.0],
                "ma4": [9.0, 10.0, 5.0, 6.0],
                "lgbm": [10.0, 11.0, 5.0, 7.0],
                "price": [2.0, 2.0, 3.0, 3.0],
            }
        )

    def test_best_model_has_zero_fva_vs_best(self):
        table = evaluate(self.preds)
        self.assertEqual(table.iloc[0]["model"], "lgbm")
        self.assertAlmostEqual(table.iloc[0]["fva_vs_best"], 0.0)
        self.assertAlmostEqual(table.loc[table["model"] == "naive", "fva_vs_naive"].iloc[0], 0.0)

    def test_tracking_signal(self):
        self.assertAlmostEqual(tracking_signal(np.array([1.0, 1.0, 1.0, 1.0])), 4.0)
        self.assertAlmostEqual(tracking_signal(np.array([1.0, -1.0])), 0.0)

    def test_per_series_has_one_row_per_model(self):
        detail = per_series(self.preds)
        self.assertEqual(len(detail), 8)
        self.assertEqual(set(detail["model"]), {"naive", "seasonal", "ma4", "lgbm"})

    def test_newsvendor_detail_and_frontier(self):
        detail, frontier = newsvendor(self.preds, "lgbm", 0.30, 0.25)
        self.assertEqual(len(detail), 2)
        self.assertAlmostEqual(detail["critical_ratio"].iloc[0], 0.30 / 0.55)
        self.assertTrue((detail["cost_empirical"] >= 0).all())
        self.assertEqual(len(frontier), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
