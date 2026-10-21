"""FVA / tracking signal / newsvendor tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from forecast_diagnostics import (  # noqa: E402
    critical_ratio,
    fva,
    forecast_errors,
    newsvendor,
    newsvendor_cost,
    summarise,
    tracking_signal,
)


class TestMetrics(unittest.TestCase):
    def test_fva(self):
        self.assertAlmostEqual(fva(50, 100), 0.5)
        self.assertAlmostEqual(fva(100, 100), 0.0)

    def test_tracking_signal(self):
        self.assertAlmostEqual(tracking_signal(np.array([1.0, 1.0, 1.0, 1.0])), 4.0)
        self.assertAlmostEqual(tracking_signal(np.array([1.0, -1.0])), 0.0)

    def test_critical_ratio(self):
        self.assertAlmostEqual(critical_ratio(0.30, 0.25), 0.30 / 0.55)

    def test_newsvendor_cost(self):
        self.assertAlmostEqual(newsvendor_cost(np.array([1.0]), 0.0, 0.3, 0.25), 0.3)
        self.assertAlmostEqual(newsvendor_cost(np.array([-1.0]), 0.0, 0.3, 0.25), 0.25)
        self.assertAlmostEqual(newsvendor_cost(np.array([0.0]), 0.0, 0.3, 0.25), 0.0)


class TestForecastErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-05", periods=60, freq="W")
        sales = 1000 + np.arange(60) * 5 + rng.normal(0, 30, 60)
        cls.df = pd.DataFrame({"Store": 1, "Date": dates, "Weekly_Sales": sales})
        cls.errors = forecast_errors(cls.df, holdout=5)

    def test_three_models_and_holdout_length(self):
        self.assertEqual(set(self.errors["model"]), {"naive", "seasonal", "ma4"})
        self.assertEqual(len(self.errors), 15)

    def test_summary_has_one_row_per_model(self):
        table = summarise(self.errors)
        self.assertEqual(len(table), 3)
        self.assertIn("fva_vs_best", table.columns)
        self.assertAlmostEqual(table["fva_vs_best"].max(), 0.0)

    def test_newsvendor_returns_detail_and_frontier(self):
        cr, detail, frontier = newsvendor(self.errors, "ma4", 0.379, 0.30, 0.25)
        self.assertAlmostEqual(cr, 0.30 / 0.55)
        self.assertEqual(len(detail), 1)
        self.assertGreater(len(frontier), 0)
        self.assertLess(detail["cost_model"].iloc[0], detail["cost_empirical"].iloc[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
