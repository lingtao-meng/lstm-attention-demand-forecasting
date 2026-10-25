"""Weekly aggregation, features and split tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m5 import add_baselines, add_features, time_split, weekly_panel  # noqa: E402


class TestWeeklyPanel(unittest.TestCase):
    def setUp(self):
        self.selected = pd.DataFrame(
            {
                "id": ["A"],
                "item_id": ["I1"],
                "dept_id": ["D1"],
                "cat_id": ["C1"],
                "store_id": ["S1"],
                "state_id": ["CA"],
                **{f"d_{i}": [i] for i in range(1, 15)},
            }
        )
        self.calendar = pd.DataFrame(
            {
                "d": [f"d_{i}" for i in range(1, 15)],
                "date": pd.date_range("2020-01-06", periods=14),
                "wm_yr_wk": [1] * 14,
                "event_name_1": [None] * 14,
                "snap_CA": [1] * 14,
                "snap_TX": [0] * 14,
                "snap_WI": [0] * 14,
            }
        )

    def test_two_weeks_and_sums(self):
        weekly = weekly_panel(self.selected, self.calendar)
        self.assertEqual(len(weekly), 2)
        self.assertEqual(weekly["sales"].tolist(), [28, 77])

    def test_snap_and_event_flags(self):
        weekly = weekly_panel(self.selected, self.calendar)
        self.assertTrue((weekly["snap"] == 1).all())
        self.assertTrue((weekly["event"] == 0).all())


class TestFeaturesAndBaselines(unittest.TestCase):
    def setUp(self):
        weeks = pd.period_range("2020-01-06", periods=6, freq="W")
        self.weekly = pd.DataFrame(
            {
                "id": ["A"] * 6,
                "item_id": ["I1"] * 6,
                "dept_id": ["D1"] * 6,
                "cat_id": ["C1"] * 6,
                "store_id": ["S1"] * 6,
                "state_id": ["CA"] * 6,
                "week": weeks,
                "sales": [10, 12, 11, 15, 14, 16],
                "event": [0, 0, 1, 0, 0, 0],
                "snap": [0, 0, 0, 0, 0, 0],
            }
        )

    def test_lag_and_naive_have_no_leakage(self):
        df = add_baselines(add_features(self.weekly))
        self.assertEqual(df.loc[1, "lag_1"], 10)
        self.assertEqual(df.loc[1, "naive"], 10)
        self.assertAlmostEqual(df.loc[4, "ma4"], np.mean([10, 12, 11, 15]))

    def test_time_split_uses_the_last_weeks(self):
        df = add_baselines(add_features(self.weekly))
        train, test, start = time_split(df, holdout=2)
        self.assertEqual(len(test), 2)
        self.assertLess(train["week"].max(), start)
        self.assertEqual(test["week"].min(), start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
