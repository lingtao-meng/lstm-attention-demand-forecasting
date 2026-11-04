"""Sample selection must not depend on later sales or calibration targets."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.m5 import select_series, selection_history  # noqa: E402


class TestTrainingOnlySelection(unittest.TestCase):
    def setUp(self):
        self.sales = pd.DataFrame({"id": [f"I{i}" for i in range(8)], "store_id": "S1",
                                   "d_1": [10, 9, 8, 7, 6, 5, 0, 1], "d_2": [10, 9, 8, 7, 6, 5, 0, 1],
                                   "d_3": [0, 0, 0, 0, 0, 0, 10000, 0], "d_4": [0] * 8})

    def test_future_sales_cannot_change_historical_sample(self):
        before = select_series(self.sales, 6, 42, selection_days=["d_1", "d_2"])
        changed = self.sales.copy()
        changed["d_3"] = np.arange(8) * 100000
        changed["d_4"] = np.arange(8)[::-1] * 100000
        after = select_series(changed, 6, 42, selection_days=["d_1", "d_2"])
        self.assertEqual(before["id"].tolist(), after["id"].tolist())

    def test_item_with_no_selection_history_is_not_eligible(self):
        selected = select_series(self.sales, 6, 42, selection_days=["d_1", "d_2"])
        self.assertNotIn("I6", selected["id"].tolist())
        self.assertEqual(len(selected), 6)

    def test_full_period_mode_retains_legacy_ranking(self):
        self.assertEqual(select_series(self.sales, 5, 42)["id"].iloc[0], "I6")
        self.assertEqual(select_series(self.sales, 5, 42, selection_days=["d_1", "d_2"])["id"].iloc[0], "I0")

    def test_empty_or_unknown_history_is_rejected(self):
        for days in [[], ["d_99"]]:
            with self.assertRaises(ValueError):
                select_series(self.sales, 6, 42, selection_days=days)

    def test_cutoff_excludes_entire_calibration_and_test_blocks(self):
        sales = pd.DataFrame({"id": ["A"], **{f"d_{i}": [1] for i in range(1, 71)}})
        calendar = pd.DataFrame({"d": [f"d_{i}" for i in range(1, 71)],
                                 "date": pd.date_range("2020-01-04", periods=70),
                                 "wm_yr_wk": np.repeat(np.arange(10), 7)})
        days, metadata = selection_history(sales, calendar, holdout=2, gap_weeks=2)
        self.assertEqual(len(days), 42)
        self.assertLess(metadata["selection_end"], metadata["calibration_start"])
        self.assertLess(metadata["calibration_start"], metadata["test_start"])
        repeated, _ = selection_history(sales, calendar, test_start=metadata["test_start"], gap_weeks=2)
        self.assertEqual(days, repeated)


if __name__ == "__main__":
    unittest.main()
