"""Regression checks for time boundaries and calibrated decision scoring."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probabilistic_backtest import (
    CAL_WEEKS,
    FOLDS,
    TEST_WEEKS,
    fold_windows,
    score_fold,
    summarize,
)


class ProbabilisticBacktestTest(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2015-01-03", periods=16, freq="7D")
        actual = np.arange(1, 17)
        self.calibration = pd.DataFrame({
            "id": "known",
            "week_start": dates,
            "actual": actual,
            "naive": 0,
            "price": 2,
            "q10": actual - 2,
            "q50": actual - 1,
            "q90": actual + 1,
        })
        self.test = pd.DataFrame({
            "id": ["known"],
            "week_start": [dates[-1] + pd.Timedelta(weeks=1)],
            "actual": [15],
            "naive": [1],
            "price": [2],
            "q10": [10],
            "q50": [12],
            "q90": [16],
        })

    def test_same_calibration_dates_and_paired_costs(self):
        scored = score_fold(self.calibration, self.test)
        row = scored.iloc[0]
        self.assertEqual(row["order_naive_conformal"], 11)  # 1 + ceil(17 * 0.30/0.55)
        self.assertEqual(row["order_p50_conformal"], 13)
        self.assertEqual((row["lower80"], row["upper80"]), (11, 15))
        result = summarize(scored)
        self.assertAlmostEqual(result["cost_naive_conformal"], 2.4)
        self.assertAlmostEqual(result["cost_p50_conformal"], 1.2)
        self.assertAlmostEqual(result["cost_reduction"], 0.5)
        self.assertAlmostEqual(result["fill_rate_p50_conformal"], 13 / 15)
        self.assertEqual(result["coverage_80"], 1)

    def test_no_scoring_unmatched_series_or_overlapping_periods(self):
        with self.assertRaisesRegex(ValueError, "no pre-test calibration history"):
            score_fold(self.calibration, self.test.assign(id="new"))
        with self.assertRaisesRegex(ValueError, "before the first test week"):
            score_fold(self.calibration, self.test.assign(week_start=self.calibration["week_start"].max()))

    def test_rolls_nonoverlapping_retail_weeks_and_excludes_last_partial_week(self):
        days = 7 * (FOLDS * TEST_WEEKS + CAL_WEEKS + 1) + 2
        dates = pd.date_range("2011-01-29", periods=days, freq="D")
        names = [f"d_{d + 1}" for d in range(days)]
        sales = pd.DataFrame(columns=names)
        calendar = pd.DataFrame({"d": names, "date": dates, "wm_yr_wk": np.arange(days) // 7})
        windows = list(fold_windows(sales, calendar))
        self.assertEqual(len(windows), FOLDS)
        self.assertEqual(windows[0][2][-1], dates[-9])
        for _, cal, test in windows:
            self.assertEqual(len(cal), CAL_WEEKS)
            self.assertEqual(len(test), TEST_WEEKS)
            self.assertEqual(test[0] - cal[-1], pd.Timedelta(weeks=1))
        all_test_dates = [date for _, _, test in windows for date in test]
        self.assertEqual(len(set(all_test_dates)), FOLDS * TEST_WEEKS)


if __name__ == "__main__":
    unittest.main()
