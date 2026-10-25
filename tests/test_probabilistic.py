"""Probabilistic metrics and conformal calibration tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probabilistic import (  # noqa: E402
    conformal_adjustment,
    conformal_quantile,
    dm_test,
    interval_coverage,
    interval_width,
    pinball_loss,
)


class TestPinball(unittest.TestCase):
    def test_perfect_forecast_has_zero_loss(self):
        actual = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(pinball_loss(actual, actual, 0.5), 0.0)

    def test_low_quantile_penalises_overprediction(self):
        actual = np.array([0.0])
        self.assertAlmostEqual(pinball_loss(actual, np.array([1.0]), 0.1), 0.9)
        self.assertAlmostEqual(pinball_loss(actual, np.array([1.0]), 0.9), 0.1)


class TestConformal(unittest.TestCase):
    def test_adjustment_is_a_high_quantile(self):
        scores = np.arange(1, 101, dtype=float)
        value = conformal_adjustment(scores, 0.2)
        self.assertGreaterEqual(value, 80)

    def test_conformal_quantile(self):
        residuals = np.arange(1, 11, dtype=float)
        value = conformal_quantile(residuals, 0.5)
        self.assertGreaterEqual(value, 5)
        self.assertLessEqual(value, 10)


class TestIntervals(unittest.TestCase):
    def test_coverage_and_width(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([2.0, 2.0, 2.0, 2.0])
        self.assertAlmostEqual(interval_coverage(actual, lower, upper), 0.5)
        self.assertAlmostEqual(interval_width(lower, upper), 2.0)


class TestDM(unittest.TestCase):
    def test_identical_errors_are_not_significant(self):
        errors = np.array([1.0, -2.0, 3.0, -1.0])
        _, p_value = dm_test(errors, errors)
        self.assertEqual(p_value, 1.0)

    def test_better_model_has_positive_statistic(self):
        rng = np.random.default_rng(0)
        good = rng.normal(0, 1, 200)
        bad = rng.normal(0, 3, 200)
        statistic, p_value = dm_test(bad, good)
        self.assertGreater(statistic, 0)
        self.assertLess(p_value, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
