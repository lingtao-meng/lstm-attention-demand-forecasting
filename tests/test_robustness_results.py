"""Regression checks for complete robustness evidence and finite calibration."""

import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('robustness_results', ROOT / 'scripts/verify_robustness.py')
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class TestRobustnessResults(unittest.TestCase):
    def test_all_saved_experiments_reconcile(self):
        verify.check(ROOT)

    def test_one_ulp_target_reconstruction_does_not_change_frozen_metrics(self):
        original = verify.walmart_targets

        def one_ulp(*args):
            return tuple(np.nextafter(a, np.full_like(a, np.inf)) for a in original(*args))

        with patch.object(verify, 'walmart_targets', side_effect=one_ulp):
            verify.check(ROOT)

    def test_material_target_reconstruction_change_is_rejected(self):
        original = verify.walmart_targets

        def changed(*args):
            truth, naive, ma4 = original(*args)
            return truth + np.float32(0.001), naive, ma4

        with patch.object(verify, 'walmart_targets', side_effect=changed):
            with self.assertRaises(AssertionError):
                verify.check(ROOT)

    def test_finite_sample_order_statistics(self):
        self.assertEqual(verify.ordered_quantile(list(range(1, 17)), 0.8), 14)
        self.assertEqual(verify.ordered_quantile(list(range(1, 17)), 0.30 / 0.55), 10)

    def test_invalid_calibration_is_rejected(self):
        for values in [[], [1, float('nan')]]:
            with self.assertRaises(ValueError):
                verify.ordered_quantile(values, 0.8)

    def test_corrupted_aggregate_is_rejected(self):
        with self.assertRaises(ValueError):
            verify.near(0.10, 0.11, 'MAE')


if __name__ == '__main__':
    unittest.main()
