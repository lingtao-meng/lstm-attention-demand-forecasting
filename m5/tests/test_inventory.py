"""Check that the replay obeys physical balances and time boundaries."""

import unittest

import numpy as np
from src.inventory import protection_offsets, simulate_inventory


class InventoryTest(unittest.TestCase):
    def test_case_pack_rounds_only_positive_shortfalls(self):
        demand = np.array([[0], [2], [1], [0]])
        target = np.array([[7], [7], [9], [10]])
        result = simulate_inventory(demand, target, np.array([1]), lead=2, case_pack=6)
        np.testing.assert_array_equal(result["order"].ravel(), [6, 0, 6, 0])
        np.testing.assert_array_equal(result["arrivals"].ravel(), [0, 0, 6, 0])
        for invalid in (0, -2, 1.5, True):
            with self.assertRaises(ValueError):
                simulate_inventory(demand, target, np.array([1]), 2, invalid)

    def test_lead_two_does_not_count_open_orders_as_available_stock(self):
        actual = np.full((4, 1), 5, dtype=int)
        target = np.full((4, 1), 10, dtype=int)
        result = simulate_inventory(actual, target, np.array([4]), lead=2)
        expected = {
            "opening_stock": [4, 0, 0, 1], "arrivals": [0, 0, 6, 4],
            "pipeline_before_order": [0, 6, 4, 0], "order": [6, 4, 0, 5],
            "fulfilled": [4, 0, 5, 5], "lost": [1, 5, 0, 0], "closing_stock": [0, 0, 1, 0],
            "closing_pipeline": [6, 10, 4, 5],
        }
        for key, values in expected.items():
            np.testing.assert_array_equal(result[key].ravel(), values)

    def test_order_precedes_current_week_demand(self):
        actual = np.array([[1], [1], [1], [9]])
        target = np.array([[3], [3], [3], [3]])
        a = simulate_inventory(actual, target, np.array([2]), lead=1)
        b = simulate_inventory(np.ones_like(actual), target, np.array([2]), lead=1)
        np.testing.assert_array_equal(a["order"], b["order"])
        np.testing.assert_array_equal(a["closing_stock"][:-1], b["closing_stock"][:-1])
        self.assertGreater(a["lost"][-1, 0], b["lost"][-1, 0])

    def test_offsets_calibrate_only_complete_protection_windows(self):
        actual = np.array([[2], [3], [4], [5], [6]])
        forecast = np.array([[1], [2], [10], [100], [999]])
        # Three full windows at L+1=3. Residuals [6, 6, -15] have median 6.
        np.testing.assert_array_equal(protection_offsets(actual, forecast, 3, 0.5), [6])
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            protection_offsets(actual, forecast, 6, 0.5)


if __name__ == "__main__":
    unittest.main()
