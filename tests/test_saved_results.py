"""Check that corrupted prediction/order exports cannot pass report checks."""

import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_results", ROOT / "scripts/verify_results.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class SavedResultsTests(unittest.TestCase):
    def test_saved_predictions_reconcile(self):
        verify.build(ROOT)

    def test_negative_order_quantity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data", "results", "m5/results"):
                shutil.copytree(ROOT / name, root / name)
            path = root / "m5/results/quantile_predictions.csv"
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["order_model"] = "-1"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "invalid order quantity"):
                verify.build(root)

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            verify.close(float("inf"), 0.0, "metric")


if __name__ == "__main__":
    unittest.main()
