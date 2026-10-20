import tempfile
import unittest
from pathlib import Path

from experiments.reproduce_results import reproduce, verify_frozen_inputs
from src.utils.config import RESULTS_TABLES


class ReproductionTests(unittest.TestCase):
    def test_historical_thresholds_and_coverage_reproduce(self):
        with tempfile.TemporaryDirectory() as directory:
            result = reproduce(Path(directory))
        self.assertTrue(result["historical_coverage_and_thresholds_match"])
        self.assertEqual(result["strict_rows_checked"], 15)
        self.assertEqual(result["external_rows_checked"], 20)

    def test_frozen_inputs_pass_hash_checks(self):
        self.assertGreaterEqual(len(verify_frozen_inputs()["files"]), 3)

    def test_reference_directory_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            reproduce(RESULTS_TABLES)
