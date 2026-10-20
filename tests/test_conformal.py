import math
import unittest

import numpy as np

from src.conformal.efficiency import joint_interval_efficiency
from src.conformal.joint_conformal import calibrate_joint_split_conformal, joint_covered
from src.conformal.mondrian_conformal import bounded_interval
from src.conformal.split_conformal import calibrate_split_conformal, predict_interval
from src.data_prep.group_split import assert_no_leakage, sample_one_variant_per_cell, split_by_original_cell


class ConformalTests(unittest.TestCase):
    def test_small_calibration_requires_unbounded_interval(self):
        threshold = calibrate_split_conformal(np.arange(5), alpha=0.1)
        self.assertTrue(math.isinf(threshold))
        self.assertEqual(predict_interval(0.5, threshold), (-math.inf, math.inf))

    def test_small_sample_bounded_domain_is_full_support(self):
        self.assertEqual(bounded_interval("circularity", 0.7, math.inf), (0.0, 1.0))
        self.assertEqual(bounded_interval("nc_ratio", 0.7, math.inf), (0.0, math.inf))

    def test_exact_order_statistic(self):
        self.assertEqual(calibrate_split_conformal(np.arange(1, 11), alpha=0.2), 9.0)

    def test_boundary_sample_size(self):
        self.assertEqual(calibrate_split_conformal(np.arange(9), alpha=0.1), 8.0)

    def test_cqr_negative_scores_are_preserved(self):
        self.assertEqual(calibrate_split_conformal(np.arange(-10, 0), alpha=0.2), -2.0)

    def test_nonfinite_scores_use_valid_subset(self):
        self.assertEqual(calibrate_split_conformal(np.r_[np.arange(10), np.nan, np.inf], alpha=0.2), 8.0)

    def test_empty_scores_and_invalid_alpha(self):
        for scores, alpha in [([], .1), ([np.nan], .1), ([1], 0), ([1], 1), ([1], math.nan)]:
            with self.subTest(scores=scores, alpha=alpha), self.assertRaises(ValueError):
                calibrate_split_conformal(scores, alpha)

    def test_invalid_interval_arguments(self):
        for point, width in [(np.nan, 1), (np.inf, 1), (0, np.nan), (0, -1), (0, -np.inf)]:
            with self.subTest(point=point, width=width), self.assertRaises(ValueError):
                predict_interval(point, width)

    def test_joint_failures_are_not_covered(self):
        calibration = {"scales": {"a": 1., "b": 1.}, "q_hat_joint": math.inf}
        np.testing.assert_array_equal(joint_covered({"a": np.array([0., np.nan]), "b": np.array([1., 1.])}, calibration), [True, False])

    def test_joint_calibration_small_sample(self):
        c = calibrate_joint_split_conformal({"a": np.arange(5.), "b": np.arange(5.)}, .1, {"a": 1., "b": 2.})
        self.assertTrue(math.isinf(c["q_hat_joint"]))

    def test_joint_efficiency_and_failure_denominator(self):
        rows = [
            {"pred_nc_ratio": 1., "gt_nc_ratio": 1., "pred_circularity": .5, "gt_circularity": .5},
            {"pred_nc_ratio": 1., "gt_nc_ratio": 1., "pred_circularity": math.nan, "gt_circularity": .5},
        ]
        metrics = joint_interval_efficiency(rows, {"q_hat_joint": 1., "scales": {"nc_ratio": .5, "circularity": .5}})
        self.assertEqual(metrics["n_joint_interval_rows"], 1)
        self.assertEqual(metrics["joint_mean_width_nc_ratio"], 1.)
        self.assertEqual(metrics["joint_mean_width_circularity"], 1.)
        self.assertEqual(metrics["joint_full_domain_rate_circularity"], 1.)
        self.assertEqual(metrics["joint_mean_rectangle_area"], 1.)

    def test_efficiency_empty_has_no_false_zero_width(self):
        metrics = joint_interval_efficiency([], {"q_hat_joint": 1., "scales": {"nc_ratio": 1., "circularity": 1.}})
        self.assertIsNone(metrics["joint_mean_width_circularity"])

    def test_original_cells_remain_disjoint_and_complete(self):
        cells = [str(i) for i in range(917)]
        split = split_by_original_cell(cells)
        assert_no_leakage(split["train"], split["calibration"], split["test"])
        self.assertEqual(set().union(*map(set, split.values())), set(cells))
        self.assertEqual(split, split_by_original_cell(cells))

    def test_duplicate_cell_and_leakage_are_rejected(self):
        with self.assertRaises(ValueError):
            split_by_original_cell(["x", "x"])
        with self.assertRaises(AssertionError):
            assert_no_leakage(["x"], ["x"], [])

    def test_variant_sampling_is_order_independent(self):
        first = sample_one_variant_per_cell({"b": ["2", "1"], "a": ["2", "1"]}, 42)
        second = sample_one_variant_per_cell({"a": ["1", "2"], "b": ["1", "2"]}, 42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b"})
