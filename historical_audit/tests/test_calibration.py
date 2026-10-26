"""ABOUTME: Test ordinal calibration boundaries and the old experiment's compatibility."""
from __future__ import annotations
import unittest
import numpy as np
from historical_audit.calibration import empirical_upper_threshold

class OrdinalCalibrationTests(unittest.TestCase):
    def test_all_zero_nulls_are_not_flagged(self) -> None:
        threshold=empirical_upper_threshold([0.0]*10,0.1)
        self.assertEqual(sum(x>=threshold for x in [0.0]*10),0)
        self.assertTrue(1>=threshold)

    def test_ties_never_exceed_calibration_target(self) -> None:
        for values in ([0,0,0,1],[0,0,1,1],[0,1,1,1],[0]*9+[1],[0]*8+[1,1]):
            threshold=empirical_upper_threshold(values,0.1)
            self.assertLessEqual(sum(x>=threshold for x in values),int(len(values)*0.1))

    def test_inadequate_scores_are_rejected(self) -> None:
        for values in ([],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):empirical_upper_threshold(values,0.1)

if __name__=='__main__':unittest.main()
