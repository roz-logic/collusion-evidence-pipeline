"""ABOUTME: Empirical upper-tail calibration for discrete or tied detector scores.

This enforces a finite-sample calibration target, not a population guarantee.
The inclusive comparator used by Oz's evaluation remains unchanged.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
import numpy as np


def empirical_upper_threshold(scores: Sequence[float], target_fpr: float) -> float:
    """Return an inclusive >= cutoff allowing at most floor(n * target_fpr) flags."""
    values=np.asarray(scores,dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 <= target_fpr < 1:
        raise ValueError('Need finite nonempty scores and 0 <= target_fpr < 1')
    allowed=math.floor(len(values)*target_fpr)
    boundary=np.sort(values)[::-1][allowed]
    return float(np.nextafter(boundary,np.inf))
