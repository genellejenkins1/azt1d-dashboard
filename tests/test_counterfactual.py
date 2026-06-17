"""Unit tests for counterfactual metric utilities.

These tests focus on purely algebraic properties of the metric definitions.
We use small synthetic glucose traces where the expected TIR/TBR/TAR values
can be verified by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src.counterfactual import (
    DEFAULT_LOWER,
    DEFAULT_UPPER,
    RangeMetrics,
    apply_uniform_offset,
    compute_offset_counterfactual_metrics,
    compute_range_metrics,
)


class TestRangeMetrics:
    """Tests for compute_range_metrics on small synthetic examples."""

    def test_simple_three_point_trace(self):
        """Single-point contributions to TIR/TBR/TAR are computed correctly.

        Trace: [60, 100, 190] mg/dL
        Default range: [70, 180] mg/dL

        - 60  < 70  -> below range
        - 100 in    -> in range
        - 190 > 180 -> above range
        """

        g = pd.Series([60.0, 100.0, 190.0])
        metrics = compute_range_metrics(g, lower=70.0, upper=180.0)

        assert isinstance(metrics, RangeMetrics)
        assert metrics.mean_glucose == pytest.approx((60 + 100 + 190) / 3.0)
        assert metrics.tir == pytest.approx(100.0 * (1.0 / 3.0))
        assert metrics.tbr == pytest.approx(100.0 * (1.0 / 3.0))
        assert metrics.tar == pytest.approx(100.0 * (1.0 / 3.0))

    def test_ignores_nans_and_requires_some_data(self):
        """NaNs are dropped; all-NaN input raises a ValueError."""

        g = pd.Series([np.nan, 120.0, 150.0])
        metrics = compute_range_metrics(g, lower=70.0, upper=180.0)
        assert metrics.tir == pytest.approx(100.0)  # both finite values in range

        g_all_nan = pd.Series([np.nan, np.nan])
        with pytest.raises(ValueError):
            compute_range_metrics(g_all_nan, lower=70.0, upper=180.0)

    def test_invalid_bounds_raise(self):
        """Non-physiological or inverted bounds are rejected."""

        g = pd.Series([100.0, 110.0, 120.0])

        with pytest.raises(ValueError):
            compute_range_metrics(g, lower=10.0, upper=180.0)
        with pytest.raises(ValueError):
            compute_range_metrics(g, lower=80.0, upper=70.0)


class TestOffsetCounterfactual:
    """Tests for CGM offset-based counterfactual metrics."""

    def test_apply_uniform_offset_is_affine(self):
        """Offset operation is additive and preserves NaN structure."""

        g = pd.Series([np.nan, 100.0, 150.0])
        shifted = apply_uniform_offset(g, offset_mgdl=10.0)

        assert np.isnan(shifted.iloc[0])
        assert shifted.iloc[1] == pytest.approx(110.0)
        assert shifted.iloc[2] == pytest.approx(160.0)

    def test_offset_changes_range_membership_as_expected(self):
        """A positive offset moves some points from below-range into range."""

        base = pd.Series([60.0, 100.0, 190.0])

        # Baseline metrics under 70-180 mg/dL
        base_metrics = compute_range_metrics(base, lower=70.0, upper=180.0)

        # Apply +10 mg/dL offset: [70, 110, 200]
        cf_metrics = compute_offset_counterfactual_metrics(
            base,
            offset_mgdl=10.0,
            lower=70.0,
            upper=180.0,
        )

        # After offset, 70 and 110 are in range, 200 is above
        assert cf_metrics.tir == pytest.approx(100.0 * (2.0 / 3.0))
        assert cf_metrics.tbr == pytest.approx(0.0)
        assert cf_metrics.tar == pytest.approx(100.0 * (1.0 / 3.0))

        # Sanity check: mean glucose increases by exactly 10 mg/dL
        assert cf_metrics.mean_glucose == pytest.approx(
            base_metrics.mean_glucose + 10.0
        )