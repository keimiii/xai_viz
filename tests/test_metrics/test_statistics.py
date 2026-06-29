"""Tests for Cohen's d effect size computation.

Verifies both paired (d_z) and unpaired Cohen's d, including edge cases
like zero-variance inputs and sign conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_attention.metrics.statistics import cohens_d, paired_ttest


class TestCohensD:
    """Test cohens_d() for both paired and unpaired variants."""

    # --- Unpaired (standard Cohen's d) ---

    def test_unpaired_identical_arrays_returns_zero(self):
        """Identical arrays should produce d = 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, b, paired=False) == 0.0

    def test_unpaired_known_effect(self):
        """Verify unpaired d against manual calculation.

        a = [2, 4, 6], mean=4, var=4
        b = [1, 3, 5], mean=3, var=4
        pooled_var = (2*4 + 2*4) / 4 = 4, pooled_std = 2
        d = (4 - 3) / 2 = 0.5
        """
        a = np.array([2.0, 4.0, 6.0])
        b = np.array([1.0, 3.0, 5.0])
        d = cohens_d(a, b, paired=False)
        assert d == pytest.approx(0.5)

    # --- Paired (d_z) ---

    def test_paired_identical_arrays_returns_zero(self):
        """Identical paired arrays should produce d_z = 0."""
        a = np.array([0.5, 0.6, 0.7, 0.8])
        b = np.array([0.5, 0.6, 0.7, 0.8])
        assert cohens_d(a, b, paired=True) == 0.0

    def test_paired_known_effect(self):
        """Verify d_z = mean(diff) / std(diff) against manual calculation.

        a = [0.5, 0.7, 0.9], b = [0.3, 0.4, 0.5]
        diff = [0.2, 0.3, 0.4], mean=0.3, std=0.1 (ddof=1)
        d_z = 0.3 / 0.1 = 3.0
        """
        a = np.array([0.5, 0.7, 0.9])
        b = np.array([0.3, 0.4, 0.5])
        d_z = cohens_d(a, b, paired=True)
        assert d_z == pytest.approx(3.0)

    def test_paired_dz_larger_than_unpaired_d(self):
        """Paired d_z should be larger than unpaired d for correlated data.

        When subjects (images) have high between-subject variance but
        consistent within-subject effects, d_z inflates because SD(diff)
        is smaller than pooled SD.
        """
        rng = np.random.default_rng(42)
        # Subjects with high between-subject variance
        baseline = rng.normal(loc=50, scale=10, size=100)
        # Consistent small treatment effect
        a = baseline + rng.normal(loc=0, scale=0.5, size=100)
        b = baseline + 2.0 + rng.normal(loc=0, scale=0.5, size=100)

        d_z = cohens_d(a, b, paired=True)
        d_unpaired = cohens_d(a, b, paired=False)

        # d_z should be substantially larger (the whole point of the fix)
        assert abs(d_z) > abs(d_unpaired)

    # --- Edge cases ---

    def test_zero_variance_paired_returns_zero(self):
        """When all differences are identical, std < 1e-10 → return 0.0."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        assert cohens_d(a, b, paired=True) == 0.0

    def test_zero_variance_unpaired_returns_zero(self):
        """When both arrays are constant and equal, pooled_std < 1e-10 → return 0.0."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b, paired=False) == 0.0

    def test_sign_convention_positive_when_a_greater(self):
        """Effect size should be positive when model A scores higher.

        Uses non-constant differences to avoid the zero-variance guard.
        """
        a = np.array([0.8, 0.9, 1.0, 0.85])
        b = np.array([0.1, 0.3, 0.2, 0.15])
        assert cohens_d(a, b, paired=True) > 0
        assert cohens_d(a, b, paired=False) > 0

    def test_sign_convention_negative_when_b_greater(self):
        """Effect size should be negative when model B scores higher.

        Uses non-constant differences to avoid the zero-variance guard.
        """
        a = np.array([0.1, 0.3, 0.2, 0.15])
        b = np.array([0.8, 0.9, 1.0, 0.85])
        assert cohens_d(a, b, paired=True) < 0
        assert cohens_d(a, b, paired=False) < 0


class TestPairedTTest:
    """Test paired_ttest() edge cases around zero-variance differences."""

    def test_identical_arrays_are_not_significant(self) -> None:
        a = np.array([0.4, 0.6, 0.8, 1.0])
        b = np.array([0.4, 0.6, 0.8, 1.0])

        t_stat, p_value = paired_ttest(a, b)

        assert t_stat == 0.0
        assert p_value == 1.0

    def test_constant_nonzero_shift_returns_zero_p_value(self) -> None:
        a = np.array([1.5, 1.5, 1.5, 1.5])
        b = np.array([1.0, 1.0, 1.0, 1.0])

        t_stat, p_value = paired_ttest(a, b)

        assert t_stat == float("inf")
        assert p_value == 0.0
