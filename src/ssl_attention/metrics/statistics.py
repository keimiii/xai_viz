"""Statistical tests for model comparison.

This module provides statistical rigor for comparing 5 SSL models on 139 images:
- Paired tests (t-test, Wilcoxon) for pairwise model comparison
- Bootstrap confidence intervals for robust uncertainty estimation
- Cohen's d for effect size interpretation
- Multiple comparison correction (Holm) for family-wise error control

With 5 models, there are 10 pairwise comparisons. Without correction,
you'd expect ~0.5 false positives at α=0.05. Holm correction controls this.

Example:
    from ssl_attention.metrics import paired_comparison, compare_all_models

    result = paired_comparison(dinov2_ious, clip_ious)
    print(f"DINOv2 vs CLIP: d={result.cohens_d:.2f}, p={result.p_value:.4f}")

    all_results = compare_all_models({
        "dinov2": dinov2_ious,
        "clip": clip_ious,
        "mae": mae_ious,
    })
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch import Tensor


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models.

    Attributes:
        model_a: Name of first model.
        model_b: Name of second model.
        statistic: Test statistic value.
        p_value: Two-sided p-value.
        test_name: Name of statistical test used.
        model_a_mean: Mean score for model A.
        model_b_mean: Mean score for model B.
        difference: Mean difference (A - B).
        cohens_d: Effect size (positive = A > B).
        significant: Whether p < 0.05 after correction.
        corrected_p: P-value after multiple comparison correction.
    """

    model_a: str
    model_b: str
    statistic: float
    p_value: float
    test_name: str
    model_a_mean: float
    model_b_mean: float
    difference: float
    cohens_d: float
    significant: bool
    corrected_p: float | None = None


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Compute paired t-test.

    Uses the formula: t = mean(d) / (std(d) / sqrt(n))
    where d = a - b.

    Args:
        a: Scores for model A.
        b: Scores for model B.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    from scipy import stats

    # Compute differences
    d = a - b
    n = len(d)

    if n < 2:
        return 0.0, 1.0

    # t-statistic
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)

    if std_d < 1e-10:
        # No variance in differences
        if abs(mean_d) < 1e-10:
            return 0.0, 1.0
        return np.inf, 0.0

    t_stat = mean_d / (std_d / np.sqrt(n))

    # Two-sided p-value from t-distribution
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return float(t_stat), float(p_value)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Compute Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        a: Scores for model A.
        b: Scores for model B.

    Returns:
        Tuple of (W_statistic, p_value).
    """
    from scipy import stats

    # Remove zero differences (ties)
    d = a - b
    non_zero_mask = np.abs(d) > 1e-10
    d_nonzero = d[non_zero_mask]

    if len(d_nonzero) < 10:
        # Too few samples for reliable test
        return 0.0, 1.0

    result = stats.wilcoxon(d_nonzero, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray, paired: bool = True) -> float:
    """Compute Cohen's d effect size.

    When paired=True, computes d_z = mean(diff) / SD(diff), which reflects
    the standardized mean difference relative to within-subject variability.
    This tends to be larger than unpaired d for the same raw effect because
    between-subject variance is removed from the denominator.

    When paired=False, computes standard (unpaired) Cohen's d using pooled SD.
    Unpaired benchmarks (Cohen 1988): |d| < 0.2 small, 0.2-0.8 medium, > 0.8 large.

    Note: These benchmarks do NOT apply to paired d_z, which has no widely
    accepted threshold conventions. Compare d_z values across conditions
    rather than interpreting absolute magnitudes.

    Args:
        a: Scores for model A.
        b: Scores for model B.
        paired: Whether samples are paired (same images).

    Returns:
        Effect size (positive = A > B). d_z if paired, Cohen's d if unpaired.
    """
    if paired:
        # Paired Cohen's d uses SD of differences
        d = a - b
        mean_d = np.mean(d)
        std_d = np.std(d, ddof=1)
        if std_d < 1e-10:
            return 0.0
        return float(mean_d / std_d)
    else:
        # Unpaired Cohen's d uses pooled SD
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

        # Pooled standard deviation
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = np.sqrt(pooled_var)

        if pooled_std < 1e-10:
            return 0.0
        return float((mean_a - mean_b) / pooled_std)


def paired_comparison(
    model_a_scores: np.ndarray | list[float] | Tensor,
    model_b_scores: np.ndarray | list[float] | Tensor,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    test: Literal["auto", "ttest", "wilcoxon"] = "auto",
) -> ComparisonResult:
    """Perform paired statistical comparison between two models.

    Args:
        model_a_scores: Per-image scores for model A.
        model_b_scores: Per-image scores for model B.
        model_a_name: Name for model A in results.
        model_b_name: Name for model B in results.
        test: Which test to use:
            - "auto": Use t-test if differences look normal, else Wilcoxon
            - "ttest": Paired t-test
            - "wilcoxon": Wilcoxon signed-rank test

    Returns:
        ComparisonResult with statistics and effect size.
    """
    # Convert to numpy
    if isinstance(model_a_scores, Tensor):
        model_a_scores = model_a_scores.numpy()
    if isinstance(model_b_scores, Tensor):
        model_b_scores = model_b_scores.numpy()
    a = np.asarray(model_a_scores, dtype=np.float64)
    b = np.asarray(model_b_scores, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Score arrays must have same length: {len(a)} != {len(b)}")

    # Choose test
    if test == "auto":
        # Check normality of differences using Shapiro-Wilk
        d = a - b
        if len(d) >= 20:
            from scipy import stats

            _, shapiro_p = stats.shapiro(d)
            test = "ttest" if shapiro_p > 0.05 else "wilcoxon"
        else:
            # Small sample, default to t-test (more power)
            test = "ttest"

    # Run test
    if test == "ttest":
        statistic, p_value = paired_ttest(a, b)
        test_name = "paired t-test"
    else:
        statistic, p_value = wilcoxon_signed_rank(a, b)
        test_name = "Wilcoxon signed-rank"

    # Compute effect size
    effect_size = cohens_d(a, b, paired=True)

    return ComparisonResult(
        model_a=model_a_name,
        model_b=model_b_name,
        statistic=statistic,
        p_value=p_value,
        test_name=test_name,
        model_a_mean=float(np.mean(a)),
        model_b_mean=float(np.mean(b)),
        difference=float(np.mean(a) - np.mean(b)),
        cohens_d=effect_size,
        significant=p_value < 0.05,
    )


def bootstrap_ci(
    scores: np.ndarray | list[float] | Tensor,
    statistic: Literal["mean", "median"] = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        scores: Sample scores.
        statistic: Which statistic to compute ("mean" or "median").
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    # Convert to numpy
    if isinstance(scores, Tensor):
        scores = scores.numpy()
    scores = np.asarray(scores, dtype=np.float64)

    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Set seed
    rng = np.random.default_rng(seed)

    # Select statistic function
    stat_func = np.mean if statistic == "mean" else np.median

    # Point estimate
    point_estimate = float(stat_func(scores))

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)

    # Compute percentile CI
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, lower, upper


def multiple_comparison_correction(
    p_values: list[float],
    method: Literal["holm", "bonferroni", "fdr_bh"] = "holm",
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of raw p-values.
        method: Correction method:
            - "holm": Holm-Bonferroni (step-down, recommended)
            - "bonferroni": Simple Bonferroni (conservative)
            - "fdr_bh": Benjamini-Hochberg FDR (liberal)
        alpha: Significance threshold.

    Returns:
        List of (corrected_p, is_significant) tuples.

    Note:
        With 5 models (10 pairwise comparisons), Holm correction is
        recommended as a balance between Type I and Type II error.
    """
    n = len(p_values)
    if n == 0:
        return []

    p_array = np.array(p_values)

    if method == "bonferroni":
        # Simple: multiply all p-values by n
        corrected = np.minimum(p_array * n, 1.0)
        significant = corrected < alpha
        return [(float(p), bool(s)) for p, s in zip(corrected, significant, strict=True)]

    elif method == "holm":
        # Holm step-down procedure
        # Sort p-values
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Compute Holm correction factors
        correction_factors = n - np.arange(n)
        corrected_sorted = np.minimum(sorted_p * correction_factors, 1.0)

        # Enforce monotonicity (corrected p-values can't decrease)
        for i in range(1, n):
            if corrected_sorted[i] < corrected_sorted[i - 1]:
                corrected_sorted[i] = corrected_sorted[i - 1]

        # Map back to original order
        corrected = np.zeros(n)
        corrected[sorted_indices] = corrected_sorted

        significant = corrected < alpha
        return [(float(p), bool(s)) for p, s in zip(corrected, significant, strict=True)]

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Compute BH correction
        rank = np.arange(1, n + 1)
        corrected_sorted = np.minimum(sorted_p * n / rank, 1.0)

        # Enforce monotonicity (from right to left for BH)
        for i in range(n - 2, -1, -1):
            if corrected_sorted[i] > corrected_sorted[i + 1]:
                corrected_sorted[i] = corrected_sorted[i + 1]

        # Map back to original order
        corrected = np.zeros(n)
        corrected[sorted_indices] = corrected_sorted

        significant = corrected < alpha
        return [(float(p), bool(s)) for p, s in zip(corrected, significant, strict=True)]

    else:
        raise ValueError(f"Unknown correction method: {method}")


def compare_all_models(
    model_scores: dict[str, np.ndarray | list[float] | Tensor],
    test: Literal["auto", "ttest", "wilcoxon"] = "auto",
    correction: Literal["holm", "bonferroni", "fdr_bh"] = "holm",
    alpha: float = 0.05,
) -> dict[tuple[str, str], ComparisonResult]:
    """Perform all pairwise comparisons between models with correction.

    Args:
        model_scores: Dict mapping model name to per-image scores.
        test: Statistical test to use.
        correction: Multiple comparison correction method.
        alpha: Significance threshold.

    Returns:
        Dict mapping (model_a, model_b) tuples to ComparisonResult.
        Results include corrected p-values and adjusted significance.

    Example:
        >>> results = compare_all_models({
        ...     "dinov2": [0.5, 0.6, 0.7],
        ...     "clip": [0.4, 0.5, 0.6],
        ...     "mae": [0.3, 0.4, 0.5],
        ... })
        >>> for (a, b), result in results.items():
        ...     if result.significant:
        ...         print(f"{a} > {b}: d={result.cohens_d:.2f}")
    """
    model_names = sorted(model_scores.keys())
    n_models = len(model_names)

    # Perform all pairwise comparisons
    comparisons: list[ComparisonResult] = []
    pairs: list[tuple[str, str]] = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a, model_b = model_names[i], model_names[j]
            result = paired_comparison(
                model_scores[model_a],
                model_scores[model_b],
                model_a_name=model_a,
                model_b_name=model_b,
                test=test,
            )
            comparisons.append(result)
            pairs.append((model_a, model_b))

    # Apply multiple comparison correction
    raw_p_values = [r.p_value for r in comparisons]
    corrected = multiple_comparison_correction(raw_p_values, method=correction, alpha=alpha)

    # Update results with corrected values
    results: dict[tuple[str, str], ComparisonResult] = {}
    for (model_a, model_b), result, (corrected_p, significant) in zip(
        pairs, comparisons, corrected, strict=True
    ):
        # Create new result with corrected values
        results[(model_a, model_b)] = ComparisonResult(
            model_a=result.model_a,
            model_b=result.model_b,
            statistic=result.statistic,
            p_value=result.p_value,
            test_name=result.test_name,
            model_a_mean=result.model_a_mean,
            model_b_mean=result.model_b_mean,
            difference=result.difference,
            cohens_d=result.cohens_d,
            significant=significant,
            corrected_p=corrected_p,
        )

    return results


def rank_models(
    model_scores: dict[str, np.ndarray | list[float] | Tensor],
) -> list[tuple[str, float, float, float]]:
    """Rank models by mean score with confidence intervals.

    Args:
        model_scores: Dict mapping model name to per-image scores.

    Returns:
        Sorted list of (model_name, mean, ci_lower, ci_upper) tuples,
        sorted by mean score in descending order.
    """
    rankings: list[tuple[str, float, float, float]] = []

    for model_name, scores in model_scores.items():
        mean, lower, upper = bootstrap_ci(scores, statistic="mean")
        rankings.append((model_name, mean, lower, upper))

    # Sort by mean descending
    rankings.sort(key=lambda x: x[1], reverse=True)

    return rankings
