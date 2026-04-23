"""
Statistical Analysis for thesis evaluation.

Provides rigorous statistical comparison between systems:
  - Shapiro-Wilk normality test
  - Paired t-test / Wilcoxon signed-rank (depending on normality)
  - Cohen's d effect size
  - Bootstrap confidence intervals

All tests use alpha=0.05 unless otherwise specified.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StatisticalResult:
    """Result of a statistical comparison between two systems."""

    def __init__(
        self,
        metric_name: str,
        system_a: str,
        system_b: str,
        mean_a: float,
        mean_b: float,
        std_a: float,
        std_b: float,
        improvement: float,
        improvement_pct: float,
        p_value: float,
        test_name: str,
        is_normal_a: bool,
        is_normal_b: bool,
        effect_size: float,
        effect_size_label: str,
        ci_lower: float,
        ci_upper: float,
        is_significant: bool,
        n_samples: int,
        meets_thesis_threshold: bool = False,
        p_bh: Optional[float] = None,
        sig_bh: Optional[bool] = None,
        p_holm: Optional[float] = None,
        sig_holm: Optional[bool] = None,
        correction_family_size: Optional[int] = None,
    ):
        self.metric_name = metric_name
        self.system_a = system_a
        self.system_b = system_b
        self.mean_a = mean_a
        self.mean_b = mean_b
        self.std_a = std_a
        self.std_b = std_b
        self.improvement = improvement
        self.improvement_pct = improvement_pct
        self.p_value = p_value
        self.test_name = test_name
        self.is_normal_a = is_normal_a
        self.is_normal_b = is_normal_b
        self.effect_size = effect_size
        self.effect_size_label = effect_size_label
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.is_significant = is_significant
        self.n_samples = n_samples
        # Thesis requires at least medium effect size (|d| >= 0.5)
        self.meets_thesis_threshold = meets_thesis_threshold
        # Multiple-comparison correction (populated by
        # apply_corrections_to_results). None until then.
        self.p_bh = p_bh
        self.sig_bh = sig_bh
        self.p_holm = p_holm
        self.sig_holm = sig_holm
        self.correction_family_size = correction_family_size

    def to_dict(self) -> Dict:
        d = {
            "metric": self.metric_name,
            "system_a": self.system_a,
            "system_b": self.system_b,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "std_a": round(self.std_a, 4),
            "std_b": round(self.std_b, 4),
            "improvement": round(self.improvement, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "p_raw": self.p_value,
            "p_value": round(self.p_value, 6),
            "test": self.test_name,
            "is_normal_a": self.is_normal_a,
            "is_normal_b": self.is_normal_b,
            "effect_size": round(self.effect_size, 4),
            "effect_label": self.effect_size_label,
            "meets_thesis_threshold": self.meets_thesis_threshold,
            "ci_95_lower": round(self.ci_lower, 4),
            "ci_95_upper": round(self.ci_upper, 4),
            "significant": self.is_significant,
            "n": self.n_samples,
        }
        if self.p_bh is not None:
            d["p_bh"] = self.p_bh
            d["sig_bh"] = bool(self.sig_bh)
        if self.p_holm is not None:
            d["p_holm"] = self.p_holm
            d["sig_holm"] = bool(self.sig_holm)
        if self.correction_family_size is not None:
            d["correction_family_size"] = self.correction_family_size
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        sig = "YES" if self.is_significant else "NO"
        thesis = "MET" if self.meets_thesis_threshold else "NOT MET"
        return (
            f"{self.metric_name}: {self.system_b} vs {self.system_a} | "
            f"improvement={self.improvement_pct:+.1f}% | "
            f"p={self.p_value:.4f} ({self.test_name}) | "
            f"effect={self.effect_size:.3f} ({self.effect_size_label}) | "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}] | "
            f"significant={sig} | thesis_threshold={thesis}"
        )


# ============================================================
# Core statistical functions
# ============================================================

def shapiro_wilk_test(data: List[float], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Shapiro-Wilk normality test.

    Returns:
        (is_normal, p_value) - is_normal is True if p > alpha
    """
    from scipy import stats

    arr = np.array(data, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 3:
        logger.warning("Shapiro-Wilk: need >= 3 samples, got %d", len(arr))
        return False, 0.0

    if len(arr) > 5000:
        # Shapiro-Wilk limited to 5000; use Anderson-Darling instead
        result = stats.anderson(arr, dist="norm")
        # Use 5% significance level (index 2)
        is_normal = result.statistic < result.critical_values[2]
        return is_normal, 0.0  # Anderson doesn't give exact p-value

    stat, p_value = stats.shapiro(arr)
    return (p_value > alpha), float(p_value)


def paired_comparison(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Paired statistical test.

    If both distributions are normal: paired t-test
    Otherwise: Wilcoxon signed-rank test

    Args:
        scores_a: Scores for system A (baseline)
        scores_b: Scores for system B (proposed)
        alpha: Significance level

    Returns:
        Dict with test_name, statistic, p_value, is_significant
    """
    from scipy import stats

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return {
            "test_name": "insufficient_data",
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "n": len(a),
        }

    # Check normality of differences
    diff = b - a
    is_normal_diff, _ = shapiro_wilk_test(diff.tolist(), alpha)

    # Also check normality of each distribution
    is_normal_a, _ = shapiro_wilk_test(a.tolist(), alpha)
    is_normal_b, _ = shapiro_wilk_test(b.tolist(), alpha)

    if is_normal_diff and is_normal_a and is_normal_b:
        # Paired t-test
        stat, p_value = stats.ttest_rel(a, b)
        test_name = "paired_t_test"
    else:
        # Wilcoxon signed-rank
        try:
            stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
            test_name = "wilcoxon_signed_rank"
        except ValueError:
            # All differences are zero
            stat, p_value = 0.0, 1.0
            test_name = "wilcoxon_zero_diff"

    return {
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_significant": p_value < alpha,
        "is_normal_a": is_normal_a,
        "is_normal_b": is_normal_b,
        "n": len(a),
    }


def cohens_d(scores_a: List[float], scores_b: List[float]) -> Tuple[float, str]:
    """
    Cohen's d effect size for paired samples.

    Returns:
        (d_value, label) where label is negligible/small/medium/large
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 2:
        return 0.0, "insufficient_data"

    diff = b - a
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        d = 0.0
    else:
        d = mean_diff / std_diff

    # Interpret effect size (Cohen's thresholds)
    abs_d = abs(d)
    if abs_d < 0.2:
        label = "negligible"
    elif abs_d < 0.5:
        label = "small"
    elif abs_d < 0.8:
        label = "medium"
    else:
        label = "large"

    return float(d), label


def apply_multiple_comparison_correction(
    pvalues: List[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply a multiple-comparison correction across a family of p-values.

    Uses statsmodels.stats.multitest.multipletests. Intended to be applied
    globally across ALL comparisons in a family (e.g., 3 systems x 4
    metrics = 12 tests for the exp8 retrieval-metrics analysis), NOT
    per-metric, per audit §15.2 Flag 108 and §16 Module 16 recompute.

    Args:
        pvalues: Raw two-sided p-values from pairwise tests, in any fixed
            order. The order is preserved in the returned arrays, so the
            caller can zip them back onto their originating comparisons.
        method: "fdr_bh" (Benjamini-Hochberg, recommended) or "holm".
        alpha: Significance threshold after correction.

    Returns:
        (p_adjusted, is_significant) as two lists the same length as
        pvalues, in the same order.
    """
    from statsmodels.stats.multitest import multipletests

    if not pvalues:
        return [], []

    reject, p_adjusted, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method=method,
    )
    return list(p_adjusted), list(reject)


def apply_corrections_to_results(
    results: List["StatisticalResult"],
    alpha: float = 0.05,
) -> List["StatisticalResult"]:
    """
    In-place: annotate each StatisticalResult with BH and Holm corrected
    p-values computed across the whole family of results passed in.

    This is the integration point for audit §21.5 T0.8. Callers that
    serialize `results` to JSON after this call will emit each comparison
    with its `p_bh`, `sig_bh`, `p_holm`, `sig_holm` alongside its
    `metric`, `system_a`, `system_b` identifiers — so downstream consumers
    can identify each row without assuming array order.

    Args:
        results: List of StatisticalResult from `run_all_comparisons`
            (or any equivalent pairwise run). Corrections are applied
            across ALL entries in this list as a single family.
        alpha: Post-correction significance threshold.

    Returns:
        The same list (mutated), for convenient chaining.
    """
    if not results:
        return results

    raw = [r.p_value for r in results]
    p_bh, sig_bh = apply_multiple_comparison_correction(
        raw, method="fdr_bh", alpha=alpha
    )
    p_holm, sig_holm = apply_multiple_comparison_correction(
        raw, method="holm", alpha=alpha
    )
    family_size = len(results)
    for r, pb, sb, ph, sh in zip(results, p_bh, sig_bh, p_holm, sig_holm):
        r.p_bh = float(pb)
        r.sig_bh = bool(sb)
        r.p_holm = float(ph)
        r.sig_holm = bool(sh)
        r.correction_family_size = family_size
    return results


def bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for mean difference (B - A).

    Args:
        scores_a: Baseline scores
        scores_b: Proposed scores
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed

    Returns:
        (ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return 0.0, 0.0

    n = len(a)
    diff_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diff_means[i] = np.mean(b[idx] - a[idx])

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(diff_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(diff_means, 100 * (1 - alpha / 2)))

    return ci_lower, ci_upper


# ============================================================
# High-level comparison function
# ============================================================

def compare_systems(
    metric_name: str,
    system_a_name: str,
    system_b_name: str,
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> StatisticalResult:
    """
    Complete statistical comparison between two systems.

    Args:
        metric_name: Name of metric being compared
        system_a_name: Baseline system name
        system_b_name: Proposed system name
        scores_a: Scores for system A (one per query)
        scores_b: Scores for system B (one per query)
        alpha: Significance level
        n_bootstrap: Bootstrap iterations
        seed: Random seed

    Returns:
        StatisticalResult with all tests
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]

    # Descriptive stats
    mean_a = float(np.mean(a_clean)) if len(a_clean) > 0 else 0.0
    mean_b = float(np.mean(b_clean)) if len(b_clean) > 0 else 0.0
    std_a = float(np.std(a_clean, ddof=1)) if len(a_clean) > 1 else 0.0
    std_b = float(np.std(b_clean, ddof=1)) if len(b_clean) > 1 else 0.0

    improvement = mean_b - mean_a
    improvement_pct = (improvement / mean_a * 100) if mean_a != 0 else 0.0

    # Paired test
    test_result = paired_comparison(scores_a, scores_b, alpha)

    # Effect size
    d_value, d_label = cohens_d(scores_a, scores_b)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(
        scores_a, scores_b, n_bootstrap=n_bootstrap, seed=seed
    )

    return StatisticalResult(
        metric_name=metric_name,
        system_a=system_a_name,
        system_b=system_b_name,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        improvement=improvement,
        improvement_pct=improvement_pct,
        p_value=test_result["p_value"],
        test_name=test_result["test_name"],
        is_normal_a=test_result.get("is_normal_a", False),
        is_normal_b=test_result.get("is_normal_b", False),
        effect_size=d_value,
        effect_size_label=d_label,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=test_result["is_significant"],
        n_samples=test_result["n"],
        meets_thesis_threshold=abs(d_value) >= 0.5,
    )


def run_all_comparisons(
    results_by_system: Dict[str, Dict[str, List[float]]],
    baseline_name: str,
    metrics: List[str] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> List[StatisticalResult]:
    """
    Run statistical comparisons of all systems against a baseline.

    Args:
        results_by_system: {system_name: {metric_name: [scores]}}
        baseline_name: Name of baseline system to compare against
        metrics: List of metric names to compare (None = all)
        alpha: Significance level
        seed: Random seed

    Returns:
        List of StatisticalResult objects
    """
    if baseline_name not in results_by_system:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")

    baseline_metrics = results_by_system[baseline_name]
    all_results = []

    for system_name, system_metrics in results_by_system.items():
        if system_name == baseline_name:
            continue

        for metric_name, scores_b in system_metrics.items():
            if metrics and metric_name not in metrics:
                continue

            scores_a = baseline_metrics.get(metric_name, [])
            if not scores_a or not scores_b:
                continue

            # Ensure same length (match by query index)
            min_len = min(len(scores_a), len(scores_b))
            scores_a = scores_a[:min_len]
            scores_b = scores_b[:min_len]

            result = compare_systems(
                metric_name=metric_name,
                system_a_name=baseline_name,
                system_b_name=system_name,
                scores_a=scores_a,
                scores_b=scores_b,
                alpha=alpha,
                seed=seed,
            )
            all_results.append(result)

    # Per audit §21.5 T0.8: apply Benjamini-Hochberg FDR correction across
    # the whole family of comparisons before returning, so downstream
    # reporting has both `p_raw` and `p_bh` per labeled comparison.
    apply_corrections_to_results(all_results, alpha=alpha)

    return all_results


def format_statistical_summary(results: List[StatisticalResult]) -> str:
    """Format statistical results as a readable summary."""
    if not results:
        return "No statistical results to display."

    lines = ["=" * 100]
    lines.append("STATISTICAL ANALYSIS SUMMARY")
    lines.append("=" * 100)

    # Group by metric
    by_metric = {}
    for r in results:
        by_metric.setdefault(r.metric_name, []).append(r)

    for metric, rs in by_metric.items():
        lines.append(f"\n--- {metric} ---")
        for r in rs:
            sig_marker = "*" if r.is_significant else " "
            lines.append(
                f"  {sig_marker} {r.system_b:30s} vs {r.system_a:30s} | "
                f"delta={r.improvement_pct:+6.1f}% | p={r.p_value:.4f} | "
                f"d={r.effect_size:.3f} ({r.effect_size_label})"
            )

    sig_count = sum(1 for r in results if r.is_significant)
    lines.append(f"\n{'='*100}")
    lines.append(f"Total comparisons: {len(results)} | Significant (p<0.05): {sig_count}")
    lines.append(f"* = statistically significant at alpha=0.05")
    return "\n".join(lines)
