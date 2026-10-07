import numpy as np
import pytest
from scipy.stats import norm, shapiro, ttest_rel, wilcoxon

from src.evaluation.study_statistics import paired


def test_normal_differences_select_paired_t_and_wilcoxon_sensitivity():
    d = norm.ppf((np.arange(20) + .5) / 20) + .4
    assert shapiro(d).pvalue > .05
    result = paired(d)
    assert result['primary_test'] == 'paired_t'
    assert result['sensitivity_test'] == 'wilcoxon'
    assert result['p'] == pytest.approx(ttest_rel(d, np.zeros_like(d)).pvalue)
    assert result['sensitivity_p'] == pytest.approx(wilcoxon(d).pvalue)
    assert result['dz'] == pytest.approx(d.mean()/d.std(ddof=1))
    assert result == paired(d)


def test_clearly_nonnormal_differences_select_wilcoxon_and_paired_t_sensitivity():
    d = np.array([1]*18 + [5, 20], dtype=float)
    assert shapiro(d).pvalue < .05
    result = paired(d)
    assert result['primary_test'] == 'wilcoxon'
    assert result['sensitivity_test'] == 'paired_t'
    assert result['p'] == pytest.approx(wilcoxon(d, method='auto').pvalue)
    assert result['sensitivity_p'] == pytest.approx(ttest_rel(d, np.zeros_like(d)).pvalue)


def test_bootstrap_is_paired_seeded_signed_and_reports_undefined_draws():
    d = [0, 0, 1, 2, 3]
    result = paired(d, resamples=1000)
    assert result == paired(d, resamples=1000)
    assert result['undefined_bootstrap_dz'] > 0
    assert result['ci95_dz'] is None
    assert result['dz_interval_status'] == 'undefined_zero_variance_draws'
    assert result['ci95_mean'] is not None and result['dz'] is not None
    negative = paired([-x for x in d], resamples=1000)
    assert negative['dz'] == pytest.approx(-result['dz'])
    assert negative['ci95_mean'] == pytest.approx([-result['ci95_mean'][1], -result['ci95_mean'][0]])


@pytest.mark.parametrize('values', [[], [1], [1, 2]])
def test_less_than_three_pairs_never_selects_a_primary_test(values):
    result = paired(values)
    assert result['status'] == 'insufficient' and result['p'] is None and result['primary_test'] is None


@pytest.mark.parametrize('value', [0, 2])
def test_constant_differences_do_not_invent_normality_or_effect_size(value):
    result = paired([value]*20)
    assert result['status'] == 'constant_difference'
    assert result['shapiro']['p'] is None and result['dz'] is None and result['p'] is None
    assert result['undefined_bootstrap_dz'] == 10000 and result['ci95_dz'] is None
    if value == 0:
        assert result['tests']['wilcoxon']['p'] == 1


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')], [[1, 2], [3, 4]]])
def test_invalid_numeric_data_fail_closed(values):
    with pytest.raises(ValueError):
        paired(values)
