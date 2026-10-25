"""Iteration-5 preregistered paired tests and participant-pair bootstrap.

Differences are hybrid minus no_rag. Shapiro alpha .05 selects the primary
two-sided paired test; the other test is sensitivity. Degenerate differences do
not become infinite effects or invented normality evidence.
"""
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon


def paired(differences, *, seed=42, resamples=10000):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError('Finite one-dimensional paired differences required')
    if type(resamples) is not int or resamples < 1:
        raise ValueError('Positive bootstrap resample count required')
    n = len(d)
    result = dict(n=n, status='insufficient' if n < 3 else 'ok',
                  primary_test=None, p=None, sensitivity_test=None, sensitivity_p=None,
                  shapiro=dict(alpha=.05, statistic=None, p=None, normal=None),
                  tests=dict(paired_t=None, wilcoxon=None), mean_difference=float(d.mean()) if n else None,
                  dz=None, ci95_mean=None, ci95_dz=None, undefined_bootstrap_dz=resamples)
    if n < 2:
        return result
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    samples = d[np.random.default_rng(seed).integers(0, n, size=(resamples, n))]
    means, sds = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    defined = sds > 0
    # Dropping zero-variance draws would give a conditional, mislabeled CI.
    effects = means / sds if defined.all() else None
    result.update(dz=mean / sd if sd else None,
                  ci95_mean=np.percentile(means, [2.5, 97.5]).tolist(),
                  ci95_dz=np.percentile(effects, [2.5, 97.5]).tolist() if effects is not None else None,
                  undefined_bootstrap_dz=int((~defined).sum()),
                  dz_interval_status='defined' if effects is not None else 'undefined_zero_variance_draws')
    if n < 3:
        return result
    w = 1.0 if not np.any(d) else float(wilcoxon(d, alternative='two-sided', zero_method='wilcox', method='auto').pvalue)
    result['tests']['wilcoxon'] = dict(p=w)
    if not sd:
        result.update(status='constant_difference', normality_not_defined=True)
        return result
    sw = shapiro(d)
    normal = bool(sw.pvalue >= .05)
    t = ttest_rel(d, np.zeros_like(d), alternative='two-sided')
    result['shapiro'].update(statistic=float(sw.statistic), p=float(sw.pvalue), normal=normal)
    result['tests']['paired_t'] = dict(statistic=float(t.statistic), p=float(t.pvalue), df=n-1)
    primary, sensitivity = ('paired_t', 'wilcoxon') if normal else ('wilcoxon', 'paired_t')
    result.update(primary_test=primary, p=result['tests'][primary]['p'],
                  sensitivity_test=sensitivity, sensitivity_p=result['tests'][sensitivity]['p'])
    return result
