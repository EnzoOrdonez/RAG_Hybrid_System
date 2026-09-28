"""Paired study analysis; no models, historical experiment changes or imputation."""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.ui.components.session_storage import read_json
from src.ui.components.study_protocol import LIKERT_IDS, digest, scores, sus_score


def read_exports(paths):
    records, hashes = [], {}
    for path in map(Path, paths):
        manifest = read_json(path.parent / 'export_manifest.json')
        actual = digest(path)
        if path.name != 'full_session.json' or manifest.get('files') != {path.name: actual}:
            raise ValueError('Participant export integrity mismatch')
        records.append(read_json(path))
        hashes[str(path.resolve())] = actual
    return records, hashes


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(1, adjusted)
    return result.tolist()


def paired(differences, *, seed=42, resamples=10000):
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if not np.isfinite(d).all():
        raise ValueError('Non-finite outcome')
    if n < 2:
        return dict(n=n, p=1.0, mean_difference=None, dz=None, ci95_mean=None,
                    ci95_dz=None, undefined_bootstrap_dz=resamples, status='insufficient')
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    p = 1.0 if not np.any(d) else float(wilcoxon(d, alternative='two-sided', zero_method='wilcox', method='auto').pvalue)
    samples = d[np.random.default_rng(seed).integers(0, n, size=(resamples, n))]
    means, sds = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    finite = sds > 0
    effects = means[finite] / sds[finite]
    return dict(n=n, p=p, mean_difference=mean, dz=mean / sd if sd else (0.0 if mean == 0 else None),
        ci95_mean=np.percentile(means, [2.5, 97.5]).tolist(),
        ci95_dz=np.percentile(effects, [2.5, 97.5]).tolist() if len(effects) else None,
        undefined_bootstrap_dz=int((~finite).sum()), status='constant_difference' if not sd else 'ok')


def outcomes(block):
    if set(block['likert']) != set(LIKERT_IDS):
        raise ValueError('Incomplete Likert block')
    scores(list(block['likert'].values()), 10)
    sus = sus_score(block['sus'])
    if sus != block['sus_score']:
        raise ValueError('SUS score does not match raw items')
    x = block['likert']
    return dict(SUS=sus, F=(x['F1'] + x['F2'] + 6 - x['F3'] + x['F4']) / 4,
                U=(x['U1'] + x['U2'] + x['U3']) / 3,
                R1=x['R1'], R2=x['R2'], R2_reversed=6-x['R2'], I1=x['I1'])


def analyze(records):
    """Primary contrast hybrid minus no_rag; fixed BH family SUS/F/U.

    Profiles and R/I are descriptive. Pilots, abandoned and missing pairs never
    become synthetic observations. Technical retry errors do not delete an
    otherwise complete participant. Bootstrap unit is the participant pair.
    """
    included, excluded, seen, slots, identity = [], [], set(), set(), None
    errors = 0
    for row in records:
        if row.get('schema_version') != 3:
            raise ValueError('Expected study schema 3')
        pid = row['assignment']['participant_id']
        if row['purpose'] != 'study' or row['stage'] != 'complete':
            excluded.append(dict(participant_id=pid, reason='non_study' if row['purpose'] != 'study' else row['stage']))
            continue
        if pid in seen or row['assignment']['primary_slot'] in slots:
            raise ValueError('Duplicate participant/primary slot')
        seen.add(pid)
        slots.add(row['assignment']['primary_slot'])
        current = (row['protocol_fingerprint'], row['labels'])
        if identity is not None and current != identity:
            raise ValueError('Do not pool different frozen study configurations')
        identity = current
        blocks = row['instruments']
        if len(blocks) != 2 or {b['condition'] for b in blocks} != {'hybrid', 'no_rag'}:
            excluded.append(dict(participant_id=pid, reason='missing_pair'))
            continue
        if any(row['labels'].get(b['label']) != b['condition'] for b in blocks):
            raise ValueError('Block label differs from frozen mapping')
        values = {b['condition']: outcomes(b) for b in blocks}
        errors += sum(a['status'] == 'error' for a in row['attempts'])
        included.append((row, values))
    contrasts = {name: paired([v['hybrid'][name] - v['no_rag'][name] for _, v in included]) for name in ('SUS', 'F', 'U')}
    for result, adjusted in zip(contrasts.values(), bh([r['p'] for r in contrasts.values()]), strict=True):
        result.update(p_bh=adjusted, significant_bh=adjusted < .05 and result['n'] >= 2)
    descriptive = {condition: {key: [v[condition][key] for _, v in included]
                   for key in ('R1', 'R2', 'R2_reversed', 'I1')} for condition in ('hybrid', 'no_rag')}
    profiles = {profile: {name: [v['hybrid'][name] - v['no_rag'][name] for r, v in included
                 if r['assignment']['profile'] == profile] for name in contrasts}
                 for profile in ('without_experience', 'with_experience')}
    blind = Counter()
    for r, _ in included:
        choice = r['blinding']['choice']
        correct = next('Sistema ' + label for label, condition in r['labels'].items() if condition == 'hybrid')
        blind['correct' if choice == correct else 'unsure' if choice == 'No sabría decir' else 'incorrect'] += 1
    return dict(schema_version=1, certainty='computed_from_supplied_exports', contrast='hybrid minus no_rag',
        bootstrap=dict(seed=42, resamples=10000, unit='participant_pair', interval='percentile_95'),
        included=[r['assignment']['participant_id'] for r, _ in included], excluded=excluded,
        contrasts=contrasts, descriptive=descriptive, profiles_descriptive=profiles,
        blinding=dict(blind, denominator=len(included), accuracy=blind['correct']/len(included) if included else None),
        comparative={key: dict(Counter(r['comparative'][key] for r, _ in included)) for key in ('C1', 'C2', 'C3')},
        qualitative=[dict(participant_id=r['assignment']['participant_id'], comparative=r['comparative']['C4'],
            blinding_reason=r['blinding']['reason'], free_queries=[a for a in r['attempts'] if a['analysis_role'] == 'free_query'])
            for r, _ in included], technical_errors_in_included_sessions=errors)
