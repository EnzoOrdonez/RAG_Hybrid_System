import copy
import json

import numpy as np
import pytest

from scripts.manage_study import consolidate, main
from src.evaluation.study_analysis import analyze, bh, outcomes, legacy_paired as paired, read_exports
from src.ui.components.study_protocol import LIKERT_IDS, digest, sus_score
from src.ui.components.study_sessions import StudyStore
from tests.study_helpers import configured
from tests.test_study_sessions import finish


def sample(n=20):
    rows = []
    for i in range(n):
        sus = [4, 2] * 5 if i % 2 else [5, 1] * 5
        blocks = [dict(condition='hybrid', label='B', sus=sus, sus_score=sus_score(sus),
                  likert={k: (2 if k in ('F3', 'R2') else 4) for k in LIKERT_IDS}),
                  dict(condition='no_rag', label='A', sus=[3]*10, sus_score=50, likert={k: 3 for k in LIKERT_IDS})]
        rows.append(dict(schema_version=3, purpose='study', stage='complete', protocol_fingerprint='same',
            labels={'A': 'no_rag', 'B': 'hybrid'}, assignment=dict(participant_id=f'P{i+1:02}', primary_slot=f'P{i+1:02}',
            profile='with_experience' if i % 2 else 'without_experience'), instruments=blocks, attempts=[],
            comparative=dict(C1='B', C2='B', C3='B', C4='synthetic'),
            blinding=dict(choice='Sistema B', reason='synthetic')))
    return rows


def test_known_effect_sign_magnitude_mapping_and_fixed_family():
    result = analyze(sample())
    assert result['contrasts']['SUS']['mean_difference'] == 37.5
    assert 2.8 < result['contrasts']['SUS']['dz'] < 3
    assert result['contrasts']['F']['mean_difference'] == 1
    assert result['contrasts']['U']['mean_difference'] == 1
    assert all(x['significant_bh'] for x in result['contrasts'].values())
    assert result['blinding']['accuracy'] == 1
    assert set(result['descriptive']['hybrid']) == {'R1', 'R2', 'R2_reversed', 'I1'}
    assert result == analyze(sample())


def test_null_constant_missing_pairs_pilot_and_abandonment():
    rows = sample(4)
    for r in rows:
        r['instruments'][0].update(sus=[3]*10, sus_score=50)
    rows[1]['instruments'].pop()
    rows[2]['purpose'] = 'pilot'
    rows[3]['stage'] = 'abandoned'
    result = analyze(rows)
    assert result['included'] == ['P01'] and len(result['excluded']) == 3
    assert result['contrasts']['SUS']['status'] == 'insufficient'
    assert paired([0]*20)['p'] == 1 and paired([0]*20)['dz'] == 0
    assert paired([10]*20)['dz'] is None  # undefined variance is not infinity


def test_seeded_null_calibration_has_no_excess_false_positives():
    rng = np.random.default_rng(42)
    # Regression bound, not a claimed empirical validation of the real study.
    false_positives = sum(paired(rng.normal(size=20), resamples=100)['p'] < .05 for _ in range(100))
    assert false_positives <= 11


@pytest.mark.parametrize('bad', ['duplicate', 'identity', 'score', 'label', 'scale'])
def test_corrupt_or_incompatible_rows_fail_closed(bad):
    rows = sample(3)
    if bad == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif bad == 'identity':
        rows[0]['protocol_fingerprint'] = 'different'
    elif bad == 'score':
        rows[0]['instruments'][0]['sus_score'] = 1
    elif bad == 'label':
        rows[0]['instruments'][0]['label'] = 'A'
    else:
        rows[0]['instruments'][0]['likert']['F1'] = 6
    with pytest.raises(ValueError):
        analyze(rows)


def test_inverse_items_and_bh_values():
    row = sample(1)[0]['instruments'][0]
    assert outcomes(row)['F'] == 4 and outcomes(row)['R2_reversed'] == 4
    assert bh([.01, .04, .03]) == pytest.approx([.03, .04, .04])


def test_operator_freeze_invite_export_and_tamper(tmp_path, capsys):
    c, a, protocol = configured(tmp_path)
    root = tmp_path / 'sessions'
    argv = ['--config', str(c), '--assignments', str(a), '--root', str(root)]
    main(argv + ['freeze'])
    main(argv + ['preflight'])
    assert 'NOT_RUN' in capsys.readouterr().out
    main(argv + ['invite', 'P01'])
    token = capsys.readouterr().out.strip()
    store = StudyStore(root, protocol)
    session = store.admit(token)
    finish(session)
    output = consolidate(store, tmp_path / 'export')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert all(digest(output / name) == sha for name, sha in manifest['files'].items())
    assert json.loads((output / 'analysis.json').read_text())['included'] == ['P01']
    with pytest.raises(ValueError, match='NEW'):
        consolidate(store, output)
    path = session.export()
    path.write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        read_exports([path])
