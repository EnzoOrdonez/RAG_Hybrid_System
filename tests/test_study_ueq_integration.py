import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.study_operator.session_data import clean, export_by_code, inventory
from scripts.study_operator.ueq_source import upgrade_protocol
from src.evaluation.study_analysis import UEQ_FAMILY, analyze, bh
from src.ui.components.session_storage import SessionStorageError, atomic_json
from src.ui.components.study_protocol import LIKERT_IDS, digest, sus_score
from src.ui.components.study_sessions import StudySession, StudyStore
from src.ui.components.study_ueq import score, SOURCE_FILE
from tests.study_helpers import configured
from tests.test_study_analysis import sample
from tests.test_study_sessions import finish, response


def test_example_ueq_binding_matches_git_canonical_bytes_on_both_platforms():
    root=Path(__file__).resolve().parents[1]
    raw=subprocess.check_output(['git','-C',str(root),'hash-object','--no-filters',str(SOURCE_FILE)],timeout=30)
    normalized=subprocess.check_output(['git','-C',str(root),'hash-object',
        '--path=config/UEQS_ES_official.json',str(SOURCE_FILE)],timeout=30)
    exported=SOURCE_FILE.read_bytes()
    reference=json.loads((root/'config/study.example.json').read_bytes())
    assert raw==normalized
    assert b'\r' not in exported
    assert reference['ueq_s_source_sha256']==hashlib.sha256(exported).hexdigest()
    assert reference['ueq_s']==json.loads(exported)


def current_sample(n=20):
    rows = sample(n)
    for i, row in enumerate(rows):
        row['schema_version'] = 4
        for j, block in enumerate(row['instruments']):
            values = [1 + (i + j + k) % 7 for k in range(8)]
            block.update(ueq_s=values, ueq_s_scores=score(values))
    return rows


def test_current_analysis_bh_family_and_descriptive_likert():
    result = analyze(current_sample())
    assert set(result['contrasts']) == {'SUS', *UEQ_FAMILY}
    assert 'p_bh' not in result['contrasts']['SUS']
    assert result['multiplicity']['family'] == list(UEQ_FAMILY)
    expected = bh([result['contrasts'][name]['p'] for name in UEQ_FAMILY])
    assert [result['contrasts'][name]['p_bh'] for name in UEQ_FAMILY] == expected
    assert set(LIKERT_IDS) <= set(result['descriptive']['hybrid'])
    assert 'UEQ_S_overall' not in result['contrasts']
    assert 'UEQ_S_overall' in result['descriptive']['no_rag']
    assert all('p' not in values for values in result['profiles_descriptive'].values())
    assert result == analyze(current_sample())


def test_current_sus_recomputed_and_legacy_never_pooled_or_imputed():
    rows = current_sample(3)
    assert analyze(rows)['contrasts']['SUS']['mean_difference'] == pytest.approx(
        sum(sus_score(row['instruments'][0]['sus'])-50 for row in rows)/3)
    with pytest.raises(ValueError, match='pool'):
        analyze(rows + sample(1))
    legacy = analyze(sample(3))
    assert legacy['not_iteration5_analysis'] is True
    assert legacy['UEQ_S'] == 'NOT_COLLECTED_NO_IMPUTATION'


@pytest.mark.parametrize('fault', ['raw', 'score', 'missing', 'duplicate', 'identity', 'label'])
def test_current_analysis_rejects_corruption(fault):
    rows = current_sample(3)
    block = rows[0]['instruments'][0]
    if fault == 'raw':
        block['ueq_s'][0] = 0
    elif fault == 'score':
        block['ueq_s_scores']['overall'] = 9
    elif fault == 'missing':
        block.pop('ueq_s')
    elif fault == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif fault == 'identity':
        rows[0]['protocol_fingerprint'] = 'altered'
    else:
        block['label'] = 'A'
    with pytest.raises(ValueError):
        analyze(rows)


def test_missing_pair_excluded_and_degenerate_family_never_shrunk():
    rows = current_sample(4)
    for row in rows:
        row['instruments'][0].update(ueq_s=[4]*8, ueq_s_scores=score([4]*8))
        row['instruments'][1].update(ueq_s=[4]*8, ueq_s_scores=score([4]*8))
    rows[0]['instruments'].pop()
    result = analyze(rows)
    assert result['excluded'] == [dict(participant_id='P01', reason='missing_pair')]
    assert all(result['contrasts'][name]['p_bh'] is None for name in UEQ_FAMILY)
    assert len(result['included']) == 3


def test_upgrade_preserves_prior_values_is_idempotent_and_rejects_alteration(tmp_path):
    config, _, _ = configured(tmp_path)
    prior = json.loads(config.read_bytes())
    prior.pop('ueq_s')
    prior.pop('ueq_s_source_sha256')
    prior['schema_version'] = 1
    atomic_json(config, prior)
    new = upgrade_protocol(config)
    assert {key: value for key, value in new.items() if key not in ('ueq_s', 'ueq_s_source_sha256', 'schema_version')} == {
        key: value for key, value in prior.items() if key != 'schema_version'}
    original_bytes = config.read_bytes()
    assert upgrade_protocol(config) == new and config.read_bytes() == original_bytes
    new['ueq_s']['items'][0][0] = 'changed'
    atomic_json(config, new)
    with pytest.raises(ValueError, match='altered'):
        upgrade_protocol(config)


def test_all_eight_capture_reconnect_export_and_withdraw(tmp_path):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, purpose='rehearsal')
    store.freeze()
    atomic_json(store.root / '_i4_root.json', dict(schema_version=1, purpose='rehearsal'))
    token = store.issue('P998', cell=1, profile='without_experience')
    session = store.admit(token)
    session.familiarization_done()
    for _ in range(4):
        session.begin('SYNTHETIC_FREE_QUERY')
        session.finish(answer=response().answer, elapsed_ms=1)
        session.shown()
        session.acknowledge()
    before = session.path.read_bytes()
    for invalid in (None, [4]*7, [4]*7+[None], [4]*7+[8]):
        with pytest.raises(ValueError):
            session.submit_instruments([3]*10, {key: 3 for key in LIKERT_IDS}, invalid)
        assert session.path.read_bytes() == before
        assert session.data['instruments'] == []
    raw = [1, 2, 3, 4, 5, 6, 7, 4]
    session.submit_instruments([3]*10, {key: 3 for key in LIKERT_IDS}, raw)
    restored = StudySession.load(store, session.session_id)
    assert restored.data['instruments'][0]['ueq_s'] == raw
    assert restored.data['instruments'][0]['ueq_s_scores'] == score(raw)
    from tests.test_study_sessions import complete_block
    complete_block(restored)
    restored.submit_comparative(dict(C1='A', C2='iguales', C3='ninguno', C4=''))
    restored.submit_blinding('No sabría decir', '')
    exported = json.loads(restored.export().read_bytes())
    assert exported['schema_version'] == 4
    coded = export_by_code([exported])['pseudonymous_by_code'][0]
    assert coded['participant_code'] == 'P998'
    assert coded['instruments'][0]['ueq_s'] == raw
    assert coded['instruments'][0]['ueq_s_scores'] == score(raw)
    plan = inventory(store.root, code='P998', synthetic_only=True)
    downloaded = []
    for index, row in enumerate(plan['files']):
        source = Path(row['path'])
        copy_path = tmp_path / f'verified-download-{index}'
        copy_path.write_bytes(source.read_bytes())
        assert digest(copy_path) == row['sha256']
        downloaded.append(row)
    assert clean(plan, downloaded)['empty']
    assert not inventory(store.root, code='P998', synthetic_only=True)['files']


def test_closing_session_automatically_attempts_generation_verified_ueq_backup(tmp_path, monkeypatch):
    from scripts import cloud_storage
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, purpose='rehearsal')
    store.freeze()
    session = store.admit(store.issue('P998', cell=1, profile='without_experience'))
    calls = []

    def backup(folder, bucket, prefix):
        data = json.loads((folder / 'full_session.json').read_bytes())
        assert data['stage'] == 'complete'
        assert len(data['instruments']) == 2
        assert all(block['ueq_s_scores'] == score(block['ueq_s']) for block in data['instruments'])
        calls.append((bucket.name, prefix))
        raise OSError('synthetic backup failure')

    monkeypatch.setattr(cloud_storage, 'backup_session', backup)
    monkeypatch.setenv('CLOUDRAG_BACKUP_BUCKET', 'synthetic-private-bucket')
    monkeypatch.setenv('CLOUDRAG_BACKUP_PREFIX', 'synthetic-ueq')
    with pytest.raises(OSError, match='synthetic backup failure'):
        finish(session)
    assert calls == [('synthetic-private-bucket', 'synthetic-ueq')]
    assert session.data['stage'] == 'complete'
    with pytest.raises(SessionStorageError, match='missing'):
        store.issue('P998', cell=1, profile='without_experience')
