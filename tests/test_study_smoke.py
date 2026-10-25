"""Synthetic verification of the human smoke launcher; never a real smoke or gate."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import study_smoke as smoke
from src.ui.components.session_storage import atomic_json
from src.ui.components.study_protocol import ROOT, draw_study_configuration
from src.ui.components.study_sessions import StudyStore
from tests.test_study_sessions import finish


@pytest.fixture
def package(tmp_path):
    config = tmp_path / 'draw'
    protocol = draw_study_configuration(ROOT / 'config/study.example.json',
        ROOT / 'config/study_assignments.example.csv', config)
    root, backup = tmp_path / 'sessions', tmp_path / 'second-disk'
    store = StudyStore(root, protocol, 'smoke')
    store.freeze()
    session = store.admit(store.issue('P999', cell=1, profile='without_experience'))
    finish(session)
    receipt = dict(smoke.plan(config, root, backup), build_id=session.data['build_id'])
    atomic_json(root / 'smoke_launch.json', receipt)
    return root, backup, config, protocol, json.loads(session.export().read_text(encoding='utf-8'))


def test_plan_is_repeatable_read_only_and_not_a_smoke(package, tmp_path):
    _, backup, config, _, _ = package
    root = tmp_path / 'not-created'
    first = smoke.plan(config, root, backup)
    assert smoke.plan(config, root, backup) == first
    assert not root.exists()
    assert first['real_smoke'] == 'NOT_RUN' and first['physical_disk'] == 'NOT_CHECKED'
    with pytest.raises(ValueError, match='separate'):
        smoke.plan(config, root, root / 'backup')


def test_synthetic_package_verifies_export_and_backup_idempotently(package, monkeypatch):
    root, backup, config, _, row = package
    monkeypatch.setattr(smoke, 'same_physical_disk', lambda *_: False)
    export = next(root.glob('*/full_session.json'))
    before = export.read_bytes()
    result = smoke.verify(root, config, backup)
    assert result['marker'] == 'SMOKE_NOT_GATE'
    assert result['counts'] == dict(tasks=6, free_queries=2, SUS=2, Likert=2, comparative=4, blinding=1,
                                    classes={'answered': 8}, UEQ_S=2)
    assert row['instrument_responses_synthetic'] is True
    assert smoke.verify(root, config, backup) == result
    assert export.read_bytes() == before
    assert (Path(result['backup']['destination']) / 'full_session.json').read_bytes() == before


@pytest.mark.parametrize('bad', ['purpose', 'marker', 'synthetic', 'task', 'free', 'retry', 'decline',
                                'version', 'text', 'sus', 'likert', 'comparative', 'blinding', 'practice'])
def test_incomplete_or_corrupt_smoke_never_passes(package, bad):
    *_, protocol, original = package
    row = copy.deepcopy(original)
    if bad == 'purpose':
        row['purpose'] = 'study'
    elif bad == 'marker':
        row['gate_marker'] = 'GO'
    elif bad == 'synthetic':
        row['instrument_responses_synthetic'] = False
    elif bad == 'task':
        row['attempts'][0]['query_id'] = 'q999'
    elif bad == 'free':
        row['attempts'][3]['analysis_role'] = 'tasks'
    elif bad == 'retry':
        row['attempts'].append(copy.deepcopy(row['attempts'][0]))
    elif bad == 'decline':
        row['attempts'][0]['decline_class'] = 'pure_decline'
    elif bad == 'version':
        row['attempts'][0]['decline_classifier_version'] = 'v1'
    elif bad == 'text':
        row['attempts'][0]['answer'] = ''
    elif bad == 'sus':
        row['instruments'][0]['sus'].pop()
    elif bad == 'likert':
        row['instruments'][0]['likert'].pop('F4')
    elif bad == 'comparative':
        row['comparative'].pop('C4')
    elif bad == 'blinding':
        row['blinding']['choice'] = 'other'
    else:
        row['events'][0]['answer'] = 'PRACTICE_LEAK'
    with pytest.raises(ValueError):
        smoke.validate_export(row, protocol)


def test_drive_letters_and_mount_points_do_not_substitute_for_physical_identity():
    inventory = [dict(DiskNumber=0, AccessPaths=['C:\\', 'D:\\']),
                 dict(DiskNumber=1, AccessPaths=['C:\\mount\\'])]
    assert smoke.disk_number('C:/sessions', inventory) == smoke.disk_number('D:/backup', inventory)
    assert smoke.disk_number('C:/mount/backup', inventory) == 1
    with pytest.raises(ValueError, match='Cannot resolve'):
        smoke.disk_number('Z:/absent', inventory)
    with pytest.raises(ValueError, match='Ambiguous'):
        smoke.disk_number('D:/backup', inventory + [dict(DiskNumber=2, AccessPaths=['D:\\'])])


def test_backup_failure_preserves_export_for_manual_copy_retry(package, monkeypatch):
    root, backup, config, _, _ = package
    export = next(root.glob('*/full_session.json'))
    before = export.read_bytes()
    monkeypatch.setattr(smoke, 'same_physical_disk', lambda *_: True)
    with pytest.raises(ValueError, match='physical disk'):
        smoke.verify(root, config, backup)
    assert export.read_bytes() == before
    assert not (root / 'smoke_verification.json').exists()
    monkeypatch.setattr(smoke, 'same_physical_disk', lambda *_: False)
    assert smoke.verify(root, config, backup)['status'] == 'VERIFIED_EXPORT_AND_BACKUP'


def test_monitor_expires_monotonically_without_starting_any_query(tmp_path):
    ticks = iter([0, 1, 3])
    with pytest.raises(TimeoutError):
        smoke.monitor(SimpleNamespace(poll=lambda: None), tmp_path, 2,
                      clock=lambda: next(ticks), sleep=lambda _: None)
    assert not list(tmp_path.iterdir())


def test_monitor_stops_on_first_error_without_retry(package):
    root, *_ = package
    path = next(root.glob('*/study_checkpoint.json'))
    row = json.loads(path.read_text(encoding='utf-8'))
    row['attempts'][0]['status'] = 'error'
    atomic_json(path, row)
    with pytest.raises(RuntimeError, match='diagnose'):
        smoke.monitor(SimpleNamespace(poll=lambda: None), root, 10)


def test_stop_only_targets_the_owned_live_child(monkeypatch):
    calls = []
    monkeypatch.setattr(smoke.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    smoke.stop_owned(SimpleNamespace(pid=123, poll=lambda: None))
    smoke.stop_owned(SimpleNamespace(pid=456, poll=lambda: 0))
    assert calls == [['taskkill', '/PID', '123', '/T', '/F']]


def test_virtual_disks_are_not_accepted_as_independent_physical_media(monkeypatch):
    records = [dict(DiskNumber=0, AccessPaths=['C:\\'], BusType='NVMe'),
               dict(DiskNumber=1, AccessPaths=['D:\\'], BusType='File Backed Virtual')]
    monkeypatch.setattr(smoke, 'disk_inventory', lambda: records)
    with pytest.raises(ValueError, match='independence'):
        smoke.same_physical_disk('C:/sessions', 'D:/backup')
    records[1]['BusType'] = 'USB'
    assert not smoke.same_physical_disk('C:/sessions', 'D:/backup')


def test_timeout_receipt_cannot_be_relabelled_as_a_pass(package, monkeypatch):
    root, backup, config, *_ = package
    path = root / 'smoke_launch.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    receipt.update(status='FAILED_PRESERVE_EVIDENCE', phase='HUMAN_UI', error='TimeoutError')
    atomic_json(path, receipt)
    monkeypatch.setattr(smoke, 'same_physical_disk', lambda *_: False)
    with pytest.raises(ValueError, match='Terminal'):
        smoke.verify(root, config, backup)


@pytest.mark.parametrize('purpose,pid', [('smoke', 'P999'), ('rehearsal', 'P998')])
def test_operator_cli_can_create_only_the_designated_test_participant(package, tmp_path, capsys, purpose, pid):
    from scripts.manage_study import main

    _, _, config, protocol, _ = package
    root = tmp_path / purpose
    args = ['--config', str(config / 'study.json'), '--assignments', str(config / 'assignments.csv'),
            '--root', str(root), '--purpose', purpose]
    main(args + ['freeze'])
    capsys.readouterr()
    main(args + ['invite', pid, '--cell', '1', '--profile', 'without_experience'])
    token = capsys.readouterr().out.strip()
    store = StudyStore(root, protocol, purpose)
    assert store.admit(token).data['assignment']['participant_id'] == pid
    with pytest.raises(ValueError):
        store.issue('P900', cell=1, profile='without_experience')


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != 'win32', reason='Frozen Windows launcher contract')
def test_launch_wires_real_app_environment_and_cleans_only_owned_child(package, tmp_path, monkeypatch):
    from scripts import gate_job
    from src.utils import deployment_artifacts

    _, backup, config, protocol, _ = package
    root = tmp_path / 'new-human-smoke'
    calls, tokens = [], []
    issue = StudyStore.issue

    def record_token(store, *args, **kwargs):
        token = issue(store, *args, **kwargs)
        tokens.append(token)
        return token

    def popen(command, **kwargs):
        calls.append(('app', command, kwargs['env']))
        return SimpleNamespace(pid=123, poll=lambda: None)

    def monitor(process, target, seconds):
        assert target == root and seconds == 60
        monkeypatch.setenv('CLOUDRAG_BUILD_ID', 'test-build')
        finish(StudyStore(root, protocol, 'smoke').admit(tokens[0]))

    monkeypatch.setattr(StudyStore, 'issue', record_token)
    monkeypatch.setattr(smoke.subprocess, 'check_output',
                        lambda cmd, **kw: b'' if cmd[1] == 'status' else 'test-build')
    monkeypatch.setattr(deployment_artifacts, 'verify_manifest', lambda *_: calls.append('manifest'))
    monkeypatch.setattr(smoke, 'same_physical_disk', lambda *_: False)
    monkeypatch.setattr(gate_job, 'enter', lambda: calls.append('owned-job'))
    monkeypatch.setattr(smoke.subprocess, 'Popen', popen)
    monkeypatch.setattr(smoke, 'monitor', monitor)
    monkeypatch.setattr(smoke, 'stop_owned', lambda p: calls.append(('stop', p.pid)))
    smoke.launch(SimpleNamespace(config_dir=config, root=root, backup=backup, artifact_manifest='trusted.json',
                                model_digest='a' * 64, max_minutes=1, port=8501, operator_checks_confirmed=True))
    app = next(c for c in calls if isinstance(c, tuple) and c[0] == 'app')
    assert '--server.address' in app[1] and '127.0.0.1' in app[1]
    assert app[2]['CLOUDRAG_STUDY_PURPOSE'] == 'smoke'
    assert app[2]['CUDA_VISIBLE_DEVICES'] == ''
    assert app[2]['CLOUDRAG_MODEL_DIGEST'] == 'a' * 64
    assert calls[-1] == ('stop', 123)
    assert json.loads((root / 'smoke_launch.json').read_text())['status'] == 'VERIFIED_EXPORT_AND_BACKUP'
