from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts.study_operator.audit_package import MANIFEST
from scripts.study_operator.evidence import add
from scripts.study_operator.final_report import CRITERIA, SECTIONS
from scripts.study_operator.finalization import admitted, external_paths, finalize


def fixture(tmp_path):
    root = tmp_path/'iteration5-run-test'
    root.mkdir()
    ext = tmp_path/'iteration5-finalization-test'
    ext.mkdir()
    state = dict(iteration=5, agent='Codex', model='fixture-only', status='ACTIVE',
        closure_reserved_utc='2026-10-09T23:01:39Z',
        resources=[dict(type='vm',name='original',id='123',zone='us-central1-a')])
    (root/'STATE.json').write_text(json.dumps(state))
    (root/'COMMANDS.log').write_text('')
    (root/'proof.json').write_text('{"synthetic":true}')
    (root/'command.json').write_text(json.dumps(dict(command=['fixture'],exit_code=0)))
    add(root,[dict(key='fixture-'+str(i),certainty='VERIFICADO',statement='Synthetic fixture '+str(i),
        evidence=['proof.json'],command_receipts=['command.json']) for i in range(7)])
    report = dict(sections={s:[] for s in SECTIONS},
        attributes={s:dict(status='NO_MEDIDO',claims=[]) for s in CRITERIA})
    report['sections']['Resumen ejecutivo'] = ['I5-V'+str(i).zfill(3) for i in range(1,8)]
    report_path = ext/'report-plan.json'
    report_path.write_text(json.dumps(report))
    plan = dict(package=str(root),external=str(ext),report_plan=str(report_path))
    return root,ext,plan


class Cloud:
    def __init__(self):
        self.calls = []
    def command(self, args, **kwargs):
        self.calls.append(args)
        assert args == ['compute','instances','list']
        return [dict(name='original',id='123',zone='zones/us-central1-a',status='TERMINATED')]


def retired(plan, path):
    root = Path(plan['package'])
    assert json.loads((root/'STATE.json').read_bytes())['status'] == 'CLOSING'
    receipt = dict(status='OWN_WRITERS_QUIESCENT_BACKUPS_PRESERVED',remaining_python=0,
                   retired_tasks=['fixture-ordinary'],remaining_safety_tasks=['fixture-backup'])
    (root/'finalization-task-retirement.json').write_text(json.dumps(receipt))
    return receipt


def cleanup(plan, path):
    root, external = Path(plan['package']), Path(plan['external'])
    assert (root/MANIFEST).exists() and (external/'seal.json').exists()
    return dict(status='OWN_SAFETY_TASKS_RETIRED_AFTER_VERIFIED_SEAL',synthetic=True)


def test_reserve_scope_and_actor_are_hard(tmp_path):
    root,ext,plan = fixture(tmp_path)
    cloud = Cloud()
    before = {p.name:p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match='before'):
        finalize(plan,ext/'entry.json',cloud,quiesce=retired,now=datetime(2026,10,8,tzinfo=timezone.utc))
    assert before == {p.name:p.read_bytes() for p in root.iterdir()} and not cloud.calls
    with pytest.raises(ValueError, match='sibling'):
        external_paths(dict(plan,external=str(root/'inside')))
    with pytest.raises(ValueError, match='actor'):
        admitted(dict(iteration=4,closure_reserved_utc='2026-10-09T23:01:39Z'),datetime(2026,10,10,tzinfo=timezone.utc))


def test_retirement_failure_still_closes_cloud_but_never_seals(tmp_path):
    root,ext,plan = fixture(tmp_path)
    cloud = Cloud()
    def failed(*args):
        raise RuntimeError('PRIVATE_CANARY_NOT_FOR_LOGS')
    def no_seal(*args):
        pytest.fail('Unproven quiescence cannot seal')
    with pytest.raises(RuntimeError,match='incomplete'):
        finalize(plan,ext/'entry.json',cloud,quiesce=failed,sealer=no_seal,
            now=datetime(2026,10,10,tzinfo=timezone.utc))
    assert len(cloud.calls) == 2 and not (root/MANIFEST).exists()
    assert 'PRIVATE_CANARY' not in ''.join(p.read_text() for p in root.iterdir() if p.is_file())


def test_synthetic_full_closure_and_backup_real_verifier_preserve_package(tmp_path):
    root,ext,plan = fixture(tmp_path)
    cloud = Cloud()
    outcome = finalize(plan,ext/'entry.json',cloud,quiesce=retired,
        cleanup=cleanup,
        now=datetime(2026,10,10,tzinfo=timezone.utc))
    assert outcome['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED'
    state = json.loads((root/'STATE.json').read_bytes())
    assert state['status'] == 'CLOSED_AWAITING_EXTERNAL_SEAL'
    assert state['scheduled_tasks'] == ['fixture-backup']
    assert state['scheduled_tasks_pending_external_cleanup'] is True
    assert outcome['external_task_cleanup']['synthetic'] is True
    text = (root/'REPORT_FINAL.md').read_text(encoding='utf-8')
    assert text.count('\n## ') == 22 and 'NO_MEDIDO' in text
    assert 'aptitud reservado' in text and 'GO' not in text
    before = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = finalize(plan,ext/'entry.json',None,quiesce=lambda *a:pytest.fail('No replay'),cleanup=cleanup)
    assert result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED'
    assert before == {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_manifest_without_external_pin_cannot_be_repaired(tmp_path):
    root,ext,plan = fixture(tmp_path)
    (root/MANIFEST).write_text('preserved failed seal')
    before = {p.name:p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError,match='no repin'):
        finalize(plan,ext/'entry.json',None)
    assert before == {p.name:p.read_bytes() for p in root.iterdir()}


def test_report_failure_keeps_backup_definitions_and_still_closes_cloud(tmp_path):
    root,ext,plan = fixture(tmp_path)
    Path(plan['report_plan']).write_text('{}')
    cloud = Cloud()
    with pytest.raises(KeyError):
        finalize(plan,ext/'entry.json',cloud,quiesce=retired,
            cleanup=lambda *a:pytest.fail('Never retire backups before verified seal'),
            now=datetime(2026,10,10,tzinfo=timezone.utc))
    receipt = json.loads((root/'finalization-task-retirement.json').read_bytes())
    assert receipt['remaining_safety_tasks'] == ['fixture-backup']
    assert cloud.calls and not (root/MANIFEST).exists()


def test_sealer_failure_keeps_independent_retries(tmp_path):
    root,ext,plan = fixture(tmp_path)

    def failed(*args):
        raise OSError('fixture sealer failure')

    with pytest.raises(OSError,match='sealer failure'):
        finalize(plan,ext/'entry.json',Cloud(),quiesce=retired,sealer=failed,
            cleanup=lambda *a:pytest.fail('Failed seal cannot retire backup'),
            now=datetime(2026,10,10,tzinfo=timezone.utc))
    assert json.loads((root/'STATE.json').read_bytes())['scheduled_tasks'] == ['fixture-backup']
    assert not (root/MANIFEST).exists()


def test_verifier_rejection_cannot_retire_independent_retries(tmp_path):
    root,ext,plan = fixture(tmp_path)
    result = finalize(plan,ext/'entry.json',Cloud(),quiesce=retired,
        sealer=lambda *a:dict(status='EXTERNAL_VERIFICATION_FAILED'),
        cleanup=lambda *a:pytest.fail('Rejected external verification keeps backups'),
        now=datetime(2026,10,10,tzinfo=timezone.utc))
    assert result['status'] == 'EXTERNAL_VERIFICATION_FAILED'
    assert json.loads((root/'STATE.json').read_bytes())['scheduled_tasks'] == ['fixture-backup']


def test_cleanup_failure_is_retried_only_externally_without_cloud_or_package_writes(tmp_path):
    root,ext,plan = fixture(tmp_path)

    def fail(*args):
        raise RuntimeError('fixture external cleanup failure')

    with pytest.raises(RuntimeError,match='external cleanup'):
        finalize(plan,ext/'entry.json',Cloud(),quiesce=retired,cleanup=fail,
            now=datetime(2026,10,10,tzinfo=timezone.utc))
    before = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = finalize(plan,ext/'entry.json',None,
        quiesce=lambda *a:pytest.fail('No cloud or writer replay after seal'),cleanup=cleanup)
    assert result['external_task_cleanup']['synthetic']
    assert before == {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
