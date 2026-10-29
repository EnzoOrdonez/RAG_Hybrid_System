"""Cost-free SDK admission: local components only, never a provider call."""
import copy
from datetime import datetime, timezone
import json
import subprocess

import pytest

from tests.test_study_i7_ledger import human_operator


class Fixed(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 1, tzinfo=timezone.utc)


def test_reserve_missing_beta_blocks_before_admit_paid(tmp_path, monkeypatch):
    op, cloud = human_operator(tmp_path)
    before = copy.deepcopy(op.state)
    def missing():
        raise ValueError('GCLOUD_BETA_REQUIRED')
    monkeypatch.setattr(cloud, 'require_beta', missing, raising=False)
    monkeypatch.setattr(op, 'admit_paid', lambda *a, **kw: pytest.fail('admit_paid before SDK preflight'))
    with pytest.raises(ValueError, match='^GCLOUD_BETA_REQUIRED$'):
        op.reserve(4)
    assert cloud.calls == []
    assert op.state == before


def test_reserve_with_beta_checks_before_admission(tmp_path, monkeypatch):
    op, cloud = human_operator(tmp_path)
    events = []
    admit = op.admit_paid
    def checked():
        events.append('beta')
    def admitted(*args, **kwargs):
        events.append('admit_paid')
        return admit(*args, **kwargs)
    monkeypatch.setattr(cloud, 'require_beta', checked)
    monkeypatch.setattr(op, 'admit_paid', admitted)
    assert op.reserve(4)['status'] == 'SPECIFIC_RESERVATION_HELD'
    assert events == ['beta', 'admit_paid']


def test_cloud_beta_query_is_local_and_creates_no_operation_receipts(tmp_path):
    from scripts.study_operator.cloud_client import Cloud
    calls = []
    def invoke(argv, **kwargs):
        calls.append(argv)
        assert kwargs['timeout'] == 15
        assert kwargs['env']['CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK'] == 'true'
        assert kwargs['env']['CLOUDSDK_CORE_DISABLE_PROMPTS'] == '1'
        return subprocess.CompletedProcess(argv, 0, b'[{"id":"beta","current_version_string":"2026.09.25"}]')
    cloud = Cloud('SDK-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    assert cloud.require_beta() == {'status': 'GCLOUD_BETA_INSTALLED', 'beta_version': '2026.09.25'}
    assert calls == [['SDK-fixture', 'components', 'list', '--only-local-state', '--format=json', '--quiet']]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('payload,code', [
    ([], 0), ([{'id': 'core', 'current_version_string': 'x'}], 0),
    ([{'id': 'beta'}], 0), ([{'id': 'beta', 'current_version_string': None}], 0),
    ([{'id': 'beta', 'current_version_string': ' '}], 0),
    ([{'id': 'beta', 'current_version_string': 'x'}] * 2, 0),
    ({'id': 'beta'}, 0), (['beta'], 0), ('malformed JSON', 0),
    ([{'id': 'beta', 'current_version_string': 'x'}], 1)])
def test_beta_probe_fails_closed(payload, code):
    from scripts.study_operator.i7_sdk_preflight import verify_beta
    content = payload if isinstance(payload, str) else json.dumps(payload)
    with pytest.raises(ValueError, match='^GCLOUD_BETA_REQUIRED$'):
        verify_beta('fixture', invoke=lambda *a, **kw: subprocess.CompletedProcess([], code, content))


@pytest.mark.parametrize('error', [OSError('missing executable'), subprocess.TimeoutExpired('fixture', 15)])
def test_beta_probe_unavailable_or_timeout(error):
    from scripts.study_operator.i7_sdk_preflight import verify_beta
    def unavailable(*a, **kw):
        raise error
    with pytest.raises(ValueError, match='^GCLOUD_BETA_REQUIRED$'):
        verify_beta('fixture', invoke=unavailable)


def test_no_discoverable_sdk_names_the_beta_blocker(monkeypatch):
    from scripts.study_operator import i7_sdk_preflight
    monkeypatch.setattr(i7_sdk_preflight.shutil, 'which', lambda *a: None)
    with pytest.raises(ValueError, match='^GCLOUD_BETA_REQUIRED$'):
        i7_sdk_preflight.verify_beta(invoke=lambda *a, **kw: pytest.fail('missing executable invoked'))


def test_check_names_beta_blocker_without_constructing_cloud(tmp_path, monkeypatch, capsys):
    from scripts.study_operator import cloud_client, i7_phase_b, i7_sdk_preflight
    def missing(*a, **kw):
        raise ValueError('GCLOUD_BETA_REQUIRED')
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', missing)
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: pytest.fail('Cloud constructed by --check'))
    assert i7_phase_b.main(['--check', '--sdk', 'fixture', '--root', str(tmp_path), 'gpu-restore']) == 0
    result = json.loads(capsys.readouterr().out)
    assert {'owner': 'code', 'code': 'GCLOUD_BETA_REQUIRED'} in result['blockers']


@pytest.mark.parametrize('step', ['cpu-build', 'gpu-restore', 'ready', 'tls-three', 'contexts',
    'smoke', 'restore', 'privacy', 'withdraw-purge', 'stimulus', 'gate'])
def test_phase_b_missing_beta_blocks_before_paid_dispatch(tmp_path, monkeypatch, capsys, step):
    from scripts.study_operator import cloud_client, i7, i7_browser_preflight, i7_cpu_pipeline
    from scripts.study_operator import i7_phase_b, i7_readiness, i7_sdk_preflight, i7_streamlit_preflight
    from scripts.study_operator import i7_workflows
    def missing(*a, **kw):
        raise ValueError('GCLOUD_BETA_REQUIRED')
    monkeypatch.setattr(i7_phase_b, 'datetime', Fixed)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(i7_streamlit_preflight, 'verify', lambda *a: {})
    monkeypatch.setattr(i7_streamlit_preflight, 'verify_before_cpu', lambda: {})
    monkeypatch.setattr(i7_cpu_pipeline, 'verify_image_or_stop_cpu', lambda *a: {})
    monkeypatch.setattr(i7_readiness, 'evaluate', lambda *a, **kw: {'blockers': []})
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', missing)
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: object())
    monkeypatch.setattr(i7, 'Operator', lambda *a, **kw: object())
    monkeypatch.setattr(i7_workflows, 'verify_tool_source', lambda *a: pytest.fail('SDK gate bypassed'))
    monkeypatch.setattr(i7_phase_b, 'dispatch', lambda *a: pytest.fail('paid dispatch before SDK gate'))
    with pytest.raises(SystemExit) as error:
        i7_phase_b.main(['--live', '--root', str(tmp_path), '--sdk', 'fixture', step])
    assert error.value.code == 2
    assert 'GCLOUD_BETA_REQUIRED' in capsys.readouterr().err


@pytest.mark.parametrize('operation', ['reserve', 'start', 'gate-start', 'stimulus-start', 'privacy',
    'series', 'schedule', 'restore-check', 'invite'])
def test_owner_cli_missing_beta_blocks_before_cloud(tmp_path, monkeypatch, capsys, operation):
    from scripts.study_operator import cloud_client, i7, i7_sdk_preflight
    def missing(*a, **kw):
        raise ValueError('GCLOUD_BETA_REQUIRED')
    monkeypatch.setattr(i7, 'datetime', Fixed)
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', missing)
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: pytest.fail('Cloud before local SDK gate'))
    with pytest.raises(SystemExit) as error:
        i7.main(['--live', '--root', str(tmp_path), '--sdk', 'fixture', operation])
    assert error.value.code == 2
    assert 'GCLOUD_BETA_REQUIRED' in capsys.readouterr().err


@pytest.mark.parametrize('step', ['close', 'reconcile', 'technical-cycle'])
def test_cleanup_can_dispatch_without_beta(tmp_path, monkeypatch, step):
    from scripts.study_operator import cloud_client, i7, i7_browser_preflight
    from scripts.study_operator import i7_phase_b, i7_sdk_preflight, i7_workflows
    monkeypatch.setattr(i7_phase_b, 'datetime', Fixed)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', lambda *a: pytest.fail('SDK must not block cleanup'))
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: object())
    monkeypatch.setattr(i7, 'Operator', lambda *a, **kw: object())
    monkeypatch.setattr(i7_workflows, 'verify_tool_source', lambda *a: None)
    monkeypatch.setattr(i7_phase_b, 'dispatch', lambda *a: {'status': 'cleanup dispatched'})
    assert i7_phase_b.main(['--live', '--root', str(tmp_path), '--sdk', 'fixture', step]) == 0


@pytest.mark.parametrize('operation', ['stop', 'guard', 'close'])
def test_owner_cleanup_can_dispatch_without_beta(tmp_path, monkeypatch, operation):
    from types import SimpleNamespace
    from scripts.study_operator import cloud_client, i7, i7_sdk_preflight, i7_workflows
    monkeypatch.setattr(i7, 'datetime', Fixed)
    monkeypatch.setattr(i7_sdk_preflight, 'verify_beta', lambda *a: pytest.fail('SDK must not block cleanup'))
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: object())
    owner = SimpleNamespace(stop=lambda: {}, guard=lambda: {})
    monkeypatch.setattr(i7, 'Operator', lambda *a, **kw: owner)
    monkeypatch.setattr(i7_workflows, 'verify_tool_source', lambda *a: None)
    monkeypatch.setattr(i7_workflows, 'dispatch', lambda *a: {})
    assert i7.main(['--live', '--root', str(tmp_path), '--sdk', 'fixture', operation]) == 0
