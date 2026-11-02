"""Attempt 8: the configured synthetic smoke code reaches the app; real (<P900) codes never do."""
import pytest

from tests.study_helpers import configured
from tests.test_study_operator_deployment import config as installation

ENV = 'CLOUDRAG_SMOKE_PARTICIPANT'


def store(tmp_path, purpose='smoke'):
    from src.ui.components.study_sessions import StudyStore
    *_, protocol = configured(tmp_path)
    value = StudyStore(tmp_path/'sessions', protocol, purpose=purpose)
    value.freeze()
    return value, protocol


def invite(value, code):
    return value.issue(code, cell=1, profile='without_experience')


def test_smoke_store_accepts_only_the_configured_code(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, 'P998')
    value, _ = store(tmp_path)
    with pytest.raises(ValueError):
        invite(value, 'P999')
    assert invite(value, 'P998')


def test_smoke_store_defaults_to_p999(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    value, _ = store(tmp_path)
    with pytest.raises(ValueError):
        invite(value, 'P998')
    assert invite(value, 'P999')


@pytest.mark.parametrize('bad', ['P899', 'P12', 'PX', '', '999', 'P1234567', 'P９９９'])
def test_invalid_configured_smoke_code_fails_closed(tmp_path, monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    value, _ = store(tmp_path)
    for code in {'P999', bad} - {''}:
        with pytest.raises(ValueError):
            invite(value, code)


def test_rehearsal_rule_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, 'P997')
    value, _ = store(tmp_path, 'rehearsal')
    with pytest.raises(ValueError):
        invite(value, 'P997')
    assert invite(value, 'P998')


def seed(**extra):
    from scripts.study_operator.host_runtime import deployment_seed
    config = dict(installation(), ollama_version='0.0.0', fingerprint='f'*64, instance_id='1', **extra)
    return deployment_seed(config, 'boot', artifacts_sha256='a'*64, policy_sha256='b'*64)


def test_installation_smoke_code_propagates_to_seed_deployment_and_app_env(tmp_path, monkeypatch):
    from scripts import cloud_entrypoint, environment_identity
    from src.ui.components.study_sessions import StudyStore
    assert seed()['smoke_participant'] == 'P999'
    from scripts.study_operator.policy import OperatorError
    with pytest.raises(OperatorError):
        seed(smoke_participant='P100')
    monkeypatch.setattr(environment_identity, 'generate', lambda settings, path: settings)
    deployment = cloud_entrypoint.freeze_settings(seed(smoke_participant='P998'), tmp_path/'deployment.json')
    assert deployment['smoke_participant'] == 'P998'
    for key in ('CLOUDRAG_BUILD_ID', 'CLOUDRAG_MODEL_DIGEST', 'CLOUDRAG_ARTIFACT_MANIFEST', 'CLOUDRAG_STUDY_CONFIG',
                'CLOUDRAG_STUDY_ASSIGNMENTS', 'CLOUDRAG_STUDY_SESSION_DIR', 'CLOUDRAG_STUDY_PURPOSE', 'CLOUDRAG_MODE',
                'CLOUDRAG_DEMO_GPU', 'CLOUDRAG_BACKUP_BUCKET', 'CLOUDRAG_BACKUP_PREFIX', 'CLOUDRAG_ISOLATED_APP',
                'CLOUDRAG_ISOLATED_SERVICE', 'CLOUDRAG_BACKUP_SOCKET', ENV):
        monkeypatch.setenv(key, 'restored-after-test')
    cloud_entrypoint.configure(deployment)
    assert __import__('os').environ[ENV] == 'P998'
    *_, protocol = configured(tmp_path)
    value = StudyStore(tmp_path/'sessions', protocol, purpose='smoke')
    value.freeze()
    assert invite(value, 'P998')
    cloud_entrypoint.configure({k: v for k, v in deployment.items() if k != 'smoke_participant'})
    assert __import__('os').environ[ENV] == 'P999'


def test_smoke_observe_accepts_configured_code_only(tmp_path, monkeypatch):
    from scripts.study_operator.session_control import dispatch
    from src.ui.components import study_protocol
    monkeypatch.setenv(ENV, 'P998')
    value, protocol = store(tmp_path)
    value.admit(invite(value, 'P998'))
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: protocol)
    deployment = {'session_root': str(value.root), 'config_dir': 'reviewed', 'purpose': 'smoke'}
    observed = dispatch({'operation': 'i7-smoke-observe'}, deployment)
    assert observed['data']['assignment']['participant_id'] == 'P998'
    monkeypatch.setenv(ENV, 'P999')
    with pytest.raises(ValueError):
        dispatch({'operation': 'i7-smoke-observe'}, deployment)
