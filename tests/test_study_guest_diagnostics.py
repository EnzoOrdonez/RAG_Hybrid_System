import json

import pytest

from scripts.study_operator import guest_bridge
from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def test_new_boot_waits_without_using_previous_boot_active_record(tmp_path,monkeypatch):
    monkeypatch.setattr(guest_bridge,'ROOT',tmp_path)
    monkeypatch.setattr(guest_bridge,'current_boot',lambda:'new-boot')
    (tmp_path/'active.json').write_text(json.dumps(dict(boot_id='previous-boot')))
    assert guest_bridge.dispatch(dict(operation='preflight'))['status'] == 'WAITING'


def test_failed_freeze_returns_readable_failure_and_diagnostics_without_active_file(tmp_path,monkeypatch):
    monkeypatch.setattr(guest_bridge,'ROOT',tmp_path)
    monkeypatch.setattr(guest_bridge,'current_boot',lambda:'new-boot')
    root = tmp_path/'boots/new-boot'
    root.mkdir(parents=True)
    (root/'meta').mkdir()
    (root/'failure.json').write_text('{"status":"FAILED","reason":"HOST_COMMAND_FAILED"}')
    (root/'command-0001.json').write_text('{"command":["docker","inspect"],"exit_code":1}')
    (root/'meta/environment_identity.json').write_text('{"source":{"commit":"synthetic"}}')
    (root/'meta/full_session.json').write_text('{"must_not_leave_private_store":true}')
    (root/'access.log').write_text('must-not-be-exported')
    assert guest_bridge.dispatch(dict(operation='preflight'))['reason'] == 'BOOTSTRAP_FAILED'
    evidence = guest_bridge.dispatch(dict(operation='technical-evidence'))
    assert evidence['boot_id'] == 'new-boot' and evidence['session_content_excluded']
    assert set(evidence['files']) == {'failure.json','command-0001.json','environment_identity.json'}
    assert 'must_not' not in json.dumps(evidence)


def test_owner_preserves_safe_guest_reason_with_next_action_and_no_traceback(tmp_path):
    operator,cloud = installation(tmp_path)
    cloud.command = lambda *args,**options:b'{"status":"ERROR","reason":"BOOTSTRAP_FAILED"}'
    with pytest.raises(OperatorError,match='BOOTSTRAP_FAILED.*diagnostics'):
        operator.bridge(dict(operation='preflight'))
    cloud.command = lambda *args,**options:b'{"status":"ERROR","reason":"arbitrary private payload"}'
    with pytest.raises(OperatorError,match='GUEST_OPERATION_REJECTED') as error:
        operator.bridge(dict(operation='preflight'))
    assert 'arbitrary' not in str(error.value)
