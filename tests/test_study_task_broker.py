from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.study_operator.task_broker import checked_request, dispatch


def fixture(tmp_path):
    app = tmp_path/'app'
    app.mkdir()
    root = tmp_path/'package'
    root.mkdir()
    plan = dict(root=str(root), app=str(app), commands=[dict(name='check', argv=['python', '--version'], timeout=10)])
    path = root/'test-plan.json'
    path.write_text(json.dumps(plan))
    request = dict(plan=str(path), plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), native_minutes=2)
    return root, app, request


def test_queue_is_scope_bound_and_never_captures_private_invites(tmp_path):
    root, app, request = fixture(tmp_path)
    assert checked_request(root, app, request).name == 'test-plan.json'
    with pytest.raises(ValueError, match='changed'):
        checked_request(root, app, dict(request, plan_sha256='a'*64))
    path = root/'test-plan.json'
    plan = json.loads(path.read_bytes())
    plan['commands'][0]['argv'] = ['python', 'invite', 'P999']
    path.write_text(json.dumps(plan))
    request['plan_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='invitation'):
        checked_request(root, app, request)


def test_queue_dispatch_is_once_only_and_fails_closed_after_deadline(tmp_path):
    root, app, request = fixture(tmp_path)
    (root/'STATE.json').write_text(json.dumps(dict(status='ACTIVE', agent='Fixture', model='Fixture',
        closure_reserved_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())))
    queue = root/'task-queue'
    queue.mkdir()
    (queue/'test.request.json').write_text(json.dumps(request))
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b'own task registered', stderr=b'')

    assert dispatch(root, app, invoke=invoke)['dispatched'] == 1
    assert dispatch(root, app, invoke=invoke)['dispatched'] == 0 and len(calls) == 1
    receipt = json.loads((queue/'test.request.receipt.json').read_bytes())
    assert receipt['registering_token_limited']
    state = json.loads((root/'STATE.json').read_bytes())
    state['status'] = 'CLOSING'
    (root/'STATE.json').write_text(json.dumps(state))
    assert dispatch(root, app, invoke=invoke)['status'] == 'ADMISSION_CLOSED'
