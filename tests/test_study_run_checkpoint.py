from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator import run_checkpoint as bridge
from scripts.study_operator.cloud_safety import targets
from scripts.study_operator.policy import OperatorError


NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, 'CLOUD_ROOT', tmp_path)
    root = tmp_path/'iteration5-run-fixture'
    root.mkdir()
    state = dict(status='ACTIVE', agent='Codex', model='fixture', resources=[], resource_intents=[],
        closure_reserved_utc='2026-10-09T00:00:00+00:00', independent_closure_verified=True,
        cloud_cutoff_usd=90, cost=dict(estimated_spend_usd=10, reserved_retention_and_closure_usd=3),
        open_exposures={'other': dict(maximum_usd=2)})
    (root/'STATE.json').write_text(json.dumps(state))
    config = dict(audit_run=str(root), ip_name='cloudrag-i5-ip', primary_vm=dict(zone='us-central1-a'))
    owner = dict(cost=dict(estimated_usd=1, margin_usd=.5, reservations={'boot': 3}))
    return root, config, owner


def test_checkpoint_lost_create_intent_drives_independent_stop(tmp_path, monkeypatch):
    root, config, owner = fixture(tmp_path, monkeypatch)
    owner['alternate_creation_intent'] = dict(name='cloudrag-i5-alt', disk_name='cloudrag-i5-alt-boot',
        zone='us-west4-a', ownership_marker='CloudRAG-I5-alternate-fixture')
    bridge.checkpoint(config, owner, now=NOW)
    state = json.loads((root/'STATE.json').read_bytes())
    vm = dict(name='cloudrag-i5-alt', id='456', zone='zones/us-west4-a',
        description='CloudRAG-I5-alternate-fixture', status='RUNNING')
    assert targets(state, [vm]) == [vm]
    assert [r['type'] for r in state['resource_intents']] == ['vm', 'disk']
    vm['description'] = 'foreign'
    with pytest.raises(ValueError):
        targets(state, [vm])


def test_actual_vm_disk_ip_snapshot_and_exposure_persist_without_session_content(tmp_path, monkeypatch):
    root, config, owner = fixture(tmp_path, monkeypatch)
    owner['alternate_vms'] = [dict(name='cloudrag-i5-alt', id='1', zone='us-central1-b',
        disk_name='cloudrag-i5-alt-boot', disk_id='2', ownership_marker='CloudRAG-I5-alt')]
    owner['snapshots'] = [dict(name='cloudrag-i5-ready-fixture', id='3', ownership_marker='CloudRAG-I5-ready')]
    owner.update(reserved_address_id='4', ip_ownership_marker='CloudRAG-I5-ip',
                 ip_reserved_utc=NOW.isoformat(), private_session_fixture='CANARY_NOT_FOR_TECHNICAL_RECORD')
    bridge.checkpoint(config, owner, now=NOW)
    content = (root/'STATE.json').read_text()
    state = json.loads(content)
    assert [(r['type'], r['id']) for r in state['resources']] == [('vm', '1'), ('disk', '2'), ('snapshot', '3'), ('address', '4')]
    assert state['open_exposures']['operator5']['maximum_usd'] == 4.5
    assert 'CANARY_NOT_FOR_TECHNICAL_RECORD' not in content
    assert state['operator5_checkpoint']['session_content_not_copied']
    bridge.checkpoint(config, owner, now=NOW)
    assert len(json.loads((root/'STATE.json').read_bytes())['resources']) == 4


def test_paid_admission_counts_other_and_owner_cost_once(tmp_path, monkeypatch):
    root, config, owner = fixture(tmp_path, monkeypatch)
    bridge.checkpoint(config, owner, now=NOW)
    bridge.admit(config, owner, 70, now=NOW)  # 10+3+2+4.5+70 =89.5; not double4.5.
    with pytest.raises(OperatorError, match='corte'):
        bridge.admit(config, owner, 70.5, now=NOW)
    state = json.loads((root/'STATE.json').read_bytes())
    state['independent_closure_verified'] = False
    (root/'STATE.json').write_text(json.dumps(state))
    with pytest.raises(OperatorError, match='supervisor'):
        bridge.admit(config, owner, 1, now=NOW)


@pytest.mark.parametrize('status', ['CLOSED', 'SEALED', 'CLOSING'])
def test_closed_admission_and_sealed_bytes_preserved(tmp_path, monkeypatch, status):
    root, config, owner = fixture(tmp_path, monkeypatch)
    state = json.loads((root/'STATE.json').read_bytes())
    state['status'] = status
    (root/'STATE.json').write_text(json.dumps(state))
    before = (root/'STATE.json').read_bytes()
    files_before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(OperatorError):
        bridge.admit(config, owner, 1, now=NOW)
    bridge.checkpoint(config, owner, now=NOW)
    if status != 'CLOSING':
        assert (root/'STATE.json').read_bytes() == before
        assert {p.name: p.read_bytes() for p in root.iterdir()} == files_before
    else:
        assert json.loads((root/'STATE.json').read_bytes())['status'] == 'CLOSING'


@pytest.mark.parametrize('defect', ['foreign_name', 'foreign_marker', 'foreign_zone', 'unknown_kind',
                                  'id_replaced', 'id_alias', 'disposed_reappears'])
def test_foreign_or_changed_identity_preserves_original_state(tmp_path, monkeypatch, defect):
    root, config, owner = fixture(tmp_path, monkeypatch)
    row = dict(type='vm', name='cloudrag-i5-own', id='1', zone='us-central1-a', ownership_marker='CloudRAG-I5-own')
    owner['audit_resources'] = [row]
    bridge.checkpoint(config, owner, now=NOW)
    if defect == 'foreign_name':
        row['name'] = 'foreign'
    elif defect == 'foreign_marker':
        row['ownership_marker'] = 'foreign'
    elif defect == 'foreign_zone':
        row['zone'] = 'europe-west1-a'
    elif defect == 'unknown_kind':
        row['type'] = 'account'
    elif defect == 'id_replaced':
        row['id'] = '2'
    elif defect == 'id_alias':
        row['name'] = 'cloudrag-i5-other'
    else:
        state = json.loads((root/'STATE.json').read_bytes())
        state['resources'][0]['disposed'] = True
        (root/'STATE.json').write_text(json.dumps(state))
    before = (root/'STATE.json').read_bytes()
    with pytest.raises(OperatorError):
        bridge.checkpoint(config, owner, now=NOW)
    assert (root/'STATE.json').read_bytes() == before


def test_unknown_package_and_deadline_block_before_paid_effect(tmp_path, monkeypatch):
    root, config, owner = fixture(tmp_path, monkeypatch)
    bridge.admit(config, owner, 1, now=NOW)
    with pytest.raises(OperatorError):
        bridge.admit(config, owner, 1, now=datetime(2026, 10, 9, tzinfo=timezone.utc))
    config['audit_run'] = str(tmp_path/'operator-iteration4')
    with pytest.raises(OperatorError, match='ajeno'):
        bridge.checkpoint(config, owner)
    assert root.exists()


@pytest.mark.parametrize('invalid', [True, -1, float('nan'), float('inf'), '1'])
def test_invalid_cost_not_admitted(tmp_path, monkeypatch, invalid):
    _, config, owner = fixture(tmp_path, monkeypatch)
    with pytest.raises(OperatorError, match='costo'):
        bridge.admit(config, owner, invalid, now=NOW)


def test_future_operator_without_active_audit_does_not_touch_old_package():
    bridge.checkpoint({}, {})
    bridge.admit({}, {}, 1, now=NOW)
