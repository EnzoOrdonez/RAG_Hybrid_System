import copy
import json

import pytest

from scripts.study_operator.bucket_history import collect, validate
from scripts.study_operator.gcs import Storage
from test_study_soft_delete_contract import setup_anchor


def history(anchor):
    return dict(status='COMPLETE_ADMIN_ACTIVITY_QUERY', bucket=anchor['name'], complete_pagination=True,
        mutations=[dict(method=r['method'], server_timestamp=r['server_timestamp'],
            server_insert_id=r['server_insert_id'], status_code=0) for r in anchor['zero_retention_history']])


def test_live_policy_and_exact_audit_chain_are_both_required():
    anchor = setup_anchor()
    anchor['versioning'] = {'enabled': False}
    proof = validate(anchor, anchor, anchor, history(anchor))
    assert proof['complete_history'] and proof['no_soft_deleted_listing_claim']
    assert proof['soft_delete_retention_seconds'] == 0 and not proof['versioning_enabled']


@pytest.mark.parametrize('fault', ['missing_create', 'new_mutation', 'incomplete', 'retention', 'versioning', 'metadata_changed'])
def test_absence_or_policy_toggle_never_authorizes_deletion(fault):
    anchor = setup_anchor()
    before, after, logs = copy.deepcopy(anchor), copy.deepcopy(anchor), history(anchor)
    if fault == 'missing_create':
        logs['mutations'].pop(0)
    elif fault == 'new_mutation':
        logs['mutations'].append(dict(method='storage.buckets.update', server_timestamp='2026-10-05T00:00:00Z',
            server_insert_id='unanchored-toggle-and-reset', status_code=0))
    elif fault == 'incomplete':
        logs['complete_pagination'] = False
    elif fault == 'retention':
        after['softDeletePolicy']['retentionDurationSeconds'] = '604800'
    elif fault == 'versioning':
        after['versioning'] = {'enabled': True}
    else:
        after['metageneration'] = '6'
    with pytest.raises(ValueError):
        validate(before, after, anchor, logs)


def test_storage_proof_reads_metadata_on_both_sides_of_live_audit(monkeypatch):
    anchor = setup_anchor()
    events = []
    storage = Storage(anchor['name'], lambda: 'memory-only', creation_anchor=anchor,
        policy_history=lambda before: events.append('audit') or history(anchor))
    monkeypatch.setattr(storage, 'request', lambda path: events.append('metadata') or json.dumps(anchor).encode())
    assert storage.verify_zero_retention()['complete_history']
    assert events == ['metadata', 'audit', 'metadata']
    with pytest.raises(ValueError, match='Live audit'):
        Storage('synthetic', lambda: 'memory-only').verify_zero_retention()


def entry(bucket):
    return dict(resource={'labels': {'bucket_name': bucket}}, timestamp='2026-10-04T23:00:00Z', insertId='creation',
        protoPayload=dict(serviceName='storage.googleapis.com', methodName='storage.buckets.create',
            authenticationInfo={'principalEmail': 'PERSONAL_EMAIL_CANARY'},
            requestMetadata={'callerIp': 'CLIENT_IP_CANARY', 'callerSuppliedUserAgent': 'CLIENT_UA_CANARY'},
            request={'arbitrary_private_content': 'PRIVATE_TEXT_CANARY'}))


def test_empty_page_with_next_token_is_not_final_and_personal_metadata_stays_in_memory():
    bucket = 'cloudrag-study-i4-103950017681-20261004'
    calls = []

    def request(body):
        calls.append(body)
        return {'entries': [], 'nextPageToken': 'page-two'} if len(calls) == 1 else {'entries': [entry(bucket)]}

    proof = collect(lambda: pytest.fail('mock does not request credentials'), 'pure-loop-474323-a8', bucket,
        {'timeCreated': '2026-10-04T23:00:00Z'}, request=request)
    assert proof['complete_pagination'] and proof['pages'] == 2
    assert calls[1]['pageToken'] == 'page-two' and len(proof['mutations']) == 1
    assert all(sentinel not in json.dumps(proof) for sentinel in
        ('PERSONAL_EMAIL_CANARY', 'CLIENT_IP_CANARY', 'CLIENT_UA_CANARY', 'PRIVATE_TEXT_CANARY'))


def test_repeated_page_token_is_rejected_instead_of_silently_truncating_history():
    with pytest.raises(ValueError, match='Repeated'):
        collect(lambda: 'unused', 'pure-loop-474323-a8', 'cloudrag-study-i4-103950017681-20261004',
            {'timeCreated': '2026-10-04T23:00:00Z'}, request=lambda body: {'nextPageToken': 'repeat'})


def test_owner_audit_cli_only_writes_new_technical_receipt(monkeypatch, tmp_path):
    from scripts.study_operator.bucket_history import main

    monkeypatch.setattr('scripts.study_operator.run_control.require_limited', lambda: None)
    monkeypatch.setattr('scripts.study_operator.cloud_client.Cloud', lambda *args:
        type('SyntheticCloud', (), {'owner_token': lambda self: 'not-a-real-credential'})())
    monkeypatch.setattr(Storage, 'verify_zero_retention', lambda self:
        dict(method='synthetic', history={'mutations': []}, no_personal_metadata=True))
    installation = tmp_path/'installation.json'
    installation.write_text(json.dumps(dict(project='pure-loop-474323-a8', sessions_bucket='synthetic',
        sessions_bucket_creation={})))
    output = tmp_path/'proof.json'
    argv = ['--package', str(tmp_path), '--sdk', 'fixture', '--installation', str(installation), '--output', str(output)]
    main(argv)
    assert json.loads(output.read_bytes())['no_personal_metadata']
    with pytest.raises(ValueError, match='New own-package'):
        main(argv)
    assert 'not-a-real-credential' not in output.read_text()
