import copy
import io
import json
import urllib.error

import pytest

from scripts.study_operator.gcs import Storage


def anchor():
    return dict(id='new-bucket',name='new-bucket',timeCreated='2026-10-04T23:00:00Z',metageneration='1',
                softDeletePolicy={'retentionDurationSeconds':'0'})


def failed_list(message='Soft delete policy is required to list soft-deleted versions'):
    return urllib.error.HTTPError('https://storage.googleapis.com/b/new-bucket/o',400,'Bad Request',{},
                                  io.BytesIO(json.dumps({'error':{'message':message}}).encode()))


def configured(monkeypatch,actual=None,expected=None,message=None):
    storage = Storage('new-bucket',lambda:'memory-only',creation_anchor=expected or anchor())
    def request(path,**options):
        if path == '':
            return json.dumps(actual or anchor()).encode()
        raise failed_list(message) if message else failed_list()
    monkeypatch.setattr(storage,'request',request)
    return storage


def test_disabled_listing_is_not_reported_as_api_success(monkeypatch):
    storage = configured(monkeypatch)
    assert storage.objects('periods/',soft_deleted=True) == []
    proof = storage.soft_delete_verification
    assert proof['method'] == 'CREATION_GENERATION_ONE_ZERO_RETENTION'
    assert proof['api_list_status'] == 'HTTP400_POLICY_REQUIRED'
    assert proof['no_soft_delete_history_verified']


@pytest.mark.parametrize('key,value',[('metageneration','2'),('id','different'),('timeCreated','2025-01-01T00:00:00Z'),
                                    ('softDeletePolicy',{'retentionDurationSeconds':'604800'})])
def test_changed_history_or_identity_fails_before_deletion(monkeypatch,key,value):
    actual = copy.deepcopy(anchor())
    actual[key] = value
    storage = configured(monkeypatch,actual=actual)
    with pytest.raises(ValueError,match='history cannot be proved'):
        storage.objects('periods/',soft_deleted=True)
    assert storage.soft_delete_verification is None


def test_missing_creation_anchor_and_unrelated_error_never_become_empty(monkeypatch):
    storage = Storage('new-bucket',lambda:'memory-only')
    monkeypatch.setattr(storage,'request',lambda *args,**options: (_ for _ in ()).throw(failed_list()))
    with pytest.raises(urllib.error.HTTPError):
        storage.objects('periods/',soft_deleted=True)
    unrelated = configured(monkeypatch,message='Invalid page token')
    with pytest.raises(urllib.error.HTTPError):
        unrelated.objects('periods/',soft_deleted=True)


def setup_anchor():
    value = anchor()
    value['metageneration'] = '4'
    commands = [['sdk', 'storage', 'buckets', 'create', 'gs://new-bucket', '--soft-delete-duration=0'],
                ['sdk', 'storage', 'buckets', 'update', 'gs://new-bucket', '--no-versioning'],
                ['sdk', 'storage', 'buckets', 'add-iam-policy-binding', 'gs://new-bucket'],
                ['sdk', 'storage', 'buckets', 'add-iam-policy-binding', 'gs://new-bucket']]
    methods = ['storage.buckets.create', 'storage.buckets.update', 'storage.setIamPermissions', 'storage.setIamPermissions']
    value['zero_retention_history'] = [dict(method=method, command=command, exit_code=0,
        server_timestamp='2026-10-04T23:00:00Z', server_insert_id=str(index),
        audit_sha256='a'*64, command_receipt_sha256='b'*64)
        for index, (command, method) in enumerate(zip(commands, methods))]
    return value


def test_known_setup_mutations_require_live_identity_and_complete_evidence(monkeypatch):
    expected = setup_anchor()
    storage = configured(monkeypatch, actual=expected, expected=expected)
    assert storage.objects('periods/', soft_deleted=True) == []
    assert storage.soft_delete_verification['method'] == 'ANCHORED_ZERO_RETENTION_SETUP_HISTORY'
    actual = copy.deepcopy(expected)
    actual['metageneration'] = '5'
    with pytest.raises(ValueError, match='history cannot be proved'):
        configured(monkeypatch, actual=actual, expected=expected).objects('periods/', soft_deleted=True)


@pytest.mark.parametrize('change', ['missing', 'retention', 'unknown', 'failed', 'receipt'])
def test_unknown_history_never_authorizes_deletion(monkeypatch, change):
    expected = setup_anchor()
    if change == 'missing':
        expected['zero_retention_history'].pop()
    elif change == 'retention':
        expected['zero_retention_history'][1]['command'].append('--soft-delete-duration=7d')
    elif change == 'unknown':
        expected['zero_retention_history'][1]['method'] = 'unrecognized.update'
    elif change == 'failed':
        expected['zero_retention_history'][0]['exit_code'] = 1
    else:
        expected['zero_retention_history'][0]['command_receipt_sha256'] = ''
    with pytest.raises(ValueError, match='history cannot be proved'):
        configured(monkeypatch, actual=expected, expected=expected).objects('periods/', soft_deleted=True)
