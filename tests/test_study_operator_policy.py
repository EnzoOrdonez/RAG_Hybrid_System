from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from scripts.study_operator.policy import OperatorError, purpose_allowed, session_margin, private_session_bucket, validate_minimal_iam, publication_export, confirm_deletion


def test_study_requires_human_record_and_exact_pdf(tmp_path):
    with pytest.raises(OperatorError, match='Enzo'):
        purpose_allowed('study', tmp_path)
    assert purpose_allowed('technical', tmp_path)['ethics_verified'] is False
    root = tmp_path / 'ethics'
    root.mkdir()
    pdf = b'synthetic fixture; not ethical approval'
    (root / 'ethics_approval.pdf').write_bytes(pdf)
    record = dict(committee='synthetic fixture committee', approval_code='FIXTURE', date='2026-01-01',
                  pdf_sha256=hashlib.sha256(pdf).hexdigest(), approved_b4_version='FIXTURE')
    (root / 'ethics_approval.json').write_text(json.dumps(record))
    assert purpose_allowed('study', tmp_path)['ethics_verified']
    (root / 'ethics_approval.pdf').write_bytes(b'changed fixture')
    with pytest.raises(OperatorError, match='study bloqueado'):
        purpose_allowed('study', tmp_path)


@pytest.mark.parametrize('minutes,allowed', [(69, False), (70, True), (71, True)])
def test_margin_uses_earlier_guest_and_native_limit(minutes, allowed):
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    guest = (now + timedelta(minutes=minutes)).isoformat()
    native = (now + timedelta(hours=3)).isoformat()
    if allowed:
        assert session_margin(guest, native, now=now)['remaining_seconds'] == minutes * 60
    else:
        with pytest.raises(OperatorError, match='70 minutos'):
            session_margin(guest, native, now=now)


def test_bucket_rejects_soft_delete_versioning_and_public_access():
    valid = dict(location='US-CENTRAL1', uniform_bucket_level_access=True,
                 public_access_prevention='enforced', versioning_enabled=False,
                 soft_delete_policy={'retentionDurationSeconds': '0'})
    assert private_session_bucket(valid)['private_bucket_verified']
    for key, value in [('versioning_enabled', True), ('uniform_bucket_level_access', False),
                       ('soft_delete_policy', {'retentionDurationSeconds': '604800'})]:
        with pytest.raises(OperatorError):
            private_session_bucket(dict(valid, **{key: value}))


def test_minimal_iam_rejects_broad_roles_and_metadata_access():
    sa = 'study@example.invalid'
    member = 'serviceAccount:' + sa
    vm = {'serviceAccounts': [{'email': sa, 'scopes': ['https://www.googleapis.com/auth/devstorage.read_write']}]}
    sessions = {'bindings': [{'role': r, 'members': [member]} for r in ['roles/storage.objectCreator', 'roles/storage.objectViewer']]}
    technical = {'bindings': [{'role': 'roles/storage.objectCreator', 'members': [member], 'condition': {
        'expression': "resource.name.startsWith('projects/_/buckets/technical-bucket/objects/iteration4/')"}}]}
    kwargs = dict(sa=sa, bucket='technical-bucket', metadata_reachable=False)
    assert validate_minimal_iam(vm, {}, sessions, technical, **kwargs)['minimal_iam_verified']
    with pytest.raises(OperatorError, match='rol de proyecto'):
        validate_minimal_iam(vm, {'bindings': [{'role': 'roles/editor', 'members': [member]}]}, sessions, technical, **kwargs)
    with pytest.raises(OperatorError, match='metadatos'):
        validate_minimal_iam(vm, {}, sessions, technical, **dict(kwargs, metadata_reachable=True))


def test_publication_never_includes_free_text_or_identifiers_in_aggregate():
    payload = publication_export([dict(participant_id='P01', session_id='IDENTIFIER_SENTINEL',
        attempts=[dict(condition='hybrid', response_class='answer', analysis_role='free_query',
                       question='PERSONAL_SENTINEL', answer='PRIVATE_SENTINEL')],
        comparative={'C4': 'PRIVATE_SENTINEL'})])
    assert 'SENTINEL' not in json.dumps(payload)
    assert 'P01' not in json.dumps(payload['aggregate'])
    assert payload['manual_review_inventory'][0]['publishable'] is False
    assert payload['automatic_publication_allowed'] is False


def test_study_delete_requires_exact_scope_confirmation():
    for operation, code, expected in [('withdraw', 'P01', 'P01'), ('purge-study', None, 'PURGAR')]:
        with pytest.raises(OperatorError, match='rechazado'):
            confirm_deletion(operation, code, 'study', '')
        assert confirm_deletion(operation, code, 'study', expected) == expected


def test_export_handles_failed_attempt_without_a_response_class():
    result = publication_export([dict(participant_id='P01', attempts=[
        dict(condition='hybrid', decline_class=None, status='error'),
        dict(condition='hybrid', decline_class='answered', status='success')])])
    assert {row['response_class'] for row in result['aggregate']} == {'answered', 'error'}
