import hashlib
import json

import pytest

from scripts.study_operator.audit_package import inventory
from scripts.study_operator.package_verify import verify


def package(tmp_path):
    root = tmp_path/'package'
    root.mkdir()
    (root/'empty').write_bytes(b'')
    (root/'nested').mkdir()
    (root/'nested'/'ítems.json').write_bytes(b'{"synthetic":true}')
    staging = tmp_path/'pending'
    result = inventory(root, staging)
    staging.rename(root/'MANIFEST_SHA256.jsonl')
    return root, result['manifest_sha256']


def test_external_verifier_hashes_every_file_without_writing_audited_directory(tmp_path):
    root, pin = package(tmp_path)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    proof = verify(root, pin)
    assert proof['status'] == 'PASS' and proof['checked_files'] == proof['expected_files'] == 2
    assert proof['two_censuses_equal'] and proof['integrity_only_not_study_verdict']
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation', ['changed', 'added', 'missing'])
def test_tampering_or_membership_changes_are_failures(tmp_path, mutation):
    root, pin = package(tmp_path)
    if mutation == 'changed':
        (root/'empty').write_text('tampered')
    elif mutation == 'added':
        (root/'extra').write_text('unexpected')
    else:
        (root/'empty').unlink()
    assert verify(root, pin)['status'] == 'FAIL'


def test_second_census_detects_changes_after_first_listing(monkeypatch, tmp_path):
    root, pin = package(tmp_path)
    from scripts.study_operator import package_verify
    original = package_verify.hash_one

    def mutate(*args):
        (root/'late').write_bytes(b'not in initial census')
        return original(*args)

    monkeypatch.setattr(package_verify, 'hash_one', mutate)
    proof = verify(root, pin)
    assert proof['status'] == 'FAIL' and not proof['two_censuses_equal']


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a\\b', 'C:stream', 'a//b', 'MANIFEST_SHA256.jsonl'])
def test_unsafe_or_self_referential_rows_are_never_accepted(tmp_path, name):
    root, _ = package(tmp_path)
    manifest = root/'MANIFEST_SHA256.jsonl'
    manifest.write_text(json.dumps(dict(path=name, sha256='a'*64, size_bytes=0))+'\n')
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Unsafe'):
        verify(root, pin)


def test_manifest_pin_cannot_be_replaced_by_live_content(tmp_path):
    root, pin = package(tmp_path)
    (root/'MANIFEST_SHA256.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='never repin'):
        verify(root, pin)
