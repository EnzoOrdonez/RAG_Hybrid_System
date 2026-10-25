import hashlib
import io
import json
import tarfile

import pytest

from scripts.study_operator.installation import install


def archive(extra=None):
    value = io.BytesIO()
    with tarfile.open(fileobj=value, mode='w') as stream:
        for name in ['scripts/study_operator/cli.py', 'src/ui/components/session_storage.py', *(extra or [])]:
            data = b'fixture owner source'
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    return value.getvalue()


def test_new_install_idempotent_verified_bundle_and_old_operators_unchanged(tmp_path):
    old = tmp_path/'operator-iteration4'
    old.mkdir()
    (old/'evidence').write_bytes(b'frozen old operator')
    root = tmp_path/'operator-iteration5'
    python = tmp_path/'existing-python'
    python.write_bytes(b'existing executable fixture')
    config = dict(project='pure-loop-474323-a8', python=str(python))
    result = install(root, archive(), b'wrapper fixture', config, source_commit='a'*40)
    assert result['status'] == 'NEW_OWNER_BUNDLE_INSTALLED_ACCEPTANCE_NOT_INFERRED'
    assert install(root, archive(), b'wrapper fixture', config, source_commit='a'*40)['status'].startswith('EXISTING')
    assert (old/'evidence').read_bytes() == b'frozen old operator'
    assert not (root/'ethics').exists()
    manifest = json.loads((root/'bundle_manifest.json').read_bytes())
    assert manifest['files']['scripts/study_operator/cli.py'] == hashlib.sha256(b'fixture owner source').hexdigest()
    (root/'bundle/scripts/study_operator/cli.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bundle changed'):
        install(root, archive(), b'wrapper fixture', config, source_commit='a'*40)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'src/..\\..\\escape', 'scripts/study_operator/cli.py'])
def test_unsafe_or_duplicate_archive_never_writes_destination(tmp_path, name):
    python = tmp_path/'python'
    python.write_bytes(b'existing')
    root = tmp_path/'operator-iteration5'
    with pytest.raises(ValueError, match='archive entry'):
        install(root, archive([name]), b'wrapper', dict(project='pure-loop-474323-a8', python=str(python)), source_commit='a'*40)
    assert not root.exists()


def test_previous_operator_destination_rejected_before_any_write(tmp_path):
    old = tmp_path/'operator-iteration4'
    with pytest.raises(ValueError, match='operator5'):
        install(old, archive(), b'wrapper', {}, source_commit='a'*40)
    assert not old.exists()
