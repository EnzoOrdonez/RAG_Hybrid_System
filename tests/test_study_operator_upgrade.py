import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts.study_operator.installation import install
from scripts.study_operator.installation_upgrade import current_release, upgrade


OLD = 'a'*40
NEW = 'b'*40
WRAPPER = Path('scripts/study_operator/operator5.ps1').read_bytes()


def archive(text=b'print("RELEASE_FIXTURE")'):
    value = io.BytesIO()
    with tarfile.open(fileobj=value, mode='w') as stream:
        for name, content in [('scripts/study_operator/cli.py', text),
                              ('src/ui/components/session_storage.py', b'# dependency fixture')]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    return value.getvalue()


def installed(tmp_path, wrapper=b'legacy wrapper'):
    root = tmp_path/'operator-iteration5'
    install(root, archive(b'print("LEGACY_FIXTURE")'), wrapper,
        dict(project='pure-loop-474323-a8', python=sys.executable), source_commit=OLD)
    (root/'active.json').write_bytes(b'{"primary_creation_intent":{"status":"UNKNOWN"}}')
    (root/'runs').mkdir()
    (root/'runs'/'receipt.json').write_bytes(b'original immutable receipt')
    return root


def evidence(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*')
        if p.is_file() and (p.parts[-2] == 'runs' or p.name in ('active.json','installation.json','bundle_manifest.json')
                           or 'bundle' in p.relative_to(root).parts)}


def test_upgrade_preserves_legacy_and_all_state_and_is_idempotent(tmp_path):
    root = installed(tmp_path)
    before = evidence(root)
    result = upgrade(root, archive(), WRAPPER, source_commit=NEW)
    assert result['previous_source_commit'] == OLD
    assert result['no_cloud_effect'] and result['previous_bundles_preserved']
    for name, content in before.items():
        assert (root/name).read_bytes() == content
    assert (root/'releases/legacy-operator.ps1').read_bytes() == b'legacy wrapper'
    manifest, pointer = current_release(root)
    assert manifest['source_commit'] == NEW
    assert pointer['bundle_path'] == 'releases/'+NEW+'/bundle'
    assert not (root/'ethics').exists()
    again = upgrade(root, archive(), WRAPPER, source_commit=NEW)
    assert again['pointer_sha256'] == result['pointer_sha256']


def test_interrupted_pointer_activation_can_resume_without_changing_owner_state(tmp_path):
    root = installed(tmp_path)
    before = evidence(root)

    def fail(*args):
        raise OSError('fixture atomic activation failure')

    with pytest.raises(OSError, match='activation failure'):
        upgrade(root, archive(), WRAPPER, source_commit=NEW, activate=fail)
    assert current_release(root)[0]['source_commit'] == OLD
    assert not (root/'release.json').exists()
    for name, content in before.items():
        assert (root/name).read_bytes() == content
    assert upgrade(root, archive(), WRAPPER, source_commit=NEW)['source_commit'] == NEW


@pytest.mark.parametrize('target', ['bundle/scripts/study_operator/cli.py', 'operator.ps1'])
def test_altered_installation_rejected_before_release_preparation(tmp_path, target):
    root = installed(tmp_path)
    (root/target).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        upgrade(root, archive(), WRAPPER, source_commit=NEW)
    assert not (root/'releases').exists()
    assert not (root/'release.json').exists()


def test_interrupted_preparation_refuses_conflicting_immutable_bytes(tmp_path):
    root = installed(tmp_path)
    path = root/'releases'/NEW/'bundle/scripts/study_operator/cli.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'conflicting preparation')
    with pytest.raises(ValueError, match='immutable release differs'):
        upgrade(root, archive(), WRAPPER, source_commit=NEW)
    assert path.read_bytes() == b'conflicting preparation'
    assert (root/'operator.ps1').read_bytes() == b'legacy wrapper'


@pytest.mark.parametrize('value', ['../escape', 'short', 'A'*40])
def test_commit_scope_rejected_before_any_writes(tmp_path, value):
    root = installed(tmp_path)
    with pytest.raises(ValueError, match='Full published'):
        upgrade(root, archive(), WRAPPER, source_commit=value)
    assert not (root/'releases').exists()


def test_active_release_extra_file_and_pointer_tamper_rejected(tmp_path):
    root = installed(tmp_path)
    upgrade(root, archive(), WRAPPER, source_commit=NEW)
    path = root/'releases'/NEW/'bundle/extra.py'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='inventory differs'):
        current_release(root)
    path.unlink()  # Declared disposable synthetic fixture.
    pointer = json.loads((root/'release.json').read_bytes())
    pointer['manifest_sha256'] = '0'*64
    (root/'release.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='digest differs'):
        current_release(root)


@pytest.mark.skipif(os.name != 'nt', reason='Native PowerShell owner launcher contract')
def test_native_launcher_selects_release_and_rejects_tampered_code(tmp_path):
    root = installed(tmp_path, WRAPPER)
    command = ['powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
               '-File',str(root/'operator.ps1'),'status']
    result = subprocess.run(command, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert b'LEGACY_FIXTURE' in result.stdout
    upgrade(root, archive(), WRAPPER, source_commit=NEW)
    result = subprocess.run(command, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert b'RELEASE_FIXTURE' in result.stdout and b'LEGACY_FIXTURE' not in result.stdout
    path = root/'releases'/NEW/'bundle/scripts/study_operator/cli.py'
    path.write_bytes(b'print("MUST_NOT_RUN")')
    result = subprocess.run(command, capture_output=True, timeout=30)
    assert result.returncode != 0 and b'MUST_NOT_RUN' not in result.stdout
    assert hashlib.sha256((root/'active.json').read_bytes()).hexdigest()
