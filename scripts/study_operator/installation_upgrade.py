"""Activate a verified owner release, preserving every old bundle and owner state."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from filelock import FileLock

from scripts.study_operator.installation import ROOT, authorized_root, prepared_archive
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json


def sha(content):
    return hashlib.sha256(content).hexdigest()


def immutable_file(path, content):
    """A interrupted preparation can fill missing files, never replace evidence."""
    if path.is_symlink() or path.is_junction():
        raise ValueError('Release links forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing immutable release differs')
    else:
        with path.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def verify_bundle(bundle, manifest):
    files = manifest['files']
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual != set(files):
        raise ValueError('Installed bundle inventory differs')
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or not name.startswith(('scripts/study_operator/', 'src/', 'config/'))):
            raise ValueError('Unsafe installed bundle path')
        target = bundle/name
        if any(p.is_symlink() or p.is_junction() for p in (target, *target.parents)):
            raise ValueError('Installed bundle links forbidden')
        if sha(target.read_bytes()) != digest:
            raise ValueError('Installed bundle changed')


def current_release(root):
    """Legacy is immutable too; a pointer cannot bypass inventory verification."""
    pointer = root/'release.json'
    if not pointer.exists():
        manifest = json.loads((root/'bundle_manifest.json').read_bytes())
        verify_bundle(root/'bundle', manifest)
        return manifest, None
    value = json.loads(pointer.read_bytes())
    commit = value['source_commit']
    if (not re.fullmatch('[0-9a-f]{40}', commit) or value['schema_version'] != 1
            or value['bundle_path'] != 'releases/'+commit+'/bundle'):
        raise ValueError('Unsafe owner release pointer')
    raw = (root/'releases'/commit/'bundle_manifest.json').read_bytes()
    if sha(raw) != value['manifest_sha256']:
        raise ValueError('Release pointer digest differs')
    manifest = json.loads(raw)
    if manifest['source_commit'] != commit:
        raise ValueError('Release commit differs')
    verify_bundle(root/value['bundle_path'], manifest)
    return manifest, value


def upgrade(root, archive_bytes, wrapper, *, source_commit, activate=atomic_json):
    root = authorized_root(root)
    if not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('Full published source commit required')
    prepared = prepared_archive(archive_bytes)
    marker = dict(schema_version=1, source_commit=source_commit,
        files={name:sha(content) for name, content in prepared.items()}, wrapper_sha256=sha(wrapper),
        owner_state_not_changed=True, previous_bundles_preserved=True, ethics_approval_not_created=True)
    raw = (json.dumps(marker, sort_keys=True, indent=2)+'\n').encode()
    pointer = dict(schema_version=1, source_commit=source_commit,
        bundle_path='releases/'+source_commit+'/bundle', manifest_sha256=sha(raw))
    # CLI uses this same lock. No invocation can mutate state during activation.
    with FileLock(str(root/'operator.lock'), timeout=0):
        prior, existing = current_release(root)
        old_wrapper = (root/'operator.ps1').read_bytes()
        if sha(old_wrapper) not in (prior['wrapper_sha256'], marker['wrapper_sha256']):
            raise ValueError('Existing operator wrapper changed')
        protected = {name:(root/name).read_bytes() for name in ('active.json', 'installation.json')}
        destination = root/'releases'/source_commit
        if any(p.is_symlink() or p.is_junction() for p in (destination, destination.parent)):
            raise ValueError('Release directory links forbidden')
        for name, content in prepared.items():
            immutable_file(destination/'bundle'/name, content)
        immutable_file(destination/'bundle_manifest.json', raw)
        verify_bundle(destination/'bundle', marker)
        # Keep the installed legacy wrapper byte-for-byte, even after later upgrades.
        if existing is None and sha(old_wrapper) == prior['wrapper_sha256']:
            immutable_file(root/'releases'/'legacy-operator.ps1', old_wrapper)
        if old_wrapper != wrapper:
            temporary = root/'operator-upgrade.pending'
            with temporary.open('wb') as stream:
                stream.write(wrapper)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root/'operator.ps1')
        # New wrapper defaults to the verified legacy bundle until this point.
        activate(root/'release.json', pointer)
        if any((root/name).read_bytes() != value for name,value in protected.items()):
            raise ValueError('Owner state changed during upgrade')
        return dict(status='OWNER_RELEASE_ACTIVATED_ACCEPTANCE_NOT_INFERRED', source_commit=source_commit,
            previous_source_commit=prior['source_commit'], files=len(prepared),
            pointer_sha256=sha((root/'release.json').read_bytes()),
            owner_state_sha256={name:sha(value) for name,value in protected.items()},
            previous_bundles_preserved=True, no_cloud_effect=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require_limited()
    repo, output = Path(args.repo), Path(args.output)
    if output.exists():
        raise ValueError('New upgrade receipt required')

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments], timeout=120)

    if git('status', '--porcelain').strip():
        raise ValueError('Only clean published owner source can be activated')
    commit = git('rev-parse', 'HEAD').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/fix/interview-readiness').decode().split()[0] != commit:
        raise ValueError('Owner release lacks external anchor')
    result = upgrade(ROOT, git('archive', commit, '--', 'scripts/study_operator', 'src', 'config'),
        (repo/'scripts/study_operator/operator5.ps1').read_bytes(), source_commit=commit)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
