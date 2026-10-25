"""Install an immutable owner bundle into operator5 without touching operators3/4."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from scripts.study_operator.run_control import require_limited

ROOT = Path('C:/CloudRAG/operator-iteration5')


def authorized_root(root):
    root = Path(root).absolute()
    if (root.name != 'operator-iteration5' or root.resolve() != root.absolute()
            or any(p.is_symlink() or p.is_junction() for p in (root, *root.parents))):
        raise ValueError('Only the new operator5 destination is authorized')
    return root


def prepared_archive(archive_bytes):
    prepared = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts
                    or '\\' in member.name or ':' in member.name
                    or not member.name.startswith(('scripts/study_operator/', 'src/', 'config/'))
                    or member.name in prepared):
                raise ValueError('Unsafe or duplicate operator archive entry')
            prepared[member.name] = archive.extractfile(member).read()
    if 'scripts/study_operator/cli.py' not in prepared or 'src/ui/components/session_storage.py' not in prepared:
        raise ValueError('Owner dependency bundle incomplete')
    return prepared


def install(root, archive_bytes, wrapper, config, *, source_commit):
    root = authorized_root(root)
    if config.get('project') != 'pure-loop-474323-a8' or not Path(config['python']).is_file():
        raise ValueError('Project and existing project Python required')
    prepared = prepared_archive(archive_bytes)
    files = {name: hashlib.sha256(content).hexdigest() for name, content in prepared.items()}
    marker = dict(schema_version=1, source_commit=source_commit, files=files,
                  wrapper_sha256=hashlib.sha256(wrapper).hexdigest(),
                  initial_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  previous_operators_not_modified=True, ethics_approval_not_created=True)
    receipt = root/'bundle_manifest.json'
    if receipt.exists():
        if json.loads(receipt.read_bytes()) != marker:
            raise ValueError('Existing operator5 release differs; no overwrite')
        for name, digest in files.items():
            if hashlib.sha256((root/'bundle'/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Installed owner bundle changed')
        if hashlib.sha256((root/'operator.ps1').read_bytes()).hexdigest() != marker['wrapper_sha256']:
            raise ValueError('Installed wrapper changed')
        return dict(status='EXISTING_IMMUTABLE_OWNER_BUNDLE_VERIFIED', files=len(files), source_commit=source_commit)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty operator5 without sealed bundle; reconcile partial installation')
    root.mkdir(parents=True, exist_ok=True)
    for name, content in prepared.items():
        path = root/'bundle'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    with (root/'operator.ps1').open('xb') as stream:
        stream.write(wrapper)
    with (root/'installation.json').open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(marker, stream, indent=2)
    return dict(status='NEW_OWNER_BUNDLE_INSTALLED_ACCEPTANCE_NOT_INFERRED', files=len(files), source_commit=source_commit)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('repo', 'config', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    require_limited()
    repo = Path(args.repo)
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments], timeout=120)
    if git('status', '--porcelain').strip():
        raise ValueError('Install only the clean published final source')
    commit = git('rev-parse', 'HEAD').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/fix/interview-readiness').decode().split()[0] != commit:
        raise ValueError('Owner bundle source not externally anchored')
    archive = git('archive', commit, '--', 'scripts/study_operator', 'src', 'config')
    wrapper = (repo/'scripts/study_operator/operator5.ps1').read_bytes()
    result = install(ROOT, archive, wrapper, json.loads(Path(args.config).read_bytes()), source_commit=commit)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
