"""Clean, hash-bound build input; preserve pins and the verified private vendor."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from scripts.study_operator.run_control import Recorder, require_limited


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vendor_inventory(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Vendor links are not a reproducible private snapshot')
    manifest = json.loads((root/'manifest.json').read_bytes())
    folder = root/'thesis-paper-agents'
    actual = {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob('*') if p.is_file()}
    if not actual or actual != manifest['files']:
        raise ValueError('Private vendor manifest differs; no build input generated')
    return manifest


def prepare(repo, vendor, root, label, *, run, python):
    repo, vendor, root = Path(repo), Path(vendor), Path(root)
    if not label.isalnum() or len(label) > 16:
        raise ValueError('Safe unique context label required')
    context, archive = root/('build-context-'+label), root/('build-context-'+label+'.tar.gz')
    if context.exists() or archive.exists():
        raise ValueError('Existing build input must be reconciled, never overwritten')
    manifest = vendor_inventory(vendor)
    def command(name, argv, timeout=120):
        return run(dict(name='context-'+label+'-'+name, argv=list(map(str, argv)), timeout=timeout))
    def output(name):
        return (root/('context-'+label+'-'+name+'.stdout')).read_text(encoding='utf-8').strip()
    command('status', ['git','-C',repo,'status','--porcelain'])
    command('branch', ['git','-C',repo,'branch','--show-current'])
    command('head', ['git','-C',repo,'rev-parse','HEAD'])
    if output('status') or output('branch') != 'fix/interview-readiness':
        raise ValueError('Only clean interview-readiness source can build')
    commit = output('head')
    command('remote', ['git','-C',repo,'ls-remote','origin','refs/heads/fix/interview-readiness'])
    if output('remote').split()[0] != commit:
        raise ValueError('Source commit must be externally anchored before build')
    # Git transfer temporaries are declared before clone; source and archive stay evidence.
    with (root/'DESTRUCTION_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write('\nBEFORE build context '+label+': owned Git clone transfer temporaries disposable. '
                     'Context, archive, translated lock and vendor receipts retained. Previous files not modified.\n')
    context.mkdir()
    command('clone', ['git','clone','--no-hardlinks','--no-checkout','--single-branch',
                     '--branch','fix/interview-readiness',repo,context/'repository'], 600)
    command('checkout', ['git','-C',context/'repository','checkout','--detach',commit])
    command('clone-status', ['git','-C',context/'repository','status','--porcelain'])
    if output('clone-status'):
        raise ValueError('Detached source clone is not clean')
    shutil.copytree(vendor, context/'vendor')
    if vendor_inventory(context/'vendor') != manifest:
        raise ValueError('Copied private vendor differs')
    command('lock', [python,context/'repository/scripts/locked_vendor_requirements.py',
        '--lock',context/'repository/requirements-lock.txt','--vendor',context/'vendor/thesis-paper-agents',
        '--manifest',context/'vendor/manifest.json','--output',root/('requirements-linux-'+label+'.txt')])
    shutil.copyfile(context/'repository/.dockerignore',context/'.dockerignore')
    with tarfile.open(archive,'x:gz',compresslevel=3) as target:
        for name in ('repository','vendor','.dockerignore'):
            target.add(context/name,arcname=name)
    result = dict(status='CLEAN_PUBLISHED_BUILD_CONTEXT_VERIFIED',commit=commit,bundle=str(archive),
        sha256=file_hash(archive),bytes=archive.stat().st_size,vendor_files=len(manifest['files']),
        vendor_manifest_sha256=file_hash(vendor/'manifest.json'),vendor_revision=manifest['revision'],
        requirements_lock_sha256=file_hash(repo/'requirements-lock.txt'),pins_preserved=True,
        at=datetime.now(timezone.utc).isoformat(),build_not_started=True)
    with (root/('build-context-'+label+'-receipt.json')).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('repo','vendor','package','label'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args(argv)
    require_limited()
    recorder = Recorder(args.package,args.repo,agent='Codex',model='gpt-6.1-sol',phase=3)
    result = prepare(args.repo,args.vendor,args.package,args.label,run=recorder.run,
                     python=Path(args.repo)/'.venv-app/Scripts/python.exe')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
