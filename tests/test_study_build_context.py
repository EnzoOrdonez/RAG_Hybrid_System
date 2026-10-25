import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts.study_operator.build_context import prepare, vendor_inventory


def inputs(tmp_path):
    repo,vendor,root = [tmp_path/name for name in ('repo','vendor','package')]
    (repo/'scripts').mkdir(parents=True)
    (repo/'scripts/locked_vendor_requirements.py').write_text('# fixture')
    (repo/'.dockerignore').write_text('models\n')
    (repo/'requirements-lock.txt').write_text('PINNED_FIXTURE\n')
    (repo/'config').mkdir()
    source=b'{"synthetic_fixture":true}\n'
    (repo/'config/UEQS_ES_official.json').write_bytes(source)
    (repo/'config/study.example.json').write_text(json.dumps(dict(schema_version=2,ueq_s=json.loads(source),
        ueq_s_source_sha256=hashlib.sha256(source).hexdigest())))
    (vendor/'thesis-paper-agents').mkdir(parents=True)
    (vendor/'thesis-paper-agents/module.py').write_bytes(b'PRIVATE_FIXTURE')
    (vendor/'manifest.json').write_text(json.dumps(dict(revision='fixture-revision',
        files={'module.py':hashlib.sha256(b'PRIVATE_FIXTURE').hexdigest()})))
    root.mkdir()
    return repo,vendor,root


def runner(repo,root, *, status='', remote='a'*40):
    def run(command):
        argv = command['argv']
        name = command['name']
        result = ''
        if name.endswith('-status') and 'clone-status' not in name:
            result = status
        elif name.endswith('-branch'):
            result = 'fix/interview-readiness'
        elif name.endswith('-head'):
            result = 'a'*40
        elif name.endswith('-remote'):
            result = remote+'\trefs/heads/fix/interview-readiness'
        elif name.endswith('-clone'):
            assert '--no-hardlinks' in argv and '--no-checkout' in argv
            assert 'BEFORE build context' in (root/'DESTRUCTION_LOG.md').read_text()
            shutil.copytree(repo,argv[-1])
            (Path(argv[-1])/'.git').mkdir()
        (root/(name+'.stdout')).write_text(result)
    return run


def test_context_preserves_vendor_lock_and_git_source_without_overwrite(tmp_path):
    repo,vendor,root = inputs(tmp_path)
    result = prepare(repo,vendor,root,'candidate01',run=runner(repo,root),python='fixture-python')
    assert result['pins_preserved'] and result['build_not_started'] and result['vendor_files'] == 1
    assert result['ueq_s_cloned_reference']['canonical_eol']=='LF'
    assert json.loads((root/'build-context-candidate01-inventory.json').read_bytes()) == result
    assert not (root/'build-context-candidate01-receipt.json').exists()  # Reserved to the command recorder.
    assert (root/'build-context-candidate01/repository/requirements-lock.txt').read_bytes() == (repo/'requirements-lock.txt').read_bytes()
    with tarfile.open(result['bundle']) as archive:
        assert 'repository/.git' in archive.getnames()
        assert archive.extractfile('vendor/thesis-paper-agents/module.py').read() == b'PRIVATE_FIXTURE'
    with pytest.raises(ValueError,match='never overwritten'):
        prepare(repo,vendor,root,'candidate01',run=runner(repo,root),python='fixture-python')


@pytest.mark.parametrize('status,remote', [('M changed.py','a'*40), ('','b'*40)])
def test_dirty_or_unpublished_context_never_clones(tmp_path,status,remote):
    repo,vendor,root = inputs(tmp_path)
    with pytest.raises(ValueError):
        prepare(repo,vendor,root,'candidate01',run=runner(repo,root,status=status,remote=remote),python='fixture-python')
    assert not (root/'build-context-candidate01').exists()


def test_altered_private_vendor_rejected_before_git_or_copy(tmp_path):
    _,vendor,_ = inputs(tmp_path)
    (vendor/'thesis-paper-agents/module.py').write_bytes(b'ALTERED')
    with pytest.raises(ValueError,match='manifest differs'):
        vendor_inventory(vendor)


def test_windows_hash_binding_rejected_after_git_export_before_archive_or_build(tmp_path):
    repo,vendor,root=inputs(tmp_path)
    source=repo/'config/UEQS_ES_official.json'
    reference=repo/'config/study.example.json'
    value=json.loads(reference.read_bytes())
    value['ueq_s_source_sha256']=hashlib.sha256(source.read_bytes().replace(b'\n',b'\r\n')).hexdigest()
    reference.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='canonical cloned source bytes'):
        prepare(repo,vendor,root,'candidate02',run=runner(repo,root),python='fixture-python')
    assert not (root/'build-context-candidate02.tar.gz').exists()
    assert not (root/'requirements-linux-candidate02.txt').exists()
