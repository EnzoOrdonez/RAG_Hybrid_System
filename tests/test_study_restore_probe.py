import hashlib
from types import SimpleNamespace

import pytest

from scripts.study_operator.restore_probe import files, runtime_pair


def test_restore_files_are_really_hashed_and_unsafe_or_changed_paths_fail(tmp_path):
    (tmp_path/'key').write_bytes(b'expected bytes')
    expected = {'key': hashlib.sha256(b'expected bytes').hexdigest()}
    assert files(tmp_path, expected)['files'] == 1
    (tmp_path/'key').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        files(tmp_path, expected)
    with pytest.raises(ValueError, match='Unsafe'):
        files(tmp_path, {'../outside': 'a'*64})


def test_runtime_user_pair_is_one_variable_with_retained_isolated_containers():
    calls = []

    def invoke(argv, **options):
        calls.append(argv)
        assert '--rm=false' in argv and '--network=none' in argv and '--log-driver=none' in argv
        assert options['timeout'] == 180
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout=b'', stderr=b'getpwuid(): uid not found: 10001')
        return SimpleNamespace(returncode=0, stdout=b'RUNTIME_IMPORT_OK:cloudrag\n', stderr=b'')

    result = runtime_pair('sha256:'+'b'*64, 'owned-prefix', invoke=invoke)
    assert result['status'] == 'PAIRED_RUNTIME_USER_SUPPORTED'
    assert result['model_generation_not_run']
    assert calls[0][calls[0].index('--entrypoint=python')+1] == calls[1][calls[1].index('--entrypoint=python')+1]
    assert "os.environ['USER']='cloudrag'" not in calls[0][-1]
    assert "os.environ['USER']='cloudrag'" in calls[1][-1]


def test_other_import_failure_does_not_qualify_runtime_candidate():
    def invoke(argv, **options):
        return SimpleNamespace(returncode=1, stdout=b'', stderr=b'ModuleNotFoundError')

    assert runtime_pair('sha256:'+'b'*64, 'owned', invoke=invoke)['status'] == 'PAIRED_RUNTIME_USER_NOT_SUPPORTED'
