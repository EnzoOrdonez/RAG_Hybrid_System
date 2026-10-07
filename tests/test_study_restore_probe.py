import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.study_operator.restore_probe import files, image_files, observed, run, runtime_pair


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


def test_failed_guest_probe_reports_stage_without_private_exception(monkeypatch):
    def broken(spec, stage_observer, **options):
        stage_observer('SOURCE_FILES')
        raise FileNotFoundError('PRIVATE_CONTENT_NOT_FOR_TECHNICAL_LOGS')

    monkeypatch.setattr('scripts.study_operator.restore_probe.run', broken)
    result = observed(dict(cpu_vm_id='123', source_snapshot_id='456'))
    assert result['status'] == 'CPU_PROBE_FAILED' and result['stage'] == 'SOURCE_FILES'
    assert result['failure_code'] == 'FileNotFoundError'
    assert 'PRIVATE_CONTENT_NOT_FOR_TECHNICAL_LOGS' not in repr(result)


def test_image_source_is_checked_inside_restored_image_without_host_repository_assumption():
    expected = {'src/frozen.py': hashlib.sha256(b'frozen').hexdigest()}
    calls = []

    def invoke(argv, **options):
        calls.append(argv)
        assert '--network=none' in argv and '--read-only' in argv and '--rm=false' in argv
        assert '--log-driver=none' in argv and '--gpus' not in argv
        return SimpleNamespace(returncode=0, stdout=json.dumps(expected).encode(), stderr=b'')

    assert image_files('sha256:'+'a'*64, expected, 'synthetic', invoke=invoke)['files'] == 1
    assert '/opt/cloudrag/repository' in calls[0][-1]
    with pytest.raises(ValueError, match='differs'):
        image_files('sha256:'+'a'*64, expected, 'synthetic', invoke=lambda *a, **k:
            SimpleNamespace(returncode=0, stdout=b'{}', stderr=b''))


def test_restoration_checks_image_host_assets_and_model_blobs_before_qualifying(tmp_path):
    host, assets, models, docker = [tmp_path/name for name in ('host', 'assets', 'models', 'docker')]
    host.mkdir()
    assets.mkdir()
    (host/'infra.py').write_bytes(b'infrastructure')
    (assets/'data/indices').mkdir(parents=True)
    (assets/'data/indices/index').write_bytes(b'index bytes')
    config = json.dumps({'config': {'WorkingDir': '/opt/cloudrag/repository'}}).encode()
    image_digest = hashlib.sha256(config).hexdigest()
    image_file = docker/'image/overlay2/imagedb/content/sha256'/image_digest
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(config)
    blobs = models/'blobs'
    blobs.mkdir(parents=True)
    blob = b'model bytes'
    blob_digest = hashlib.sha256(blob).hexdigest()
    (blobs/('sha256-'+blob_digest)).write_bytes(blob)
    manifest = models/'manifests/registry.ollama.ai/library/granite4.1/8b'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(config={'digest': 'sha256:'+blob_digest}, layers=[])))
    frozen = {'src/frozen.py': hashlib.sha256(b'code').hexdigest()}
    spec = dict(cpu_vm_id='123', restored_disk_id='456', source_snapshot_id='789', zone='us-central1-a',
        code_root=str(host), asset_root=str(assets), docker_root=str(docker), image_id='sha256:'+image_digest,
        source_files=frozen, host_infrastructure_files={'infra.py': hashlib.sha256(b'infrastructure').hexdigest()},
        artifact_files={'data/indices/index': hashlib.sha256(b'index bytes').hexdigest(),
            'data/evaluation/test_queries.json': hashlib.sha256(b'Git tracked queries').hexdigest()},
        ollama_models=str(models),
        model_digest=hashlib.sha256(manifest.read_bytes()).hexdigest())

    def invoke(argv, **options):
        if any(arg.endswith('-source') for arg in argv):
            return SimpleNamespace(returncode=0, stdout=json.dumps({**frozen,
                'data/evaluation/test_queries.json': hashlib.sha256(b'Git tracked queries').hexdigest()}).encode(), stderr=b'')
        if any(arg.endswith('-missing') for arg in argv):
            return SimpleNamespace(returncode=1, stdout=b'', stderr=b'getpwuid: 10001')
        return SimpleNamespace(returncode=0, stdout=b'RUNTIME_IMPORT_OK:cloudrag', stderr=b'')

    def metadata(key):
        return {'id': '123', 'machine-type': 'machines/e2-standard-2', 'zone': 'zones/us-central1-a'}[key]
    result = run(spec, metadata=metadata, invoke=invoke, boot_id='synthetic-boot')
    assert result['all_expected_files_verified'] and result['image_config_verified']
    assert result['model_manifest_and_blobs_verified']
    assert result['source']['files'] == 1 and result['source']['verified_combined_files'] == 2
    assert result['artifacts']['files'] == 2 and result['artifacts']['host_files'] == result['artifacts']['image_files'] == 1
    assert result['runtime_user_pair']['status'] == 'PAIRED_RUNTIME_USER_SUPPORTED'
    (blobs/('sha256-'+blob_digest)).write_bytes(b'corrupted')
    failed = observed(spec, metadata=metadata, invoke=invoke, boot_id='synthetic-boot')
    assert failed['status'] == 'CPU_PROBE_FAILED' and failed['stage'] == 'OLLAMA_BLOBS'
    (assets/'data/indices/index').write_bytes(b'corrupted index')
    failed = observed(spec, metadata=metadata, invoke=invoke, boot_id='synthetic-boot')
    assert failed['status'] == 'CPU_PROBE_FAILED' and failed['stage'] == 'ARTIFACT_FILES'
