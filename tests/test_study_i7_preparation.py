import hashlib
import json
from pathlib import Path

import pytest

from scripts.study_operator.i7_preparation import ollama_command, prepare


def fixture(tmp_path):
    model = tmp_path/'models'
    manifest = model/'manifests/registry.ollama.ai/library/granite4.1/8b'
    manifest.parent.mkdir(parents=True)
    blob = b'synthetic weights'
    digest = hashlib.sha256(blob).hexdigest()
    (model/'blobs').mkdir()
    (model/'blobs'/('sha256-'+digest)).write_bytes(blob)
    manifest.write_text(json.dumps({'layers':[{'mediaType':'application/vnd.ollama.image.model',
                                              'digest':'sha256:'+digest}]}))
    jit = tmp_path/'jit'
    jit.mkdir()
    (jit/'kernel').write_bytes(b'cache')
    config = {'iteration': 7,'ollama_models': str(model),'model_digest': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'cuda_cache': str(jit),'cuda_cache_seed': {'kernel':hashlib.sha256(b'cache').hexdigest()},
        'cuda_cache_driver': '580.178.04',
        'ollama_image': 'ollama@sha256:'+'a'*64}
    return config


def test_preparation_proves_cache_and_empty_models(tmp_path):
    config = fixture(tmp_path)
    commands = []
    def command(argv, **options):
        commands.append(argv)
        return b'580.178.04\n' if '--query-gpu=driver_version' in argv else b'0, Enabled\n'
    receipt = prepare(config, command, models=list, resident=lambda path:True)
    assert receipt['status'] == 'HOST_PREPARED'
    assert receipt['blob_resident'] and receipt['models_empty']
    argv = ollama_command(config,'owned')
    assert 'CUDA_CACHE_PATH=/cuda-cache' in argv
    assert any('target=/cuda-cache' in part and 'readonly' not in part for part in argv)
    assert commands[0] == ['nvidia-smi','--persistence-mode=1']


@pytest.mark.parametrize('failure',['nvidia','blob','residency','jit','models'])
def test_every_preparation_failure_blocks_readiness(tmp_path,failure):
    config = fixture(tmp_path)
    def command(argv,**options):
        if failure == 'nvidia':
            raise ValueError('NVIDIA_FAILED')
        return b'580.178.04\n' if '--query-gpu=driver_version' in argv else b'0, Enabled\n'
    if failure == 'blob':
        next((Path(config['ollama_models'])/'blobs').iterdir()).write_bytes(b'changed')
    if failure == 'jit':
        (Path(config['cuda_cache'])/'kernel').write_bytes(b'changed')
    with pytest.raises(ValueError):
        prepare(config,command,models=lambda:[{}] if failure == 'models' else [],
                resident=lambda path:failure != 'residency')


def test_cache_from_another_driver_is_rejected(tmp_path):
    config = fixture(tmp_path)
    with pytest.raises(ValueError,match='DRIVER'):
        prepare(config,lambda argv,**kw:b'0, Enabled\n' if '--query-gpu=index,persistence_mode' in argv else b'999\n',
                models=list,resident=lambda path:True)
