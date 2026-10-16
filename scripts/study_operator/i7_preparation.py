"""Mandatory I7 host preparation, with no model generation or RAG changes."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from scripts.study_operator.deployment import bind


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resident_pages(path):
    """Inspect every page through Linux mincore without modifying the file."""
    if os.name != 'posix':
        raise ValueError('LINUX_MINCORE_REQUIRED')
    size = Path(path).stat().st_size
    if not size:
        raise ValueError('EMPTY_MODEL_BLOB')
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    with Path(path).open('rb') as stream:
        address = libc.mmap(None,size,1,1,stream.fileno(),0)
        if address in (None,ctypes.c_void_p(-1).value):
            raise ValueError('MINCORE_MAPPING_FAILED')
        try:
            vector = (ctypes.c_ubyte*((size+os.sysconf('SC_PAGE_SIZE')-1)//os.sysconf('SC_PAGE_SIZE')))()
            if libc.mincore(address,size,vector) != 0:
                raise ValueError('MINCORE_FAILED')
            return all(page & 1 for page in vector)
        finally:
            libc.munmap(address,size)


def ollama_command(config, name):
    return ['docker','run','--rm=false','--name',name,'--network','host',
            '--log-driver','none','--gpus','all','-e','OLLAMA_HOST=127.0.0.1:11434',
            '-e','CUDA_CACHE_PATH=/cuda-cache',
            *bind(config['cuda_cache'],'/cuda-cache',False),
            *bind(config['ollama_models'],'/root/.ollama/models'),config['ollama_image']]


def prepare(config, command, *, models, resident=resident_pages):
    command(['nvidia-smi','--persistence-mode=1'],timeout=30)
    observed = command(['nvidia-smi','--query-gpu=index,persistence_mode',
                        '--format=csv,noheader'],capture=True)
    if observed.decode().strip() != '0, Enabled':
        raise ValueError('PERSISTENCE_NOT_ENABLED')
    driver = command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture=True)
    if driver.decode().strip() != config['cuda_cache_driver']:
        raise ValueError('CUDA_CACHE_DRIVER_DIFFERS')
    root = Path(config['ollama_models'])
    manifest = root/'manifests/registry.ollama.ai/library/granite4.1/8b'
    if file_sha(manifest) != config['model_digest']:
        raise ValueError('MODEL_MANIFEST_CHANGED')
    layers = [layer for layer in json.loads(manifest.read_bytes())['layers']
              if layer['mediaType'] == 'application/vnd.ollama.image.model']
    if len(layers) != 1:
        raise ValueError('ONE_MODEL_BLOB_REQUIRED')
    import re
    digest = layers[0]['digest']
    if not re.fullmatch('sha256:[a-f0-9]{64}',digest):
        raise ValueError('MODEL_BLOB_DIGEST_INVALID')
    blob = root/'blobs'/digest.replace(':','-')
    # Streaming the verified bytes also preloads the OS page cache.
    if blob.is_symlink() or file_sha(blob) != digest.removeprefix('sha256:'):
        raise ValueError('MODEL_BLOB_CHANGED')
    if not resident(blob):
        raise ValueError('MODEL_BLOB_NOT_FULLY_RESIDENT')
    cache = Path(config['cuda_cache'])
    if cache.is_symlink() or not cache.is_dir() or not config.get('cuda_cache_seed'):
        raise ValueError('CUDA_CACHE_SEED_REQUIRED')
    for relative, expected in config['cuda_cache_seed'].items():
        path = cache/relative
        if (not path.resolve().is_relative_to(cache.resolve()) or path.is_symlink()
                or not path.is_file() or file_sha(path) != expected):
            raise ValueError('CUDA_CACHE_SEED_CHANGED')
    if models() != []:
        raise ValueError('PREPARATION_REQUIRES_EMPTY_MODELS')
    return {'status':'HOST_PREPARED','persistence_enabled':True,'blob_sha256':digest.removeprefix('sha256:'),
            'blob_resident':True,'cuda_seed_files':len(config['cuda_cache_seed']),'models_empty':True}


def host_seconds(config):
    if config.get('i7_profile') == 'gate':
        if config.get('iteration') != 7 or config['purpose'] != 'technical' or config.get('native_stop_s') != 32400:
            raise ValueError('TECHNICAL_GATE_PROFILE_REQUIRED')
        return 32100
    return 165*60
