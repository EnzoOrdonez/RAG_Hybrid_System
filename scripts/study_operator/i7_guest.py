"""I7 effective host preparation readback through the fixed owner bridge."""
import json
from pathlib import Path


def preparation_check(active, *, execute):
    config = active['config']
    if config.get('iteration') != 7:
        raise ValueError('I7_PREPARATION_REQUIRED')
    receipt = json.loads((Path(active['boot_root'])/'host-preparation.json').read_bytes())
    if (receipt.get('status') != 'HOST_PREPARED' or receipt.get('blob_resident') is not True
            or receipt.get('models_empty') is not True or receipt.get('persistence_enabled') is not True):
        raise ValueError('HOST_PREPARATION_RECEIPT_INVALID')
    inspected = execute(['docker','inspect',active['ollama_container']])[0]
    if (not inspected['State']['Running'] or inspected['Config']['Image'] != config['ollama_image']
            or 'CUDA_CACHE_PATH=/cuda-cache' not in inspected['Config']['Env']
            or not any(m['Destination'] == '/cuda-cache' and m['Source'] == config['cuda_cache']
                       and m['RW'] for m in inspected['Mounts'])):
        raise ValueError('CUDA_PERSISTENT_MOUNT_DIFFERS')
    # execute() expects JSON. Python returns only the checked technical boolean.
    code = ("import subprocess,json; p=subprocess.check_output(['nvidia-smi',"
            "'--query-gpu=index,persistence_mode','--format=csv,noheader'],timeout=10);"
            "print(json.dumps({'enabled':p.decode().strip()=='0, Enabled'}))")
    if execute(['python3','-c',code]) != {'enabled':True}:
        raise ValueError('NVIDIA_PERSISTENCE_CHANGED')
    return {'status':'HOST_PREPARATION_VERIFIED','receipt':receipt,'cuda_mount_verified':True}
