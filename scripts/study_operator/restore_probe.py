"""CPU-only guest restoration proof. Prints hashes/counts, never stored sessions."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files(root, expected):
    root = Path(root)
    for relative, digest in expected.items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe restoration path')
        target = root/name
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()) or sha(target) != digest:
            raise ValueError('Restored file differs: '+relative)
    return dict(files=len(expected), all_expected_files_verified=True)


def runtime_pair(image, prefix, *, invoke=subprocess.run):
    results = []
    for enabled in (False, True):
        code = ("import os; [os.environ.pop(k,None) for k in ('USER','LOGNAME','LNAME','USERNAME')]; "
                + ("os.environ['USER']='cloudrag'; " if enabled else '')
                + "import torch._dynamo; import getpass; print('RUNTIME_IMPORT_OK:'+getpass.getuser())")
        args = ['docker', 'run', '--rm=false', '--name='+prefix+('-user' if enabled else '-missing'),
                '--network=none', '--user=10001:10001', '--read-only',
                '--tmpfs=/tmp:rw,noexec,nosuid,size=256m', '--cap-drop=ALL',
                '--security-opt=no-new-privileges', '--log-driver=none', '-e', 'CUDA_VISIBLE_DEVICES=',
                '--entrypoint=python', image, '-B', '-c', code]
        begin = time.monotonic()
        result = invoke(args, capture_output=True, timeout=180)
        results.append(dict(user_set=enabled, exit_code=result.returncode,
            duration_s=time.monotonic()-begin, argv=args,
            stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
            uid_error_observed=b'getpwuid' in result.stderr and b'10001' in result.stderr,
            success_observed=result.stdout.strip() == b'RUNTIME_IMPORT_OK:cloudrag'))
    supported = results[0]['exit_code'] != 0 and results[0]['uid_error_observed'] and results[1]['exit_code'] == 0 and results[1]['success_observed']
    return dict(status='PAIRED_RUNTIME_USER_SUPPORTED' if supported else 'PAIRED_RUNTIME_USER_NOT_SUPPORTED',
                one_variable='USER environment before import', cases=results,
                model_generation_not_run=True, image_unchanged=True)


def run(spec, *, metadata=None, invoke=subprocess.run):
    if metadata is None:
        def metadata(key):
            request = Request('http://metadata.google.internal/computeMetadata/v1/instance/'+key,
                              headers={'Metadata-Flavor': 'Google'})
            with urlopen(request, timeout=5) as response:
                return response.read().decode()
    if (metadata('id') != spec['cpu_vm_id'] or metadata('machine-type').split('/')[-1] != 'e2-standard-2'
            or metadata('zone').split('/')[-1] != spec['zone']):
        raise ValueError('Guest CPU identity differs from owner receipt')
    source = files(spec['code_root'], spec['source_files'])
    artifacts = files(spec['asset_root'], spec['artifact_files'])
    digest = spec['image_id'].removeprefix('sha256:')
    image = Path(spec.get('docker_root', '/var/lib/docker'))/'image/overlay2/imagedb/content/sha256'/digest
    if sha(image) != digest:
        raise ValueError('Cached image configuration differs from image identity')
    # Manifest digest and every referenced layer/config blob must survive restore.
    manifest = Path(spec['ollama_models'])/'manifests/registry.ollama.ai/library/granite4.1/8b'
    if sha(manifest) != spec['model_digest']:
        raise ValueError('Restored Ollama manifest differs')
    model = json.loads(manifest.read_bytes())
    for layer in [model['config'], *model['layers']]:
        blob = Path(spec['ollama_models'])/'blobs'/layer['digest'].replace(':', '-')
        if sha(blob) != layer['digest'].removeprefix('sha256:'):
            raise ValueError('Restored Ollama blob differs')
    pair = runtime_pair(spec['image_id'], 'cloudrag-i5-restore-'+spec['cpu_vm_id'], invoke=invoke)
    return dict(status='CPU_RESTORATION_VERIFIED', synthetic=False,
        source_snapshot_id=spec['source_snapshot_id'], restored_disk_id=spec['restored_disk_id'],
        cpu_vm_id=spec['cpu_vm_id'], zone=spec['zone'], source=source, artifacts=artifacts,
        all_expected_files_verified=True, image_config_verified=True,
        image_id=spec['image_id'], model_manifest_and_blobs_verified=True,
        runtime_user_pair=pair, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        session_content_not_read_or_exported=True)
