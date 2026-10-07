"""CPU-only guest restoration proof. Prints hashes/counts, never stored sessions."""
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import tarfile
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


def image_files(image, expected, prefix, *, invoke=subprocess.run):
    if any(PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or '\\' in name or ':' in name for name in expected):
        raise ValueError('Unsafe image source path')
    code = ('import json,hashlib; from pathlib import Path; '
        'root=Path("/opt/cloudrag/repository"); expected=json.loads('+repr(json.dumps(expected))+'); '
        'result={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}; '
        'print(json.dumps(result))')
    args = ['docker', 'run', '--rm=false', '--name='+prefix+'-source', '--network=none',
        '--user=10001:10001', '--read-only', '--tmpfs=/tmp:rw,noexec,nosuid,size=256m',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--log-driver=none',
        '--entrypoint=python', image, '-B', '-c', code]
    result = invoke(args, capture_output=True, timeout=180)
    if result.returncode or len(result.stdout) > 65536 or json.loads(result.stdout) != expected:
        raise ValueError('Frozen source in restored image differs')
    return dict(files=len(expected), all_expected_files_verified=True,
        source='RESTORED_IMMUTABLE_IMAGE', image_id=image,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest())


def image_configuration(image, export, *, invoke=subprocess.run):
    """Hash the exported config, independent of Docker's private storage driver."""
    inspected = invoke(['docker', 'image', 'inspect', image], capture_output=True, timeout=30)
    if inspected.returncode or len(inspected.stdout) > 2**20:
        raise ValueError('Restored image inspection failed')
    rows = json.loads(inspected.stdout)
    if len(rows) != 1 or rows[0]['Id'] != image:
        raise ValueError('Restored image ID differs')
    export = Path(export)
    if export.is_symlink():
        raise ValueError('Image export must not be a symlink')
    if not export.exists():
        saved = invoke(['docker', 'image', 'save', '--output='+str(export), image],
                       capture_output=True, timeout=360)
        if saved.returncode:
            raise ValueError('Restored image export failed')
    with tarfile.open(export, 'r:') as archive:
        member = archive.getmember('manifest.json')
        if not member.isfile() or member.size > 2**20:
            raise ValueError('Image export manifest invalid')
        manifest = json.load(archive.extractfile(member))
        if len(manifest) != 1:
            raise ValueError('One image export required')
        name = manifest[0]['Config']
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe image configuration member')
        member = archive.getmember(name)
        if not member.isfile() or member.size > 2**20:
            raise ValueError('Image configuration member invalid')
        data = archive.extractfile(member).read()
        config_digest = hashlib.sha256(data).hexdigest()
        descriptor = rows[0].get('Descriptor')
        if descriptor is not None:
            # Containerd's image ID is a manifest digest; classic Docker's is
            # a config digest. Hash the correct domain and its config link.
            if (descriptor.get('digest') != image or descriptor.get('mediaType') not in {
                    'application/vnd.oci.image.manifest.v1+json',
                    'application/vnd.docker.distribution.manifest.v2+json'}):
                raise ValueError('Unsupported or altered image descriptor')
            root = archive.getmember('blobs/sha256/'+image.removeprefix('sha256:'))
            if not root.isfile() or root.size > 2**20 or root.size != descriptor['size']:
                raise ValueError('Image manifest member invalid')
            manifest_data = archive.extractfile(root).read()
            if hashlib.sha256(manifest_data).hexdigest() != image.removeprefix('sha256:'):
                raise ValueError('Image manifest digest differs')
            config_descriptor = json.loads(manifest_data)['config']
            if config_descriptor['digest'] != 'sha256:'+config_digest or config_descriptor['size'] != len(data):
                raise ValueError('Image manifest configuration differs')
        elif config_digest != image.removeprefix('sha256:'):
            raise ValueError('Cached image configuration differs from image identity')
    if json.loads(data)['config']['WorkingDir'] != '/opt/cloudrag/repository':
        raise ValueError('Image repository working directory differs')
    return dict(image_id=image, config_sha256=config_digest,
                method='DOCKER_EXPORTED_CONFIG_BYTES_HASHED',
                config_link='OCI_MANIFEST' if descriptor is not None else 'CLASSIC_CONFIG_ID',
                export_bytes=export.stat().st_size)


def run(spec, *, metadata=None, invoke=subprocess.run, stage_observer=lambda stage: None, boot_id=None):
    stage_observer('GUEST_IDENTITY')
    if metadata is None:
        def metadata(key):
            request = Request('http://metadata.google.internal/computeMetadata/v1/instance/'+key,
                              headers={'Metadata-Flavor': 'Google'})
            with urlopen(request, timeout=5) as response:
                return response.read().decode()
    if (metadata('id') != spec['cpu_vm_id'] or metadata('machine-type').split('/')[-1] != 'e2-standard-2'
            or metadata('zone').split('/')[-1] != spec['zone']):
        raise ValueError('Guest CPU identity differs from owner receipt')
    stage_observer('HOST_INFRASTRUCTURE_FILES')
    host = files(spec['code_root'], spec['host_infrastructure_files'])
    # The effective repository is the image plus its read-only runtime binds.
    # Git-tracked evaluation queries live in the image, not the HF/index root.
    asset_files = {name: value for name, value in spec['artifact_files'].items()
                   if name.startswith(('data/models/', 'data/indices/'))}
    repository_artifacts = {name: value for name, value in spec['artifact_files'].items() if name not in asset_files}
    if set(repository_artifacts) & set(spec['source_files']):
        raise ValueError('Ambiguous restoration source/artifact membership')
    stage_observer('ARTIFACT_FILES')
    assets = files(spec['asset_root'], asset_files)
    stage_observer('IMAGE_CONFIG')
    configuration = image_configuration(spec['image_id'], spec.get('image_export_path',
        '/var/tmp/cloudrag-i5-restore-config-'+spec['cpu_vm_id']+'.tar'), invoke=invoke)
    stage_observer('FROZEN_IMAGE_FILES')
    source = image_files(spec['image_id'], {**spec['source_files'], **repository_artifacts},
        'cloudrag-i5-restore-'+spec['cpu_vm_id'], invoke=invoke)
    source.update(verified_combined_files=source['files'], files=len(spec['source_files']),
        artifact_files=len(repository_artifacts))
    artifacts = dict(files=len(spec['artifact_files']), host_files=assets['files'],
        image_files=len(repository_artifacts), all_expected_files_verified=True)
    # Manifest digest and every referenced layer/config blob must survive restore.
    stage_observer('OLLAMA_MANIFEST')
    manifest = Path(spec['ollama_models'])/'manifests/registry.ollama.ai/library/granite4.1/8b'
    if sha(manifest) != spec['model_digest']:
        raise ValueError('Restored Ollama manifest differs')
    model = json.loads(manifest.read_bytes())
    stage_observer('OLLAMA_BLOBS')
    for layer in [model['config'], *model['layers']]:
        blob = Path(spec['ollama_models'])/'blobs'/layer['digest'].replace(':', '-')
        if sha(blob) != layer['digest'].removeprefix('sha256:'):
            raise ValueError('Restored Ollama blob differs')
    stage_observer('RUNTIME_USER_PAIR')
    pair = runtime_pair(spec['image_id'], 'cloudrag-i5-restore-'+spec['cpu_vm_id'], invoke=invoke)
    return dict(status='CPU_RESTORATION_VERIFIED', synthetic=False,
        source_snapshot_id=spec['source_snapshot_id'], restored_disk_id=spec['restored_disk_id'],
        cpu_vm_id=spec['cpu_vm_id'], zone=spec['zone'], source=source, host_infrastructure=host, artifacts=artifacts,
        all_expected_files_verified=True, image_config_verified=True,
        image_configuration=configuration,
        image_id=spec['image_id'], model_manifest_and_blobs_verified=True,
        runtime_user_pair=pair, boot_id=boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        session_content_not_read_or_exported=True)


def observed(spec, **options):
    """Fail visibly by stage without persisting a traceback or private message."""
    stage = ['PREPARATION']
    try:
        return run(spec, stage_observer=lambda value: stage.__setitem__(0, value), **options)
    except Exception as error:
        kind = type(error).__name__
        if kind not in {'FileNotFoundError', 'PermissionError', 'ValueError', 'KeyError',
                'AttributeError', 'TimeoutExpired', 'URLError', 'HTTPError', 'OSError'}:
            kind = 'UNEXPECTED_EXCEPTION'
        return dict(status='CPU_PROBE_FAILED', synthetic=False, stage=stage[0], failure_code=kind,
            cpu_vm_id=spec['cpu_vm_id'], source_snapshot_id=spec['source_snapshot_id'],
            guest_python=list(sys.version_info[:3]), exception_text_not_persisted=True)
