"""Guest CPU-only clean service clone. Never operates on the original snapshots."""
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.study_operator.i7_preparation import file_sha
from scripts.study_operator.i7_privacy import archive_legacy

TIMES = ('023425','024929','033336','035910','140448','143041','172058','190612','211114')
MOUNT = Path('/mnt/cloudrag-i7-service')
# A boot-disk clone shares PARTUUID/UUID/LABEL; none of these may live on it (ADR-0023).
SYSTEM_PATHS = ('/','/boot','/boot/efi')
BY_ID_TIMEOUT_S = 60
BY_ID = Path('/dev/disk/by-id')


def clone_devices(job):
    """Device numbers of the clone's whole disk and partitions, by exact by-id name."""
    base = 'google-'+job['disk_name']
    return {os.stat(p).st_rdev for p in BY_ID.glob(base+'*')
            if p.name == base or p.name.startswith(base+'-part')}


def refuse_root_on_clone(job, *, devices=clone_devices, stat=os.stat):
    clone = devices(job)
    if not clone:
        raise ValueError('SERVICE_DISK_BY_ID_MISSING')
    for path in SYSTEM_PATHS:
        try:
            if stat(path).st_dev in clone:
                raise ValueError('ROOT_ON_SERVICE_CLONE')
        except FileNotFoundError:
            continue


def wait_target(job, *, run=subprocess.run, exists=Path.exists, sleep=time.sleep,
                clock=time.monotonic, timeout=BY_ID_TIMEOUT_S):
    """A hot-attached disk appears asynchronously; wait a bounded time for its by-id link."""
    run(['udevadm','settle','--timeout=30'],check=True,timeout=40,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    link = BY_ID/('google-'+job['disk_name']+'-part1')
    deadline = clock()+timeout
    while not exists(link):
        if clock() >= deadline:
            raise ValueError('SERVICE_DISK_BY_ID_TIMEOUT')
        sleep(1)
    return link


def safe_path(root, absolute):
    path = root/absolute.lstrip('/')
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('SERVICE_COPY_PATH_REJECTED')
    return path


def verify_mount(root, job):
    refuse_root_on_clone(job)  # First: a clone serving / makes every later check misleading.
    if os.name != 'posix' or root != MOUNT or not root.is_mount():
        raise ValueError('OWN_SECONDARY_SERVICE_MOUNT_REQUIRED')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',
                                     headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=5) as response:
        if response.read().decode() != job['cpu_id']:
            raise ValueError('CLEANER_CPU_IDENTITY_CHANGED')
    device = Path('/dev/disk/by-id/google-'+job['disk_name']+'-part1').resolve()
    result = subprocess.run(['findmnt','--json','-T',str(root),'-o','SOURCE,TARGET'],
                             capture_output=True,check=True,timeout=10)
    fs = json.loads(result.stdout)['filesystems']
    if len(fs) != 1 or Path(fs[0]['source']).resolve() != device or fs[0]['target'] != str(root):
        raise ValueError('CLEANER_TARGET_DISK_CHANGED')


def legacy(root, destination):
    paths = [safe_path(root,'/srv/cloudrag/iteration3/runtime/access-20261004T'+t+'Z/caddy-public-log.stderr')
             for t in TIMES]
    receipt = archive_legacy(paths,destination)
    return dict(receipt,content_base64=base64.b64encode(Path(destination).read_bytes()).decode())


def sanitize(root, job, *, verify_target=verify_mount):
    """Only an owned secondary clone, after owner ACK of the nine original files."""
    root = Path(root).resolve()
    verify_target(root,job)
    archive = Path(job['legacy_archive'])
    if file_sha(archive) != job['legacy_archive_ack_sha256']:
        raise ValueError('OWNER_LEGACY_ARCHIVE_ACK_REQUIRED')
    image_archive = Path(job['image_archive'])
    if file_sha(image_archive) != job['image_archive_sha256']:
        raise ValueError('QUALIFIED_IMAGE_ARCHIVE_CHANGED')
    config = dict(job['installation'])
    seed = safe_path(root,'/opt/cloudrag-i7-seed')
    seed.mkdir(parents=True,exist_ok=False)
    mapping = {'assets':config['asset_root'],'models':config['ollama_models'],'code':config['host_code'],
        'reviewed':config['reviewed_config'],'tls':'/srv/cloudrag/iteration5/tls','jit':config['cuda_cache']}
    small_files = {name:safe_path(root,config[key]).read_bytes() for name,key in
                   (('deployment-artifacts.json','artifact_manifest'),('preregistration.md','preregistration_file'))}
    for source in mapping.values():
        if not safe_path(root,source).is_dir():
            raise ValueError('CLEAN_SERVICE_SOURCE_MISSING')
    for name, source in mapping.items():
        path = safe_path(root,source)
        if not path.is_dir():
            raise ValueError('CLEAN_SERVICE_SOURCE_MISSING')
        shutil.move(str(path),str(seed/name))
    for name,data in small_files.items():
        (seed/name).write_bytes(data)
    shutil.copyfile(image_archive,seed/'images.tar')
    # Clone is offline. Source snapshots and CPU source disk are never cleaned.
    for relative in ('srv/cloudrag','var/lib/docker','var/log','tmp','var/tmp','var/lib/cloud/instances'):
        target = safe_path(root,'/'+relative)
        if target.exists():
            if not target.is_dir():
                raise ValueError('CLEAN_SERVICE_DIRECTORY_REQUIRED')
            shutil.rmtree(target)
        target.mkdir(parents=True,exist_ok=True)
        if relative in ('tmp','var/tmp'):
            target.chmod(0o1777)
    for home in [root/'root',*list((root/'home').glob('*'))]:
        for name in ('.bash_history','.python_history','.lesshst','.wget-hsts'):
            path = safe_path(root,'/'+(home/name).relative_to(root).as_posix())
            if path.is_file():
                path.unlink()
    service = safe_path(root,'/srv/cloudrag/i7/service')
    service.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(seed),str(service))
    tls = safe_path(root,'/srv/cloudrag/iteration5/tls')
    tls.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(service/'tls'),str(tls))
    config.update(iteration=7,asset_root='/srv/cloudrag/i7/service/assets',
        ollama_models='/srv/cloudrag/i7/service/models',host_code='/srv/cloudrag/i7/service/code',
        reviewed_config='/srv/cloudrag/i7/service/reviewed',cuda_cache='/srv/cloudrag/i7/service/jit',
        artifact_manifest='/srv/cloudrag/i7/service/deployment-artifacts.json',
        preregistration_file='/srv/cloudrag/i7/service/preregistration.md',
        image_archive='/srv/cloudrag/i7/service/images.tar')
    config['cuda_cache_seed'] = {p.relative_to(service/'jit').as_posix():file_sha(p)
                                for p in sorted((service/'jit').rglob('*')) if p.is_file()}
    if not config['cuda_cache_seed']:
        raise ValueError('CLEAN_SERVICE_JIT_SEED_MISSING')
    config.pop('bootstrap_inputs',None)
    return {'status':'CLEAN_SERVICE_DISK_PREPARED_NOT_BOOT_VALIDATED','installation':config,
            'legacy_archive_sha256':job['legacy_archive_ack_sha256'],
            'image_archive_sha256':job['image_archive_sha256']}


def mount_service(job, *, wait=wait_target, guard=refuse_root_on_clone, run=subprocess.run,
                  verify=verify_mount, mount_point=MOUNT):
    """Root guard before and (inside verify) after mounting the hot-attached clone."""
    wait(job)
    guard(job)
    mount_point.mkdir(mode=0o700,exist_ok=True)
    if not mount_point.is_mount():
        device = '/dev/disk/by-id/google-'+job['disk_name']+'-part1'
        run(['mount',device,str(mount_point)],check=True,timeout=30,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    verify(mount_point,job)
    return {'status':'OWN_SERVICE_DISK_MOUNTED'}


def main():
    job = json.load(sys.stdin)
    import re
    if not re.fullmatch('cloudrag-i7-service-[a-f0-9]{12}',job['disk_name']):
        raise ValueError('OWN_I7_SERVICE_DISK_REQUIRED')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',
                                     headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=5) as response:
        if response.read().decode() != job['cpu_id']:
            raise ValueError('CPU_IDENTITY_CHANGED')
    if job['operation'] == 'mount':
        print(json.dumps(mount_service(job)))
        return
    verify_mount(MOUNT,job)
    if job['operation'] == 'archive':
        result = legacy(MOUNT,job['legacy_archive'])
    elif job['operation'] == 'sanitize':
        result = sanitize(MOUNT,job)
    elif job['operation'] == 'export-images':
        images = job['images']
        if len(images) != 3 or any(not re.fullmatch(r'(?:[^\s]+@)?sha256:[a-f0-9]{64}',s) for s in images):
            raise ValueError('THREE_PINNED_IMAGES_REQUIRED')
        path = Path('/srv/cloudrag/iteration7-images-'+job['disk_name']+'.tar')
        if path.exists():
            raise ValueError('IMAGE_EXPORT_EXISTS')
        subprocess.run(['docker','save','--output',str(path),*images],check=True,timeout=900,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        result = {'image_archive':str(path),'image_archive_sha256':file_sha(path)}
    elif job['operation'] == 'unmount':
        subprocess.run(['sync'],check=True,timeout=60)
        subprocess.run(['umount',str(MOUNT)],check=True,timeout=60)
        result = {'status':'OWN_SERVICE_DISK_UNMOUNTED'}
    else:
        raise ValueError('FIXED_CLEAN_SERVICE_OPERATION_REQUIRED')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
