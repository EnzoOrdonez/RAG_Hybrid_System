"""Streaming canary census and allowlisted clean-disk import, without cloud writes."""
import hashlib
import shutil
import tarfile
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.study_operator.i7_preparation import file_sha

DOMAINS = ('caddy','app','runner','journald','serial','backups','bucket')


def scan(sources, canaries, *, exclusions=()):
    if len(canaries) != 2 or any(not isinstance(c,bytes) or not c for c in canaries):
        raise ValueError('TWO_MEMORY_ONLY_CANARIES_REQUIRED')
    excluded = {}
    for item in exclusions:
        if (set(item) != {'domain','locator','sha256','reason','date'} or not item['reason']
                or date.fromisoformat(item['date']) > datetime.now(timezone.utc).date() or len(item['sha256']) != 64):
            raise ValueError('DATED_EXACT_EXCLUSION_REQUIRED')
        excluded[(item['domain'],item['locator'])] = item
    result = {'status':'INCOMPLETE','hits':0,'domains':{},'exclusions':[]}
    overlap = max(map(len,canaries))-1
    failed = False
    for domain in DOMAINS:
        count, hits = 0, 0
        try:
            provider = sources[domain]
            for locator, blocks in provider():
                count += 1
                tail, found, digest = b'', set(), hashlib.sha256()
                for block in blocks:
                    if not isinstance(block,bytes):
                        raise TypeError('BINARY_STREAM_REQUIRED')
                    digest.update(block)
                    data = tail+block
                    found.update(i for i,canary in enumerate(canaries) if canary in data)
                    tail = data[-overlap:] if overlap else b''
                exemption = excluded.get((domain,locator))
                if exemption:
                    if digest.hexdigest() != exemption['sha256']:
                        raise ValueError('EXCLUSION_CONTENT_CHANGED')
                    result['exclusions'].append({k:v for k,v in exemption.items() if k != 'locator'})
                else:
                    hits += len(found)
            result['domains'][domain] = {'complete':True,'files':count,'hits':hits}
        except Exception as error:  # noqa: BLE001 - fail closed without private exception text
            failed = True
            result['domains'][domain] = {'complete':False,'files':count,'hits':hits,
                                         'error_type':type(error).__name__}
        result['hits'] += hits
    if len(result['exclusions']) != len(excluded):
        failed = True
    result['status'] = 'INCOMPLETE' if failed else 'FAIL' if result['hits'] else 'PASS'
    return result


def chunks(stream):
    yield from iter(lambda:stream.read(1024*1024), b'')


def tree_sources(root):
    """No ignored files; unreadable directories, symlinks and archives fail closed."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('SCAN_ROOT_REQUIRED')
    import os
    def error(exc):
        raise exc
    for directory, folders, files in os.walk(root,onerror=error,followlinks=False):
        for name in folders+files:
            if (Path(directory)/name).is_symlink():
                raise ValueError('SCAN_SYMLINK_REQUIRES_REVIEW')
        for name in sorted(files):
            path = Path(directory)/name
            if name.endswith(('.tar','.tar.gz','.tgz')):
                with tarfile.open(path) as archive:
                    for member in archive:
                        if member.isfile():
                            with archive.extractfile(member) as stream:
                                yield str(path)+'!'+member.name,chunks(stream)
                        elif not member.isdir():
                            raise ValueError('ARCHIVE_LINK_REQUIRES_REVIEW')
            else:
                with path.open('rb') as stream:
                    yield str(path),chunks(stream)


def inject(hostname, ip, user_agent, *, open_url=urllib.request.urlopen):
    """Canaries stay in headers and memory, never URLs, argv or technical receipts."""
    import ipaddress
    ipaddress.ip_address(ip)
    if not hostname.endswith('.sslip.io') or hostname[:-9] != str(ipaddress.IPv4Address(hostname[:-9])):
        raise ValueError('FIXED_TLS_HOST_REQUIRED')
    if any(c in user_agent for c in '\r\n') or not user_agent:
        raise ValueError('SAFE_MEMORY_USER_AGENT_REQUIRED')
    for path in ('/','/_stcore/health','/favicon.ico'):
        request = urllib.request.Request('https://'+hostname+path,
            headers={'User-Agent':user_agent,'X-Forwarded-For':ip,'X-Real-IP':ip,'Forwarded':'for='+ip})
        with open_url(request,timeout=15) as response:
            response.read(1024*1024)


def archive_legacy(paths, destination):
    paths, destination = list(map(Path,paths)), Path(destination)
    if len(paths) != 9 or len(set(paths)) != 9 or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('NINE_ORIGINAL_LOG_FILES_REQUIRED')
    if any(destination.resolve().is_relative_to(p.parent.resolve()) for p in paths):
        raise ValueError('ARCHIVE_MUST_BE_OUTSIDE_SERVICE_TREE')
    destination.parent.mkdir(parents=True,exist_ok=True)
    hashes = [file_sha(p) for p in paths]
    with destination.open('xb') as output, tarfile.open(fileobj=output,mode='w') as archive:
        for index,path in enumerate(paths):
            archive.add(path,arcname=f'{index:02d}/'+path.name,recursive=False)
    with tarfile.open(destination) as archive:
        actual = [hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers()]
    if actual != hashes or [file_sha(p) for p in paths] != hashes:
        raise ValueError('LEGACY_ARCHIVE_VERIFICATION_FAILED')
    return {'files':9,'sha256':file_sha(destination),'original_sha256':hashes}


def clean_import(source, destination, files):
    source, destination = Path(source), Path(destination)
    if destination.exists() or not files or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError('NEW_SEPARATE_SERVICE_DISK_REQUIRED')
    for relative, expected in files.items():
        path = source/relative
        if (not path.resolve().is_relative_to(source.resolve()) or path.is_symlink()
                or any(p in {'runtime','sessions','periods','recoveries','backups'} for p in Path(relative).parts)
                or file_sha(path) != expected):
            raise ValueError('CLEAN_DISK_ALLOWLIST_REJECTED')
    destination.mkdir(parents=True,exist_ok=False)
    for relative, expected in files.items():
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,target)
        if file_sha(target) != expected:
            raise ValueError('CLEAN_DISK_COPY_HASH_DIFFERS')
    return {'status':'CLEAN_IMPORT_VERIFIED','files':len(files)}


def bucket_sources(storage):
    """Every live generation; soft-deleted objects must also be accounted for."""
    for obj in storage.objects('',versions=True):
        yield obj['name']+'#'+str(obj['generation']), [storage.read(obj['name'],obj['generation'])]
    soft = storage.objects('',soft_deleted=True)
    if soft:
        # Restoring soft-deleted objects would mutate the bucket. Do not infer zero.
        raise ValueError('SOFT_DELETED_CONTENT_REQUIRES_REVIEW')


def host_sources(active, *, invoke=None):
    """Discover effective writable mounts and upper layers of all retained containers."""
    import json
    import subprocess
    invoke = invoke or subprocess.run
    def command(argv):
        result = invoke(argv,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=60)
        if result.returncode:
            raise ValueError('PRIVACY_SOURCE_COMMAND_FAILED')
        return result.stdout
    def containers(domain):
        names = command(['docker','ps','-a','--format','{{.Names}}']).decode().splitlines()
        prefixes = {'caddy':('cloudrag-i4-caddy-',),'app':('cloudrag-i4-app-', 'cloudrag-i4-maintenance-'),
                    'runner':('cloudrag-i4-ollama-','cloudrag-i5-stimulus-','cloudrag-i7-gate-')}[domain]
        for name in names:
            if not name.startswith(prefixes):
                continue
            inspected = json.loads(command(['docker','inspect',name]))[0]
            roots = [m['Source'] for m in inspected['Mounts'] if m.get('RW')]
            upper = inspected.get('GraphDriver',{}).get('Data',{}).get('UpperDir')
            if not upper:
                raise ValueError('CONTAINER_WRITABLE_LAYER_UNRESOLVED')
            roots.append(upper)
            log = inspected.get('LogPath')
            for root in sorted(set(roots)):
                yield from tree_sources(root)
            if log:
                with Path(log).open('rb') as stream:
                    yield log,chunks(stream)
    def journal():
        # No boot/date filter: persistent records from this clean service disk count too.
        yield 'persistent-journal',[command(['journalctl','--no-pager','--output=json','--all'])]
    def backups():
        yield from tree_sources(Path(active['boot_root']).parent.parent)
        # An unexpected inherited backup tree on the service disk is scanned, not excluded.
        previous = Path('/srv/cloudrag/iteration6-cowork-backups')
        if previous.exists():
            yield from tree_sources(previous)
    return {**{domain:lambda d=domain:containers(d) for domain in ('caddy','app','runner')},
            'journald':journal,'backups':backups}
