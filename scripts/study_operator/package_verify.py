"""Standalone external package verification with bounded hashing and two censuses.

Only the named manifest excludes itself. No writes or repairs inside the package.
Directory/file reparse checks replace repeated Windows realpath resolution.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

MANIFEST = 'MANIFEST_SHA256.jsonl'


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def plain(info):
    return not (getattr(info, 'st_file_attributes', 0) & 0x400) and not stat.S_ISLNK(info.st_mode)


def scan(root):
    files, directories, pending = {}, {}, [root]
    while pending:
        folder = pending.pop()
        info = folder.lstat()
        if not plain(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Linked or invalid directory')
        directories[folder.relative_to(root).as_posix()] = fingerprint(info)
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if not plain(info):
                    raise ValueError('Linked/reparse inventory member')
                target = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(target)
                elif stat.S_ISREG(info.st_mode):
                    # Windows DirEntry.stat may omit device/inode; compare real
                    # lstat identities with the opened descriptor, never zeros.
                    info = target.lstat()
                    if not plain(info) or not stat.S_ISREG(info.st_mode):
                        raise ValueError('Inventory member changed while listing')
                    relative = target.relative_to(root).as_posix()
                    if relative != MANIFEST:
                        files[relative] = fingerprint(info)
                else:
                    raise ValueError('Non-regular inventory member')
    return files, directories


def hash_one(root, name, expected, original):
    target = root.joinpath(*PurePosixPath(name).parts)
    before = target.lstat()
    if not plain(before) or fingerprint(before) != original:
        return dict(path=name, reason='CHANGED_BEFORE_HASH')
    with target.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = target.lstat()
    if not plain(after) or fingerprint(after) != original or fingerprint(opened) != original:
        return dict(path=name, reason='CHANGED_DURING_HASH')
    if digest != expected['sha256'] or after.st_size != expected['size_bytes']:
        return dict(path=name, reason='CONTENT_OR_SIZE_MISMATCH')
    return None


def verify(root, expected_manifest_sha256, *, seconds=1200, workers=4):
    root = Path(root).resolve()
    manifest = root/MANIFEST
    started = datetime.now(timezone.utc).isoformat()
    begin, deadline = time.monotonic(), time.monotonic()+seconds
    if not re.fullmatch('[a-f0-9]{64}', expected_manifest_sha256) or not 0 < seconds <= 3600:
        raise ValueError('Pinned SHA-256 and finite verification bound required')
    if not plain(manifest.lstat()):
        raise ValueError('Manifest reparse point')
    with manifest.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_manifest_sha256:
        raise ValueError('Manifest changed; never repin or repair')
    expected = {}
    with manifest.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            name = row['path']
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                    or name != path.as_posix() or name in ('', '.', MANIFEST) or name in expected
                    or not re.fullmatch('[a-f0-9]{64}', row['sha256'])
                    or type(row['size_bytes']) is not int or row['size_bytes'] < 0):
                raise ValueError('Unsafe, duplicate or invalid manifest row')
            expected[name] = row
    before, directories = scan(root)
    missing, added = sorted(set(expected)-set(before)), sorted(set(before)-set(expected))
    errors, checked, pending = [], 0, deque()
    names = iter(sorted(set(expected) & set(before)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(64):
            name = next(names, None)
            if name is None:
                break
            pending.append(pool.submit(hash_one, root, name, expected[name], before[name]))
        while pending:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('External verification bound reached')
            error = pending.popleft().result(timeout=remaining)
            checked += 1
            if error:
                errors.append(error)
            name = next(names, None)
            if name is not None:
                pending.append(pool.submit(hash_one, root, name, expected[name], before[name]))
    after, after_directories = scan(root)
    membership_changed = before != after or directories != after_directories
    with manifest.open('rb') as stream:
        manifest_changed = hashlib.file_digest(stream, 'sha256').hexdigest() != actual
    passed = not (missing or added or errors or membership_changed or manifest_changed) and checked == len(expected)
    return dict(status='PASS' if passed else 'FAIL', started_utc=started,
        ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-begin,
        root=str(root), manifest_sha256=actual, expected_files=len(expected), checked_files=checked,
        missing=missing, added=added, errors=errors, links=[], two_censuses_equal=not membership_changed,
        manifest_changed_during_verification=manifest_changed, audited_directory_modified=False,
        inventory_modified=False, manifest_self_excluded_only=True, integrity_only_not_study_verdict=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('New external verification receipt required')
    result = verify(root, args.expected_manifest_sha256, seconds=args.seconds)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], checked_files=result['checked_files'])))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
