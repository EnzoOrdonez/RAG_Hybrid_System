"""Honest clock census and complete package manifests; no repairs of old evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from scripts.study_operator.run_control import require_limited, utc

MANIFEST = 'MANIFEST_SHA256.jsonl'


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return dict(lower_bound_seconds=sum((b-a).total_seconds() for a, b in merged),
                intervals=[dict(start=a.isoformat(), end=b.isoformat()) for a, b in merged])


def census(rows):
    phases, anomalies, accepted = {}, [], []
    for source, row in rows:
        try:
            start = datetime.fromisoformat(row.get('started_utc') or row['start'])
            end = datetime.fromisoformat(row.get('ended_utc') or row['end'])
            if start.utcoffset() is None or end.utcoffset() is None or end < start:
                raise ValueError('Invalid or inverted UTC interval')
        except (ValueError, KeyError, TypeError) as exc:
            anomalies.append(dict(source=source, reason=type(exc).__name__, original=row,
                                  duration_not_locatable=row.get('duration_s'), excluded_from_union=True))
            continue
        phase = str(row.get('phase', 'UNASSIGNED'))
        phases.setdefault(phase, []).append((start, end))
        accepted.append(dict(source=source, phase=phase, start=start.isoformat(), end=end.isoformat()))
    return dict(phases={p: union(v) for p, v in phases.items()}, clock_anomalies=anomalies,
                rows=accepted, global_union=union([v for values in phases.values() for v in values]),
                lower_bound_only=True, reconstructed_timestamps=False)


def linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0)
                                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def paths(root):
    result = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            if linked(Path(directory) / name):
                raise ValueError('Linked/reparse path prevents complete package inventory')
        result.extend(Path(directory) / name for name in files)
    return sorted(result)


def hash_row(root, path):
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Changed during hashing; no final manifest')
    return dict(path=path.relative_to(root).as_posix(), sha256=sha, size_bytes=after.st_size)


def inventory(root, staging, *, workers=4, seconds=1200):
    root, staging = Path(root).resolve(), Path(staging).resolve()
    if staging.is_relative_to(root) or (root / MANIFEST).exists():
        raise ValueError('External new staging and unsealed package required')
    deadline = time.monotonic() + seconds
    originals = paths(root)
    count, size = 0, 0
    with staging.open('x', encoding='utf-8', newline='\n') as out, ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded queue also supports the Linux image's Python version.
        pending = deque()
        remaining = iter(originals)
        for _ in range(64):
            path = next(remaining, None)
            if path is None:
                break
            pending.append(pool.submit(hash_row, root, path))
        while pending:
            row = pending.popleft().result()
            path = next(remaining, None)
            if path is not None:
                pending.append(pool.submit(hash_row, root, path))
            if time.monotonic() >= deadline:
                raise TimeoutError('Hash limit exceeded; partial staging preserved')
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            size += row['size_bytes']
        out.flush()
        os.fsync(out.fileno())
    if paths(root) != originals:
        raise RuntimeError('Package membership changed while hashing')
    with staging.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(files=count, bytes=size, manifest_sha256=sha)


def seal(root, receipt, verifier, *, seconds=1200):
    root, receipt, verifier = map(lambda p: Path(p).resolve(), (root, receipt, verifier))
    if receipt.is_relative_to(root) or receipt.exists():
        raise ValueError('New external receipt required to avoid circular inventory')
    start, begin = utc(), time.monotonic()
    staging, verification = receipt.with_suffix('.manifest.pending'), receipt.with_suffix('.verification.json')
    result = inventory(root, staging, seconds=seconds)
    manifest = root / MANIFEST
    os.replace(staging, manifest)
    # This independent verifier never modifies the inventory or audited directory.
    child = subprocess.run([sys.executable, '-B', str(verifier), '--root', str(root),
                            '--expected-manifest-sha256', result['manifest_sha256'],
                            '--output', str(verification)], capture_output=True, timeout=1500,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1'))
    proof = json.loads(verification.read_bytes()) if verification.exists() else {}
    passed = child.returncode == 0 and proof.get('status') == 'PASS'
    result.update(status='SEALED_AND_EXTERNALLY_VERIFIED' if passed else 'EXTERNAL_VERIFICATION_FAILED',
                  started_utc=start, ended_utc=utc(), duration_s=time.monotonic()-begin,
                  root=str(root), verification_receipt=str(verification), verifier_exit_code=child.returncode,
                  verifier_source_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest(),
                  inventory_self_excluded_only=True, no_root_writes_after_manifest=True,
                  not_participant_acceptance=True)
    with receipt.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
    return result


def diagnose_existing(root, failed_stderr, verifier, output, *, sample=256):
    """Pin the already-created inventory from preserved failed command evidence."""
    root, failed_stderr, verifier, output = map(Path, (root, failed_stderr, verifier, output))
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Diagnostic must stay outside audited directory')
    pins = re.findall(r"'--expected-manifest-sha256', '([0-9a-f]{64})'", failed_stderr.read_text(encoding='utf-8'))
    if len(pins) != 1:
        raise ValueError('Exactly one preserved failed-command inventory pin required')
    manifest = root / MANIFEST
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_sha != pins[0]:
        raise ValueError('Existing manifest changed since failed command')
    count, size, rows = 0, 0, []
    with manifest.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            size += row['size_bytes']
            if len(rows) < sample:
                rows.append(row)
    begin = time.monotonic()
    for row in rows:
        target = root / row['path']
        target.resolve()
        target.resolve()
        hash_row(root, target)
    elapsed = time.monotonic() - begin
    result = dict(manifest_sha256=manifest_sha, files=count, bytes=size, sample_files=len(rows),
                  sample_resolve_twice_and_hash_seconds=elapsed,
                  sample_projection_seconds=elapsed * count / len(rows) if rows else None,
                  projection_not_a_completion_guarantee=True, failed_stderr_sha256=hashlib.sha256(failed_stderr.read_bytes()).hexdigest(),
                  verifier_source_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest(),
                  root=str(root.resolve()), at=utc(), audited_directory_modified=False)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


def verify_existing(root, receipt, verifier, pin_receipt, *, seconds=3000):
    """One bounded read-only retry; preserve the original manifest and failures."""
    root, receipt, verifier, pin_receipt = map(lambda p: Path(p).resolve(), (root, receipt, verifier, pin_receipt))
    verification = receipt.with_suffix('.verification.json')
    if receipt.is_relative_to(root) or receipt.exists() or verification.exists():
        raise ValueError('New external receipts required')
    if not 0 < seconds <= 3000:
        raise ValueError('Finite verification retry bound required')
    pin = json.loads(pin_receipt.read_bytes())
    manifest = root / MANIFEST
    if (str(root) != pin['root'] or hashlib.sha256(manifest.read_bytes()).hexdigest() != pin['manifest_sha256']
            or hashlib.sha256(verifier.read_bytes()).hexdigest() != pin['verifier_source_sha256']):
        raise ValueError('Pinned manifest or unchanged external verifier differs')
    start, begin = utc(), time.monotonic()
    args = [sys.executable, '-B', str(verifier), '--root', str(root),
            '--expected-manifest-sha256', pin['manifest_sha256'], '--output', str(verification)]
    child = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1'))
    timed_out = False
    try:
        _, stderr = child.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=30, check=True)
        else:
            child.kill()
        _, stderr = child.communicate(timeout=30)
    proof = json.loads(verification.read_bytes()) if verification.exists() else {}
    passed = not timed_out and child.returncode == 0 and proof.get('status') == 'PASS'
    result = dict(status='SEALED_AND_EXTERNALLY_VERIFIED' if passed else 'TIMEOUT_PRESERVED' if timed_out else 'EXTERNAL_VERIFICATION_FAILED',
                  started_utc=start, ended_utc=utc(), duration_s=time.monotonic()-begin,
                  root=str(root), manifest_sha256=pin['manifest_sha256'], verifier_source_sha256=pin['verifier_source_sha256'],
                  verification_receipt=str(verification), verifier_exit_code=child.returncode,
                  verifier_stderr=stderr.decode('utf-8', errors='replace'), exact_command=args,
                  native_child_limit_seconds=seconds, existing_inventory_reused=True,
                  audited_directory_modified=False, not_participant_acceptance=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--verifier', required=True)
    parser.add_argument('--existing-pin')
    args = parser.parse_args(argv)
    require_limited()
    result = (verify_existing(args.root, args.receipt, args.verifier, args.existing_pin)
              if args.existing_pin else seal(args.root, args.receipt, args.verifier))
    print(json.dumps(result))
    return 0 if result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
