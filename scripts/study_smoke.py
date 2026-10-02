"""Human-driven local P999 smoke. No browser automation, model changes or gate GO."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from importlib.metadata import version
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.decline_classifier import CLASSIFIER_VERSION, classify_response  # noqa: E402
from src.evaluation.study_analysis import read_exports  # noqa: E402
from src.ui.components.session_storage import atomic_json, read_json  # noqa: E402
from src.ui.components.study_backup import backup_export  # noqa: E402
from src.ui.components.study_protocol import CELLS, LIKERT_IDS, ROOT, digest, sus_score, verify_draw  # noqa: E402
from src.ui.components.study_sessions import StudyStore  # noqa: E402

MARKER = 'SMOKE_NOT_GATE'


def disk_inventory():
    command = ("$ErrorActionPreference='Stop'; "
               "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); "
               "Get-Partition | ForEach-Object { $d=Get-Disk -Number $_.DiskNumber; "
               "[pscustomobject]@{DiskNumber=$_.DiskNumber;AccessPaths=$_.AccessPaths;"
               "BusType=[string]$d.BusType} } | ConvertTo-Json -Depth 4 -Compress")
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, encoding='utf-8', check=True, timeout=15)
    records = json.loads(result.stdout)
    return records if isinstance(records, list) else [records]


def disk_number(path, inventory):
    """Longest mounted access path wins; drive letters alone are not disk identity."""
    path = PureWindowsPath(str(path))
    matches = []
    for record in inventory:
        for mount in record.get('AccessPaths') or []:
            prefix = PureWindowsPath(mount)
            if path.is_relative_to(prefix):
                matches.append((len(prefix.parts), record['DiskNumber']))
    if not matches:
        raise ValueError('Cannot resolve physical disk; do not assume a different drive is enough')
    depth = max(n for n, _ in matches)
    disks = {d for n, d in matches if n == depth}
    if len(disks) != 1 or not all(isinstance(d, int) for d in disks):
        raise ValueError('Ambiguous physical disk mapping')
    return disks.pop()


def same_physical_disk(left, right):
    inventory = disk_inventory()
    disks = [disk_number(left, inventory), disk_number(right, inventory)]
    # Reject virtual/storage-pool/network devices: distinct virtual disk numbers
    # need not mean independent physical media.
    direct_buses = {'SATA', 'ATA', 'USB', 'NVMe', 'SAS', 'SD', 'MMC'}
    if any(r.get('BusType') not in direct_buses for r in inventory if r['DiskNumber'] in disks):
        raise ValueError('Physical media independence is not established for this storage topology')
    return disks[0] == disks[1]


def plan(config_dir, root, backup):
    protocol = verify_draw(config_dir)
    root, backup = Path(root).resolve(), Path(backup).resolve()
    if (root.is_relative_to(ROOT.parent.parent) or backup.is_relative_to(ROOT.parent.parent)
            or root.is_relative_to(backup) or backup.is_relative_to(root)):
        raise ValueError('Use separate external session and backup directories')
    return dict(marker=MARKER, mode='HUMAN_UI_REAL_MODELS', root=str(root), backup=str(backup),
                config_dir=str(Path(config_dir).resolve()), fingerprint=protocol['fingerprint'],
                synthetic_instruments=True, real_smoke='NOT_RUN', physical_disk='NOT_CHECKED')


def validate_export(row, protocol):
    if (row.get('schema_version') != 3 or row.get('purpose') != 'smoke'
            or row.get('stage') != 'complete' or row.get('gate_marker') != MARKER
            or row.get('analysis_excluded') is not True
            or row.get('instrument_responses_synthetic') is not True
            or row['assignment']['participant_id'] != 'P999'
            or row['protocol_fingerprint'] != protocol['fingerprint']
            or row['labels'] != protocol['config']['labels']):
        raise ValueError('Not a complete identified P999 smoke export')
    attempts = row['attempts']
    if len(attempts) != 8 or len({a['attempt_id'] for a in attempts}) != 8:
        raise ValueError('Smoke requires exactly eight unique responses, no retries')
    for index, (label, task_set) in enumerate(CELLS[row['assignment']['cell']]):
        group = [a for a in attempts if a['block_index'] == index]
        tasks = [a for a in group if a['analysis_role'] == 'tasks']
        free = [a for a in group if a['analysis_role'] == 'free_query']
        if (len(group) != 4 or len(free) != 1
                or [a['query_id'] for a in tasks] != protocol['config']['tasks'][task_set]):
            raise ValueError('Wrong tasks/free query count or order')
        for attempt in group:
            if (attempt['status'] != 'acknowledged' or attempt['error'] is not None
                    or not attempt['shown_at'] or not attempt['answer']
                    or attempt['condition'] != row['labels'][label] or attempt['label'] != label
                    or attempt['decline_classifier_version'] != CLASSIFIER_VERSION
                    or attempt['decline_class'] != classify_response(attempt['answer'])):
                raise ValueError('Invalid successful response or v2 metadata')
    blocks = row['instruments']
    if len(blocks) != 2 or {b['block_index'] for b in blocks} != {0, 1}:
        raise ValueError('Two instrument blocks required')
    for block in blocks:
        if (len(block['sus']) != 10 or sus_score(block['sus']) != block['sus_score']
                or set(block['likert']) != set(LIKERT_IDS)
                or any(v not in (1, 2, 3, 4, 5) for v in block['likert'].values())):
            raise ValueError('Incomplete SUS/Likert')
    if (set(row['comparative'] or {}) != {'C1', 'C2', 'C3', 'C4'}
            or row['blinding']['choice'] not in protocol['config']['blinding_choices']):
        raise ValueError('Comparative/blinding incomplete')
    if (len(row['events']) != 2 or row['incidents']
            or {e['block_index'] for e in row['events']} != {0, 1}
            or any(set(e) != {'kind', 'timestamp', 'block_index'}
                   or e['kind'] != 'familiarization_done' for e in row['events'])
            or protocol['config']['familiarization'] in json.dumps(row, ensure_ascii=False)):
        raise ValueError('Practice privacy structure or clean-smoke criterion violated')
    return dict(tasks=6, free_queries=2, SUS=2, Likert=2, comparative=4, blinding=1,
                classes=dict(Counter(a['decline_class'] for a in attempts)))


def verify(root, config_dir, backup):
    planned = plan(config_dir, root, backup)
    root = Path(root)
    receipt = read_json(root / 'smoke_launch.json')
    if receipt.get('status') == 'FAILED_PRESERVE_EVIDENCE' and receipt.get('phase') != 'VERIFY_EXPORT_BACKUP':
        raise ValueError('Terminal smoke failure; diagnosis required, do not relabel it as passed')
    for key in ('marker', 'root', 'backup', 'config_dir', 'fingerprint', 'synthetic_instruments'):
        if receipt.get(key) != planned[key]:
            raise ValueError('Smoke launch identity changed')
    paths = list(root.glob('*/full_session.json'))
    if len(paths) != 1:
        raise ValueError('Exactly one closed P999 export required')
    rows, _ = read_exports(paths)
    row = rows[0]
    if row['build_id'] != receipt['build_id']:
        raise ValueError('Smoke build changed')
    counts = validate_export(row, verify_draw(config_dir))
    state = backup_export(paths[0].parent, backup, same_physical_disk=same_physical_disk)
    if digest(Path(state['destination']) / 'export_manifest.json') != digest(paths[0].parent / 'export_manifest.json'):
        raise ValueError('Backup export manifest mismatch')
    result = dict(marker=MARKER, status='VERIFIED_EXPORT_AND_BACKUP', counts=counts,
                  sha256=digest(paths[0]), backup=state, latencies='DESCRIPTIVE_ONLY',
                  privacy='STRUCTURAL_CHECK_PLUS_PRACTICE_REGRESSION; not a whole-computer audit')
    atomic_json(root / 'smoke_verification.json', result)
    return result


def monitor(process, root, seconds, *, clock=time.monotonic, sleep=time.sleep):
    started, previous = clock(), None
    while clock() - started < seconds:
        if process.poll() is not None:
            raise RuntimeError('Owned app exited before export completion')
        checkpoints = list(Path(root).glob('*/study_checkpoint.json'))
        if checkpoints:
            row = read_json(checkpoints[0])
            if row['incidents'] or any(a['status'] == 'error' for a in row['attempts']):
                raise RuntimeError('Smoke failed; diagnose before any retry')
            progress = (row['stage'], sum(a['status'] == 'acknowledged' for a in row['attempts']))
            if progress != previous:
                print(f'{progress[1]}/8 responses; stage={progress[0]}; human pace, ETA unknown', flush=True)
                previous = progress
            if row['stage'] == 'complete' and (checkpoints[0].parent / 'export_manifest.json').exists():
                return
        sleep(1)
    raise TimeoutError('Human smoke deadline reached; evidence preserved, no automatic retry')


def stop_owned(process):
    if process.poll() is None:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, check=True, timeout=15)


def launch(args):
    from scripts.gate_job import enter
    from src.utils.deployment_artifacts import verify_manifest

    planned = plan(args.config_dir, args.root, args.backup)
    root, backup = Path(planned['root']), Path(planned['backup'])
    if root.exists():
        raise FileExistsError('Smoke directory already exists; preserve it, use verify after closure')
    if (not args.operator_checks_confirmed or not args.artifact_manifest or not args.model_digest
            or len(args.model_digest.removeprefix('sha256:')) != 64
            or not 1 <= args.max_minutes <= 120):
        raise ValueError('Operator checklist, trusted manifest, digest and bounded duration required')
    int(args.model_digest.removeprefix('sha256:'), 16)
    if (os.name != 'nt' or sys.version_info[:3] != (3, 14, 3)
            or Path(sys.executable).resolve() != (ROOT / '.venv-app/Scripts/python.exe').resolve()
            or version('torch') != '2.10.0+cpu'):
        raise RuntimeError('Use the frozen Windows Python 3.14.3 environment')
    if not 1024 <= args.port <= 65535:
        raise ValueError('Invalid local app port')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise RuntimeError('Commit and verify the worktree before real smoke')
    verify_manifest(ROOT, Path(args.artifact_manifest))
    if same_physical_disk(root, backup):
        raise ValueError('A separate physical backup disk is required')
    backup.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=backup) as probe:
        probe.write(b'backup-write-probe')
        probe.flush()
        os.fsync(probe.fileno())
    build = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    store = StudyStore(root, verify_draw(args.config_dir), 'smoke')
    store.freeze()
    token = store.issue('P999', cell=1, profile='without_experience')
    receipt = dict(planned, build_id=build, started_at=datetime.now(timezone.utc).isoformat(),
                   status='STARTED', phase='HUMAN_UI', max_minutes=args.max_minutes,
                   physical_disk='VERIFIED_DISTINCT')
    atomic_json(root / 'smoke_launch.json', receipt)
    env = dict(os.environ, CLOUDRAG_MODE='participant', CLOUDRAG_STUDY_PURPOSE='smoke',
               CLOUDRAG_STUDY_CONFIG=str(Path(args.config_dir).resolve() / 'study.json'),
               CLOUDRAG_STUDY_ASSIGNMENTS=str(Path(args.config_dir).resolve() / 'assignments.csv'),
               CLOUDRAG_STUDY_SESSION_DIR=str(root), CLOUDRAG_BUILD_ID=build,
               CLOUDRAG_ARTIFACT_MANIFEST=str(Path(args.artifact_manifest).resolve()),
               CLOUDRAG_MODEL_DIGEST=args.model_digest.removeprefix('sha256:'),
               OLLAMA_HOST='http://127.0.0.1:11434', CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONHASHSEED='42', PYTHONUTF8='1')
    enter()  # Kill-on-close job owns only this launcher and its descendants, not Ollama.
    command = [sys.executable, '-m', 'streamlit', 'run', 'src/ui/app.py', '--server.address', '127.0.0.1',
               '--server.port', str(args.port), '--server.headless', 'true',
               '--server.enableCORS', 'true', '--server.enableXsrfProtection', 'true']
    with (root / 'app.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        print(f'{MARKER}: open http://127.0.0.1:{args.port} in the designated browser.\n'
              f'Invitation (do not log/share): {token}\nSynthetic instruments only. No retry after failure.', flush=True)
        try:
            monitor(process, root, args.max_minutes * 60)
            receipt['phase'] = 'VERIFY_EXPORT_BACKUP'
            result = verify(root, args.config_dir, backup)
            receipt['status'] = result['status']
        except BaseException as exc:
            receipt.update(status='FAILED_PRESERVE_EVIDENCE', error=type(exc).__name__)
            raise
        finally:
            atomic_json(root / 'smoke_launch.json', receipt)
            stop_owned(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'launch', 'verify'])
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--backup', required=True)
    parser.add_argument('--artifact-manifest')
    parser.add_argument('--model-digest')
    parser.add_argument('--operator-checks-confirmed', action='store_true')
    parser.add_argument('--max-minutes', type=int, default=45)
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args(argv)
    if args.command == 'launch':
        launch(args)
    elif args.command == 'verify':
        print(json.dumps(verify(args.root, args.config_dir, args.backup), indent=2))
    else:
        print(json.dumps(dict(plan(args.config_dir, args.root, args.backup), mode='DRY_PLAN_ONLY'), indent=2))


if __name__ == '__main__':
    main()
