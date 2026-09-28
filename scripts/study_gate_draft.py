"""DRAFT study gate orchestrator. Synthetic ONLY; no real model/network adapter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ui.components.session_storage import atomic_json, read_json
from src.ui.components.study_protocol import INVALID_TASKS, ROOT, digest

TASKS = ('q001', 'q064', 'q171', 'q010', 'q070', 'q172')


def schedule(tasks=TASKS):
    if len(tasks) != 6 or len(set(tasks)) != 6 or set(tasks) & INVALID_TASKS:
        raise ValueError('Six distinct eligible task IDs required')
    return [dict(position=i*6+j, query_id=q, condition=condition)
            for i in range(10) for j, q in enumerate(tasks)
            for condition in (('hybrid', 'no_rag') if (i+j) % 2 == 0 else ('no_rag', 'hybrid'))]


def verify(root):
    manifest = read_json(Path(root) / 'manifest.json')
    for name, sha in manifest['files'].items():
        path = (Path(root) / name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or digest(path) != sha:
            raise ValueError('Package hash mismatch')
    return manifest


def publish(root, state):
    attempts = [read_json(root / name) for name in state['files']]
    result = {}
    for condition in ('hybrid', 'no_rag'):
        group = [a for a in attempts if a['condition'] == condition]
        valid = [a['elapsed_s'] for a in group if a['status'] == 'success' and a['valid']]
        result[condition] = dict(n=len(valid), p50=float(np.percentile(valid, 50)) if valid else None,
            p95=float(np.percentile(valid, 95)) if valid else None,
            failures=sum(a['status'] != 'success' for a in group), invalid=sum(not a['valid'] for a in group))
    summary = dict(mode='SYNTHETIC_ONLY', verdict='NOT_A_GO_DECISION', status=state['status'],
                   completed=len(attempts), planned=len(state['schedule']), systems=result)
    atomic_json(root / 'summary.json', summary)
    files = [root / name for name in state['files']] + [root / 'checkpoint.json', root / 'summary.json', root / 'preflight.json']
    atomic_json(root / 'manifest.json', dict(mode='SYNTHETIC_ONLY', files={p.name: digest(p) for p in files}))
    return summary


def run(root, *, resume=False, authorize_new_window=False, contaminated=False, clock=time.perf_counter,
        callback=None, tasks=TASKS, invalid_at=None, stop_after=None, wall=time.time):
    """No inference adapter permitted here. Callback is a synthetic test seam only.

    Clock includes durable request preparation and callback; excludes final result
    flush. A hard interruption leaves a pending marker, terminal on resumption.
    An orderly stop between complete task pairs permits explicit new-window resume.
    """
    root = Path(root).resolve()
    if root.is_relative_to(ROOT.parent.parent):
        raise ValueError('Evidence must be outside checkout')
    if resume:
        if not authorize_new_window:
            raise ValueError('Resume requires explicit new-window authorization')
        state = read_json(root / 'checkpoint.json')
        if state['mode'] != 'SYNTHETIC_ONLY' or state['schedule'] != schedule(tasks):
            raise ValueError('Cohort identity changed')
        for name, sha in state['hashes'].items():
            if digest(root / name) != sha:
                raise ValueError('Attempt changed before resume')
        if state['pending'] is not None or state['status'] in ('expired', 'interrupted') or wall() > state['deadline']:
            state['status'] = 'interrupted' if state['pending'] is not None else 'expired'
            if state['pending'] is not None:
                index = state['pending']
                name = f'aborted-{index:03}.json'
                atomic_json(root / name, dict(state['schedule'][index], status='aborted',
                    valid=False, elapsed_s=None, error='INCOMPLETO_INTERRUMPIDO'))
                state['files'].append(name)
                state['hashes'][name] = digest(root / name)
                state['pending'] = None
            atomic_json(root / 'checkpoint.json', state)
            return publish(root, state)  # no inference, no imputation
        state['deadline'] = wall() + 7200
    else:
        if root.exists():
            raise ValueError('New package must not already exist')
        root.mkdir(parents=True)
        state = dict(mode='SYNTHETIC_ONLY', status='running', schedule=schedule(tasks), pending=None,
                     files=[], hashes={}, deadline=wall()+7200, windows=[])
    if contaminated or shutil.disk_usage(root).free < 10*1024*1024:
        atomic_json(root / 'preflight.json', dict(valid=False, synthetic_contamination=contaminated))
        raise ValueError('Synthetic preflight rejected; no attempts')
    state['windows'].append(dict(started_at=datetime.now(timezone.utc).isoformat(), deadline=state['deadline']))
    atomic_json(root / 'preflight.json', dict(valid=True, synthetic=True))
    state['status'] = 'running'
    atomic_json(root / 'checkpoint.json', state)
    for index in range(len(state['files']), len(state['schedule'])):
        if wall() > state['deadline']:
            state['status'] = 'expired'
            break
        if stop_after is not None and index >= stop_after and index % 2 == 0:
            state['status'] = 'paused_between_pairs'
            break
        start = clock()
        state['pending'] = index
        atomic_json(root / 'checkpoint.json', state)
        status, error = 'success', None
        try:
            if callback:
                callback(state['schedule'][index])
        except Exception as exc:
            status, error = 'error', type(exc).__name__
        elapsed = clock() - start
        row = dict(state['schedule'][index], status=status, error=error,
                   valid=index != invalid_at, elapsed_s=elapsed if status == 'success' else None)
        name = f'attempt-{index:03}.json'
        if (root / name).exists():
            raise ValueError('Never overwrite an attempt')
        atomic_json(root / name, row)
        state['files'].append(name)
        state['hashes'][name] = digest(root / name)
        state['pending'] = None
        atomic_json(root / 'checkpoint.json', state)
        print(f"SYNTHETIC intento {index+1}/{len(state['schedule'])} | {row['condition']} | {elapsed:.3f}s | ETA synthetic only")
    else:
        state['status'] = 'complete'
    atomic_json(root / 'checkpoint.json', state)
    return publish(root, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--authorize-new-window', action='store_true')
    parser.add_argument('--simulate-contamination', action='store_true')
    args = parser.parse_args(argv)
    result = run(args.output, resume=args.resume, authorize_new_window=args.authorize_new_window,
                 contaminated=args.simulate_contamination)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
