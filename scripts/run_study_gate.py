"""Two-window operational study gate.  The human launches real mode; dry runs are synthetic only."""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ui.components.session_storage import atomic_json
from src.ui.components.study_protocol import digest

TASKS = ('q001', 'q064', 'q171', 'q010', 'q070', 'q172')
WINDOW_SECONDS = 120 * 60


def schedule(window):
    if window not in (1, 2): raise ValueError('window must be 1 or 2')
    repeats = range(1, 6) if window == 1 else range(6, 11)
    return [dict(window=window, repetition=r, query_id=query, condition=condition)
            for r in repeats for pos, query in enumerate(TASKS)
            for condition in (('hybrid', 'no_rag') if (r + pos) % 2 else ('no_rag', 'hybrid'))]


def _preflight(operator_zoom_active, screen_share_declared, backup_pending, synthetic):
    result = dict(zoom_process_declared=operator_zoom_active, screen_share_declared=screen_share_declared,
                  backup_pending=backup_pending, synthetic=synthetic)
    result['valid'] = bool(operator_zoom_active and screen_share_declared and not backup_pending)
    if not result['valid']: raise ValueError('Preflight rejected: declare active Zoom, shared app window, and no pending backup')
    return result


def run(output, *, window, dry_run=False, operator_zoom_active=False, screen_share_declared=False,
        backup_pending=False, adapter=None, monotonic=time.monotonic):
    """Run exactly one planned window. No resumption after an interrupted attempt."""
    root = Path(output).resolve()
    if root.exists(): raise FileExistsError('Evidence directory must be new')
    root.mkdir(parents=True)
    preflight = _preflight(operator_zoom_active, screen_share_declared, backup_pending, dry_run)
    atomic_json(root / 'preflight.json', preflight)
    rows, started = [], monotonic()
    terminal = None
    for index, planned in enumerate(schedule(window), 1):
        if monotonic() - started > WINDOW_SECONDS:
            terminal = 'WINDOW_EXPIRED'; break
        attempt_started = monotonic()
        try:
            if dry_run:
                elapsed, valid, error = .001, True, None
            else:
                if adapter is None: raise RuntimeError('Real adapter required')
                elapsed, valid, error = adapter(planned)
            row = dict(planned, status='success' if error is None else 'error', valid=bool(valid), error=error,
                       elapsed_s=elapsed if error is None else None)
        except BaseException as exc:
            row = dict(planned, status='aborted', valid=False, error=type(exc).__name__, elapsed_s=None)
            rows.append(row); terminal = 'INTERRUPTED_TERMINAL'; break
        rows.append(row)
        print(f'{index}/60 {planned["condition"]} ETA={(60-index)*max(monotonic()-attempt_started, 0):.1f}s')
        if row['status'] != 'success' or not row['valid']:
            terminal = 'INVALID_TERMINAL'; break
    status = terminal or ('complete' if len(rows) == 60 else 'incomplete')
    atomic_json(root / 'attempts.json', rows)
    systems = {}
    for condition in ('hybrid', 'no_rag'):
        group = [r for r in rows if r['condition'] == condition]
        values = [r['elapsed_s'] for r in group if r['status'] == 'success' and r['valid']]
        systems[condition] = dict(n=len(values), failures=sum(r['status'] != 'success' for r in group),
                                  invalid=sum(not r['valid'] for r in group), p95=float(np.percentile(values, 95)) if values else None)
    summary = dict(mode='SYNTHETIC_ONLY' if dry_run else 'REAL_NOT_A_GO_UNTIL_AGGREGATED', window=window,
                   status=status, started_at=datetime.now(timezone.utc).isoformat(), systems=systems,
                   go_decision='NOT_A_GO_DECISION')
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'manifest.json', dict(files={n: digest(root / n) for n in ('preflight.json', 'attempts.json', 'summary.json')}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--zoom-active', action='store_true'); parser.add_argument('--screen-share-declared', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output, window=args.window, dry_run=args.dry_run, operator_zoom_active=args.zoom_active, screen_share_declared=args.screen_share_declared), indent=2))

if __name__ == '__main__': main()
