"""Paired synthetic observer contrast; no inference or adjustment of RAG times."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate
from scripts import observe_interview_gate as observe


def order(index):
    return ('bare', 'observed') if index % 2 == 0 else ('observed', 'bare')


def hash_work(buffer, iterations):
    for _ in range(iterations):
        result = hashlib.sha256(buffer).hexdigest()
    return dict(status='success', iterations=iterations, buffer_bytes=len(buffer), checksum=result)


def summarize(rows):
    import numpy as np
    if len(rows) != 10 or {r['index'] for r in rows} != set(range(10)):
        raise ValueError('Ten distinct complete pairs are required')
    for row in rows:
        if not row['valid'] or row['statuses'] != ['success', 'success']:
            raise ValueError('Failed or invalid contrast; do not omit pairs')
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('bare_s', 'observed_s')):
            raise ValueError('Invalid duration')
    ratios = np.array([(r['observed_s'] / r['bare_s'] - 1) * 100 for r in sorted(rows, key=lambda r: r['index'])])
    bootstrap = np.random.default_rng(42).choice(ratios, size=(10000, 10), replace=True)
    upper = float(np.quantile(np.median(bootstrap, axis=1), .95))
    return dict(pairs=10, median_percent=float(np.median(ratios)), upper_95_percent=upper,
                relative_percent=ratios.tolist(), passed=upper <= 5, threshold_percent=5,
                method='paired percentile bootstrap, one-sided 95%, linear quantile',
                resamples=10000, seed=42, adjusts_rag_times=False)


def execute_pairs(root, work, sampler, observer_factory=observe.Observer):
    """Same durable response clock in both arms; sensor setup/finish outside clock."""
    root = Path(root)
    pairs = []
    for index in range(10):
        arms = {}
        for arm in order(index):
            before = sampler()
            controls_before = observe.assess([before], allowed_pids={os.getpid(), os.getppid()})
            gate.write_new(root / 'checks' / f'{index:02d}-{arm}-before.json', before)
            if controls_before:
                raise RuntimeError('Contrast controls: ' + ', '.join(controls_before))
            observer = observer_factory(root / 'telemetry' / f'{index:02d}.jsonl',
                allowed_pids={os.getpid(), os.getppid()}) if arm == 'observed' else None
            row = gate.measure_attempt(root / arm,
                dict(system='synthetic', phase=arm, index=index, benchmark_only=True), work, observer=observer)
            after = sampler()
            gate.write_new(root / 'checks' / f'{index:02d}-{arm}-after.json', after)
            # Endpoint controls apply equally; only the observed arm has in-work telemetry.
            # These are endpoint checks, not samples of a continuous stream. WPR stop/decode
            # is outside the response clock and may separate them by more than 15 seconds.
            reasons = observe.assess([before, after], allowed_pids={os.getpid(), os.getppid()}, continuous=False)
            arms[arm] = dict(row, endpoint_control_reasons=reasons)
            gate.write_new(root / 'arms' / f'{index:02d}-{arm}.json', arms[arm])
            if row['status'] != 'success' or row.get('conditions_invalid') or reasons:
                raise RuntimeError('Contrast arm failed/invalid; partial evidence retained')
        pair = dict(index=index, order=order(index), bare_s=arms['bare']['elapsed_s'],
                    observed_s=arms['observed']['elapsed_s'], statuses=['success', 'success'],
                    valid=True, attempt_ids={a: arms[a]['attempt_id'] for a in arms})
        gate.write_new(root / 'pairs' / f'{index:02d}.json', pair)
        pairs.append(pair)
        print(dict(index=index, bare_s=pair['bare_s'], observed_s=pair['observed_s']), flush=True)
    return pairs


def run(root):
    root = Path(root).resolve()
    checkout = Path(gate.git('rev-parse', '--git-common-dir')).resolve().parent
    if root.is_relative_to(checkout):
        raise ValueError('Contrast evidence must be outside checkout')
    gate.write_new(root / 'protocol.json', dict(at=gate.now(), commit=gate.git('rev-parse', 'HEAD'),
        purpose='TECHNICAL SYNTHETIC OBSERVER CONTRAST; NO INFERENCE',
        source_sha256={str(p): gate.digest(p) for p in (__file__, observe.__file__, gate.__file__)},
        pairs=10, buffer_bytes=64 * 1024 * 1024, target_s=10, resamples=10000, seed=42,
        upper_one_sided_confidence=.95, threshold_percent=5, adjust_rag_times=False))
    sampler = observe.Sampler()
    before = sampler()
    gate.write_new(root / 'initial-state.json', before)
    reasons = observe.assess([before], allowed_pids={os.getpid(), os.getppid()})
    if reasons:
        gate.write_new(root / 'result.json', dict(at=gate.now(), passed=False, status='blocked', reasons=reasons))
        raise RuntimeError('Contrast precheck failed: ' + ', '.join(reasons))
    try:
        buffer = bytes(64 * 1024 * 1024)
        started = time.perf_counter()
        calibration = hash_work(buffer, 10)
        elapsed = time.perf_counter() - started
        iterations = max(1, math.ceil(10 * 10 / elapsed))
        gate.write_new(root / 'calibration.json', dict(at=gate.now(), elapsed_s=elapsed,
            iterations=iterations, calibration=calibration, buffer_sha256=hashlib.sha256(buffer).hexdigest()))
        rows = execute_pairs(root, lambda: hash_work(buffer, iterations), sampler)
        report = summarize(rows)
    except Exception as exc:
        gate.write_new(root / 'result.json', dict(at=gate.now(), passed=False, status='failed_or_invalid',
            error=f'{type(exc).__name__}: {exc}'))
        raise
    gate.write_new(root / 'result.json', dict(at=gate.now(), status='complete', **report))
    print(report, flush=True)
    if not report['passed']:
        raise RuntimeError('Observer interference exceeds accepted bound; no pilot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
