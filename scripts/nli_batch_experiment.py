"""Registered 120-position NLI experiment. Real execution is human-launched only."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate

EXPERIMENT = 'shared-nli-pairs-v1'
SYSTEMS = ('hybrid', 'lexical', 'semantic')
STRATEGIES = {'control': 'per_claim', 'candidate': 'cross_claim'}
BASE_ROOT = Path('C:/CloudRAG/diag-run-20260920T223927918Z')
BASE_MANIFEST = '329b492b8c68b2afaa4c5d42c36764c58b034d1f44a6fd85bc66b1cc1f5a32bf'


def schedule():
    slots = []
    for index in range(20):
        shift = index % 3
        for system in SYSTEMS[shift:] + SYSTEMS[:shift]:
            arms = ('control', 'candidate') if (index + SYSTEMS.index(system)) % 2 == 0 else ('candidate', 'control')
            slots.extend((system, index, arm) for arm in arms)
    return slots


def key(row):
    return row['system'], row['index'], row['arm']


def remaining(rows):
    seen = [key(r) for r in rows]
    if len(seen) != len(set(seen)) or any(k not in schedule() for k in seen):
        raise ValueError('Duplicate or foreign NLI experimental position')
    if any(r.get('phase') != 'warm' for r in rows):
        raise ValueError('Only prepared warm positions are registered')
    if set(seen) != set(schedule()[:len(seen)]):
        raise ValueError('Experimental records are not a calendar prefix')
    return [slot for slot in schedule() if slot not in set(seen)]


def register(protocol):
    from scripts.unattended_diagnostic import SOURCES, verify_package
    verify_package(BASE_ROOT)
    if gate.digest(BASE_ROOT / 'manifest.json') != BASE_MANIFEST:
        raise ValueError('Approved discovery package changed')
    original = gate.read_json(BASE_ROOT / 'cohort/source-manifest.json')['protocol']
    if protocol['queries'] != original['queries']:
        raise ValueError('Registered 20 positions changed')
    fixed = ('gpu', 'hardware', 'server', 'model', 'model_digest', 'python', 'packages', 'lock_sha256')
    if any(protocol['environment'][k] != original['environment'][k] for k in fixed):
        raise ValueError('Registered hardware/software identity changed')
    sources = gate.local_records(BASE_ROOT / 'cohort')
    if len(sources) != 40 or not all(valid(r) for r in sources):
        raise ValueError('Expected 40 valid discovery responses')
    changed = gate.git('diff', '976a379', '--name-only', '--', 'src').splitlines()
    if set(changed) - {'src/generation/hallucination_detector.py'}:
        raise ValueError('Only NLI scheduling production change is authorized')
    paths = (*SOURCES, 'scripts/nli_batch_experiment.py', 'src/generation/hallucination_detector.py')
    protocol.update(nli_experiment=EXPERIMENT, experimental_schedule=schedule(),
        diagnostic_sources={p: gate.digest(gate.PROJECT / p) for p in paths},
        quality_source=str(BASE_ROOT), quality_source_sha256=BASE_MANIFEST,
        quality_source_ids=sorted(r['attempt_id'] for r in sources),
        accepted_preregistration=gate.digest(gate.PROJECT / 'docs/NLI_SHARED_PREREGISTRATION_2026-09-20.md'))


def check_protocol(protocol):
    if (protocol.get('nli_experiment') != EXPERIMENT or
            protocol.get('experimental_schedule') != [list(s) for s in schedule()] or
            os.environ.get('CLOUDRAG_NLI_EXPERIMENT') != '1' or
            os.environ.get('CLOUDRAG_UNATTENDED') != '1'):
        raise ValueError('Unregistered NLI experimental policy/environment')
    for name, expected in protocol['diagnostic_sources'].items():
        if gate.digest(gate.PROJECT / name) != expected:
            raise ValueError('Experimental source changed: ' + name)
    if gate.digest(gate.PROJECT / 'docs/NLI_SHARED_PREREGISTRATION_2026-09-20.md') != protocol['accepted_preregistration']:
        raise ValueError('Approved preregistration changed')
    gate.recipe(protocol)
    gate.preflight(protocol)


def valid(row):
    return (row['status'] == 'success' and not row.get('conditions_invalid')
            and not row.get('environment_invalid') and not row.get('observer_errors'))


def values(row):
    trace, response = row['diagnostic_trace'], row['response']
    stages = {s['name']: s['elapsed_s'] for s in trace['stages']}
    if trace['nli_pairs_attempted'] != sum(c['pairs'] for c in trace['nli']):
        raise ValueError('NLI pair count inconsistent')
    return dict(total_s=row['elapsed_s'], retrieval_s=stages['retrieval'], reranking_s=stages['reranking'],
        generation_s=stages['generation'], nli_s=stages['hallucination_check'],
        other_s=row['elapsed_s']-sum(stages.values()), tokens_output=response['llm_response']['tokens_output'],
        claims=sum(e['count'] for e in trace['extractions']), nli_calls=trace['nli_predict_calls'],
        nli_pairs=trace['nli_pairs_attempted'], prompt_chars=sum(g['prompt_chars'] for g in trace['generation']),
        answer_chars=len(response['answer']),
        chunks=len(response['retrieved_chunks']))


def metrics(items):
    return dict(n=len(items), p50=gate.percentile(items, .5), p90=gate.percentile(items, .9),
                p95=gate.percentile(items, .95), minimum=min(items) if items else None,
                maximum=max(items) if items else None, mean=statistics.mean(items) if items else None)


def summarize(rows, energy=False):
    pending = remaining(rows)
    groups = []
    for system in SYSTEMS:
        for arm in STRATEGIES:
            selected = [r for r in rows if r['system'] == system and r['arm'] == arm]
            good = [r for r in selected if valid(r) and not energy]
            measured = [values(r) for r in good]
            fields = measured[0].keys() if measured else []
            groups.append(dict(system=system, arm=arm, attempts=len(selected), valid=len(good),
                failures=sum(r['status'] != 'success' for r in selected),
                errors=sum(r['status'] == 'error' for r in selected), aborted=sum(r['status'] == 'aborted' for r in selected),
                invalid=sum(bool(r.get('conditions_invalid') or r.get('environment_invalid')) for r in selected),
                energy_aborted_records=len(selected) if energy else 0,
                metrics={field: metrics([r[field] for r in measured]) for field in fields}))
    ready = not energy and all(g['valid'] == 20 for g in groups)
    latency = not energy and all(g['valid'] == 20 and g['metrics']['total_s']['p95'] <= 60
                               for g in groups if g['arm'] == 'candidate')
    by = {key(r): r for r in rows if valid(r) and not energy}
    pairs = []
    for system in SYSTEMS:
        for index in range(20):
            control, candidate = by.get((system, index, 'control')), by.get((system, index, 'candidate'))
            if control and candidate:
                if control['window_id'] != candidate['window_id']:
                    raise ValueError('Experimental pair crosses restoration windows')
                a, b = values(control), values(candidate)
                pairs.append(dict(system=system, index=index, window_id=control['window_id'],
                    first_arm='control' if control['started_at'] < candidate['started_at'] else 'candidate',
                    control=control['attempt_id'], candidate=candidate['attempt_id'],
                    delta_candidate_minus_control={k: b[k]-a[k] for k in a}))
    return dict(experiment=EXPERIMENT, total_attempts=120, complete=not pending and not energy,
        confirmation_ready=ready, latency_pass_candidate=latency, energy_aborted=energy,
        groups=groups, pairs=pairs, pending=pending, quality_pass=False,
        verdict='NO-GO vigente; requires real quality/latency evidence and review',
        note='Failures/invalid/aborted never enter quantiles; no replacements; stage quantiles are not additive.')


def canonical(report):
    return {k: v for k, v in report.items() if k != 'processing_time_ms'}


def equivalence(detector, response):
    """Actual NLI replay, OUTSIDE response clocks; caller persists raw scores."""
    previous = detector.nli_pair_schedule
    arms = {}
    try:
        for arm, strategy in STRATEGIES.items():
            detector.nli_pair_schedule = strategy
            calls = []
            predict = detector.nli_model.predict
            def capture(sentences, **kwargs):
                record = dict(pairs=len(sentences), options=kwargs)
                calls.append(record)
                try:
                    result = predict(sentences, **kwargs)
                    record.update(status='success', raw_scores=result.tolist() if hasattr(result, 'tolist') else result)
                    return result
                except Exception as exc:
                    record.update(status='error', error_type=type(exc).__name__)
                    raise
            try:
                with patch.object(detector.nli_model, 'predict', capture):
                    report = detector.check(response['llm_response']['text'], response['retrieved_chunks']).model_dump(mode='json')
                arms[arm] = dict(report=report, calls=calls, batch_error=detector.nli_batch_error)
            except Exception as exc:
                arms[arm] = dict(error=f'{type(exc).__name__}: {exc}', calls=calls)
    finally:
        detector.nli_pair_schedule = previous
    healthy = all('report' in a and not a.get('batch_error') and
                  a['report']['method'] not in ('mixed', 'keyword_fallback') and
                  all(c['status'] == 'success' for c in a['calls']) for a in arms.values())
    identical = healthy and canonical(arms['control']['report']) == canonical(arms['candidate']['report'])
    original = response.get('hallucination_report')
    matches_recorded = bool(original) and healthy and canonical(original) == canonical(arms['control']['report'])
    return dict(passed=bool(identical and matches_recorded), healthy=healthy,
                identical=bool(identical), matches_recorded=matches_recorded, arms=arms)


def quality_check(root, name, row, detector):
    source_sha = hashlib.sha256(json.dumps(row['response'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = Path(root) / 'quality' / (name + '.json')
    if target.exists():
        result = gate.read_json(target)
        if result['source_sha256'] != source_sha:
            raise ValueError('Quality replay input changed')
    else:
        result = dict(at=gate.now(), source_sha256=source_sha, **equivalence(detector, row['response']))
        gate.write_new(target, result)
    if not result['passed']:
        raise RuntimeError('NLI equivalence failed; keep evidence, no GO: ' + str(target))
    return result


def quality_summary(root, rows):
    files = list((Path(root) / 'quality').glob('*.json'))
    results = {p.stem: gate.read_json(p) for p in files}
    protocol = gate.read_json(Path(root) / 'source-manifest.json')['protocol']
    ids = protocol.get('quality_source_ids', [])
    base_count = sum(results.get('base-'+i, {}).get('passed', False) for i in ids)
    new_count = sum(results.get('new-'+r['attempt_id'], {}).get('passed', False) for r in rows if valid(r))
    return dict(base_passed=base_count, new_passed=new_count,
                failures=[k for k, r in results.items() if not r['passed']],
                passed=len(set(ids)) == 40 and base_count == 40 and new_count == 120
                and all(r['passed'] for r in results.values()))


def resume_boundary(rows):
    """Never complete the mate across windows; unresolved cases require operator review."""
    remaining(rows)
    if len(rows) % 2:
        raise RuntimeError('Interrupted pair: preserve cohort; operator protocol decision required')


def bootstrap(rows):
    """Exploratory paired p95 differences; not a gate decision or a speed promise."""
    import numpy as np
    result = []
    by = {key(r): r for r in rows if valid(r)}
    for system in SYSTEMS:
        pairs = [(by[(system, i, 'control')], by[(system, i, 'candidate')]) for i in range(20)
                 if (system, i, 'control') in by and (system, i, 'candidate') in by]
        if len(pairs) != 20:
            result.append(dict(system=system, n=len(pairs), status='insufficient'))
            continue
        rng = np.random.default_rng(42)
        draws = rng.integers(0, 20, size=(10000, 20))
        fields = {}
        for field in ('total_s', 'nli_s'):
            a = np.array([values(p[0])[field] for p in pairs])
            b = np.array([values(p[1])[field] for p in pairs])
            deltas = np.quantile(b[draws], .95, axis=1) - np.quantile(a[draws], .95, axis=1)
            fields[field] = dict(delta_p95=float(np.quantile(b, .95)-np.quantile(a, .95)),
                ci95=[float(x) for x in np.quantile(deltas, [.025, .975])])
        a = np.array([values(p[0])['nli_s'] for p in pairs])
        b = np.array([values(p[1])['nli_s'] for p in pairs])
        saving = float(1-b.sum()/a.sum()) if a.sum() else None
        result.append(dict(system=system, n=20, resamples=10000, seed=42, exploratory=True,
            fields=fields, aggregate_nli_saving_fraction=saving, expected_saving_fraction=.65,
            discrepancy_from_expected=None if saving is None else saving-.65))
    return result


def execute_slots(root, measure, *, window_id, margin=gate.window_has_margin):
    """Shared real/synthetic calendar. Blocks of six; never reserve a partial pair."""
    from scripts.unattended_diagnostic import should_stop
    root = Path(root)
    rows = gate.local_records(root)
    resume_boundary(rows)
    started, initial = time.monotonic(), len(rows)
    for offset, slot in enumerate(remaining(rows)):
        count = initial + offset
        # Conservative 600 s per remaining response in the current six-attempt block.
        # Not a total HTTP timeout: watchdog still enforces the hard window deadline.
        if count % 6 == 0 or offset == 0:
            if not margin((6-count % 6)*600 + 120):
                gate.write_new(root / 'pauses' / f'{uuid.uuid4().hex}.json',
                    dict(at=gate.now(), window_id=window_id, completed=count, reason='block_margin'))
                return
        if not margin(120):
            raise TimeoutError('Expired/restoration margin; no further inference')
        system, index, arm = slot
        row = measure(system, index, arm, window_id)
        if key(row) != slot or row['window_id'] != window_id:
            raise ValueError('Measured response does not match registered slot/window')
        # measure_position publishes its own result; synthetic callback uses the same contract.
        current = gate.local_records(root)
        if len(current) != count+1 or key(row) not in {key(r) for r in current}:
            raise ValueError('Attempt not durably published exactly once')
        wall = time.monotonic()-started
        print(f"intento {count+1}/120 | {system} | {arm} | elapsed {row.get('elapsed_s')} s | "
              f"ETA {(119-count)*wall/(offset+1):.0f} s | {row['status']} "
              f"invalid={row.get('conditions_invalid', False)}", flush=True)
        if should_stop(row, unattended=True):
            raise RuntimeError('Technical failure: preserve attempt and restore window')


def replay_quality(root, protocol, preparation, *, margin=gate.window_has_margin):
    from scripts.unattended_diagnostic import verify_package
    source = Path(protocol['quality_source'])
    verify_package(source)
    if gate.digest(source / 'manifest.json') != protocol['quality_source_sha256']:
        raise ValueError('Quality source manifest changed')
    base = gate.local_records(source / 'cohort')
    if sorted(r['attempt_id'] for r in base) != protocol['quality_source_ids']:
        raise ValueError('Quality source response inventory changed')
    inputs = [('base-'+r['attempt_id'], r) for r in base]
    inputs += [('new-'+r['attempt_id'], r) for r in gate.local_records(root) if valid(r)]
    for index, (name, row) in enumerate(inputs):
        if not margin(600):
            return
        quality_check(root, name, row, preparation.pipelines[row['system']].hallucination_detector)
        print(f'calidad {index+1}/{len(inputs)} | {name} | fuera del cronometro', flush=True)


def measure_arm(preparation, system, arm, work):
    """Only scheduling varies; restore even if the durable query raises."""
    detector = preparation.pipelines[system].hallucination_detector
    previous = detector.nli_pair_schedule
    try:
        detector.nli_pair_schedule = STRATEGIES[arm]
        return work()
    finally:
        detector.nli_pair_schedule = previous


def require_window(root):
    """The real CLI refuses standalone/unbounded inference before preparing models."""
    identity = os.environ.get('CLOUDRAG_GATE_WINDOW_ID')
    deadline = os.environ.get('CLOUDRAG_GATE_DEADLINE')
    if not identity or not deadline:
        raise RuntimeError('Real execution requires the human-launched supervised window')
    matches = [(p, gate.read_json(p)) for p in (Path(root).parent / 'windows').glob('*/window.json')
               if gate.read_json(p).get('id') == identity]
    if len(matches) != 1:
        raise RuntimeError('Supervised window identity missing/ambiguous')
    path, window = matches[0]
    armed = path.with_name('armed.json')
    if (window.get('simulated') or not window.get('unattended') or
            not window.get('lexical_diagnostic') or window.get('manage_anydesk') or
            Path(window['cohort']).resolve() != Path(root).resolve() or
            not armed.exists() or path.with_name('restored.json').exists()):
        raise RuntimeError('Window not authorized/armed for this cohort')
    expiry = datetime.fromisoformat(deadline)
    hard = datetime.fromisoformat(window['hard_deadline_utc'])
    started = datetime.fromisoformat(window['at'])
    if (expiry.tzinfo is None or hard.tzinfo is None or started.tzinfo is None or
            expiry <= datetime.now(timezone.utc) or expiry > hard or
            hard-started > timedelta(minutes=120) or
            expiry != datetime.fromisoformat(gate.read_json(armed)['deadline_utc'])):
        raise RuntimeError('Invalid/expired supervised deadline; zero inference')
    if any(not (path.parent / 'proof' / kind / 'selftest-passed.json').exists()
           for kind in ('deadline', 'controller')):
        raise RuntimeError('Supervisor proof missing; zero inference')


def run(root):
    from scripts.run_lexical_diagnostic import measure_position
    root = Path(root)
    require_window(root)
    protocol = gate.read_json(root / 'source-manifest.json')['protocol']
    with gate.coordinator_lock(root):
        gate.recover(root)
        rows = gate.local_records(root)
        resume_boundary(rows)
        check_protocol(protocol)
        if not gate.window_has_margin(900):
            raise TimeoutError('No preparation margin; no inference')
        import torch
        if torch.version.cuda is not None:
            raise ValueError('Auxiliary runtime must be CPU-only')
        from src.ui.components.index_loader import load_hybrid_index, load_pipeline
        from src.ui.components.interview_preparation import Preparation
        preparation = Preparation(root / 'preparation',
            factory=lambda k: load_pipeline(k, _hybrid_index=load_hybrid_index()))
        scope = uuid.uuid4().hex
        receipt = preparation.prepare(scope)
        gate.write_new(root / 'warmups' / f'{scope}.json', receipt)
        def measure(system, index, arm, window_id):
            return measure_arm(preparation, system, arm,
                lambda: measure_position(root, protocol, preparation, scope, receipt, system, index,
                    extra_metadata=dict(arm=arm, nli_schedule=STRATEGIES[arm], window_id=window_id)))
        execute_slots(root, measure, window_id=os.environ['CLOUDRAG_GATE_WINDOW_ID'])
        rows = gate.local_records(root)
        if not remaining(rows):
            replay_quality(root, protocol, preparation)
        return summarize(rows)


def synthetic_response():
    """Real verifier, deterministic mock scores; no models or service calls."""
    import numpy as np
    from src.generation.hallucination_detector import HallucinationDetector
    class Model:
        def predict(self, sentences, **kwargs):
            return np.array([[.05, .9, .05] for _ in sentences])
    detector = HallucinationDetector()
    detector._nli_model = Model()
    detector._extract_claims = lambda _: ['The service supports private endpoints.', 'The service encrypts data.']
    response = dict(answer='Synthetic technical answer.', retrieved_chunks=[dict(text='service', chunk_id='A')],
                    llm_response=dict(text='Synthetic technical answer.', tokens_output=2))
    response['hallucination_report'] = detector.check(response['answer'], response['retrieved_chunks']).model_dump(mode='json')
    return detector, response


def dry_run(root):
    from types import SimpleNamespace
    from scripts.lexical_diagnostic import PipelineTrace
    from scripts import unattended_diagnostic as unattended
    root = unattended.external_root(root)
    root.mkdir(parents=True, exist_ok=False)
    unattended.dry_run(root / 'safety')
    cohort = root / 'cohort'
    gate.write_new(cohort / 'source-manifest.json', dict(protocol=dict(nli_experiment=EXPERIMENT,
        quality_source_ids=[str(i) for i in range(40)]), simulated=True))
    detector, response = synthetic_response()
    subject = SimpleNamespace(hallucination_detector=detector, llm=SimpleNamespace(generate=lambda prompt: 'answer'))
    preparation = SimpleNamespace(pipelines={s: subject for s in SYSTEMS})
    calls = []
    def measure(system, index, arm, window_id):
        calls.append((system, index, arm))
        row = unattended.synthetic_row(system, index)
        row.update(arm=arm, window_id=window_id, attempt_id=f'{system}-{index}-{arm}',
                   started_at=f'{len(calls):04d}', simulated=True)
        row['response'].update(response)
        with PipelineTrace(subject) as trace:
            report = measure_arm(preparation, system, arm,
                lambda: detector.check(response['llm_response']['text'], response['retrieved_chunks']))
        row['response']['hallucination_report'] = report.model_dump(mode='json')
        traced = trace.export()
        for field in ('nli', 'nli_pairs_attempted', 'nli_predict_calls', 'extractions'):
            row['diagnostic_trace'][field] = traced[field]
        gate.write_new(cohort / 'attempts' / row['attempt_id'] / 'result.json', row)
        return row
    checks = {}
    for number in (1, 2):
        window = root / 'windows' / str(number)
        gate.write_new(window / 'window.json', dict(simulated=True, supervisor_proved=True))
        execute_slots(cohort, measure, window_id=str(number),
            margin=lambda _: number == 2 or len(calls) < 12)
        gate.write_new(window / 'restored.json', dict(simulated=True))
        if number == 1:
            checks['checkpoint_12'] = len(calls) == 12
            try:
                unattended.check_resume(root, False)
                checks['authorization_required'] = False
            except PermissionError:
                checks['authorization_required'] = True
            unattended.check_resume(root, True)
    checks['calendar_120_once'] = calls == schedule() and len(set(calls)) == 120
    before = len(calls)
    execute_slots(root / 'expired', measure, window_id='expired', margin=lambda _: False)
    checks['expired_no_inference'] = len(calls) == before
    rows = gate.local_records(cohort)
    for i in range(40):
        quality_check(cohort, 'base-'+str(i), dict(response=response), detector)
    for row in rows:
        quality_check(cohort, 'new-'+row['attempt_id'], row, detector)
    invalid = [dict(r) for r in rows]
    invalid[0]['conditions_invalid'] = True
    report = summarize(invalid)
    checks['invalid_excluded_not_replaced'] = not report['confirmation_ready'] and not report['pending']
    checks['quality_160'] = quality_summary(cohort, rows)['passed']
    checks['safety'] = all(gate.read_json(root / 'safety/dry-run.json')['checks'].values())
    gate.write_new(root / 'dry-run.json', dict(at=gate.now(), simulated=True, checks=checks))
    unattended.package(root)
    checks['integrity'] = bool(unattended.verify_package(root))
    if not all(checks.values()):
        raise RuntimeError('Synthetic NLI scenario failed')
    print(json.dumps(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'dry-run', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'dry-run':
        dry_run(args.output)
    elif args.command == 'report':
        rows = gate.local_records(args.output)
        print(json.dumps(dict(**summarize(rows), bootstrap=bootstrap(rows))))
    else:
        try:
            run(args.output)
        except Exception as exc:
            gate.write_new(args.output / 'failures' / f'{uuid.uuid4().hex}.json',
                dict(at=datetime.now(timezone.utc).isoformat(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    main()
