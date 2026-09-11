"""One prepared, counterbalanced diagnostic; no participant recipe modifications."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate
from scripts import observe_interview_gate as observe
from scripts import contrast_interview_observer as contrast
from scripts.lexical_diagnostic import PipelineTrace, measure_traced_attempt, paired_schedule

BASELINE = 'b6f3fea006ce2ae2555bd63ffc4a1e78c37f703a'


def initialize(root):
    root = Path(root).resolve()
    checkout = (gate.PROJECT / gate.git('rev-parse', '--git-common-dir')).resolve().parent
    if root.is_relative_to(checkout):
        raise ValueError('Evidence must be external')
    gate.validate_app_identity(BASELINE)
    protocol = gate.fresh_protocol(systems=list(gate.SYSTEMS), controlled=True)
    protocol.update(diagnostic_only=True, diagnostic_schedule=paired_schedule(),
                    diagnostic_sources={name: gate.digest(gate.PROJECT / name) for name in (
                        'scripts/run_lexical_diagnostic.py', 'scripts/lexical_diagnostic.py',
                        'scripts/run_managed_gate.py', 'scripts/manage_gate_window.ps1')})
    gate.write_new(root / 'source-manifest.json', dict(created_at=gate.now(), mode='paired_diagnostic',
                   protocol=protocol, total_attempts=40, abort_consumes_slot=True))


def check_protocol(protocol):
    if not protocol.get('diagnostic_only') or protocol.get('diagnostic_schedule') != [list(p) for p in paired_schedule()]:
        raise ValueError('Not the frozen paired diagnostic')
    for name, expected in protocol['diagnostic_sources'].items():
        if gate.digest(gate.PROJECT / name) != expected:
            raise ValueError('Diagnostic source changed')
    gate.preflight(protocol)


def remaining(rows):
    slots = gate.completed_slots(rows)
    if any((s, i) not in paired_schedule() or p != 'warm' for s, p, i in slots):
        raise ValueError('Record outside diagnostic schedule')
    return [(s, i) for s, i in paired_schedule() if (s, 'warm', i) not in slots]


def report(rows):
    cells = []
    for system in ('hybrid', 'lexical'):
        selected = [r for r in rows if r['system'] == system and not r.get('warmup')]
        valid = [r for r in selected if r['status'] == 'success'
                 and not r.get('conditions_invalid') and not r.get('environment_invalid')]
        cells.append(dict(system=system, attempts=len(selected), valid=len(valid),
            failures=sum(r['status'] != 'success' for r in selected),
            invalid=sum(bool(r.get('conditions_invalid') or r.get('environment_invalid')) for r in selected),
            p50_s=gate.percentile([r['elapsed_s'] for r in valid], .5),
            p95_s=gate.percentile([r['elapsed_s'] for r in valid], .95)))
    return dict(diagnostic_only=True, complete=not remaining(rows), cells=cells,
                pending=remaining(rows), new_interview_verdict=None)


def execute(root, protocol, preparation, observer_factory=observe.Observer):
    root = Path(root)
    check_protocol(protocol)
    scope = uuid.uuid4().hex
    if not gate.window_has_margin(900):
        raise TimeoutError('No preparation margin')
    receipt = preparation.prepare(scope)
    gate.write_new(root / 'warmups' / f'{scope}.json', receipt)
    for system, index in remaining(gate.local_records(root)):
        if not gate.window_has_margin(600):
            raise TimeoutError('No query/restoration margin')
        check_protocol(protocol)
        if not preparation.ready(scope):
            raise RuntimeError('Preparation residency lost')
        pipeline = preparation.pipeline(system, scope)
        policy = gate.recipe(protocol)
        llm = pipeline.llm
        if (llm.cache_enabled or llm.seed != 42 or llm.max_retries != 1 or llm.timeout != policy['write']
                or llm.read_timeout != policy['read'] or llm.default_keep_alive != policy['keep_alive']):
            raise ValueError('Prepared pipeline recipe changed')
        import httpx
        import ollama
        from scripts.diagnose_interview_timeout import TracedClient
        trace_id = uuid.uuid4().hex
        http_root = root / 'http' / trace_id
        if isinstance(llm._ollama_client, TracedClient):
            llm._ollama_client.output = http_root
        else:
            llm._ollama_client = TracedClient(llm._ollama_client or ollama.Client(host=os.environ['OLLAMA_HOST'],
                timeout=httpx.Timeout(60, read=180, connect=5)), http_root)
        observer = observer_factory(root / 'telemetry' / f'{trace_id}.jsonl',
                                    allowed_pids={os.getpid(), os.getppid()})
        row = measure_traced_attempt(root, dict(system=system, phase='warm', index=index,
            query=protocol['queries'][index], warmup=False, consumes_slot=True,
            build_id=protocol['build_id'], configuration=pipeline.config.model_dump(mode='json'),
            preparation_id=receipt['id'], preparation_check=preparation.last_check,
            http_trace_path=str(http_root), protocol_sha256=gate.digest(root / 'source-manifest.json')),
            pipeline, observer=observer, validate_after=lambda: check_protocol(protocol))
        print({k: row.get(k) for k in ('system', 'index', 'status', 'elapsed_s', 'conditions_invalid')}, flush=True)
        if row['status'] != 'success' or row.get('conditions_invalid') or row.get('environment_invalid'):
            raise RuntimeError('Diagnostic failed/invalid; preserve position and restore window')
    return report(gate.local_records(root))


def run(root):
    root = Path(root)
    protocol = gate.read_json(root / 'source-manifest.json')['protocol']
    with gate.coordinator_lock(root):
        gate.recover(root)
        if not remaining(gate.local_records(root)):
            return report(gate.local_records(root))
        import torch
        if torch.version.cuda is not None:
            raise ValueError('Auxiliary runtime must be CPU-only')
        from src.ui.components.index_loader import load_hybrid_index, load_pipeline
        from src.ui.components.interview_preparation import Preparation
        preparation = Preparation(root / 'preparation',
            factory=lambda key: load_pipeline(key, _hybrid_index=load_hybrid_index()))
        result = execute(root, protocol, preparation)
        gate.write_new(root / 'complete.json', dict(at=gate.now(), **result))
        return result


def synthetic_work(buffer, iterations, observed):
    """Fixed synthetic stage/claim/pair shape; no models, same hashing in both arms."""
    from src.generation.hallucination_detector import HallucinationDetector
    from src.pipeline.rag_pipeline import LatencyTracker
    class Model:
        def predict(self, sentences, **kwargs):
            return contrast.hash_work(buffer, iterations)
    detector = HallucinationDetector()
    detector._nli_model = Model()
    subject = SimpleNamespace(llm=SimpleNamespace(generate=lambda prompt, **kwargs: 'answer'),
                              hallucination_detector=detector)
    tracker = LatencyTracker()
    with PipelineTrace(subject) if observed else nullcontext():
        with tracker.measure('generation'):
            subject.llm.generate('technical context ' * 700, system_prompt='synthetic')
        with tracker.measure('hallucination_check'):
            detector._extract_claims('The service supports private endpoints. ' * 20)
            for _ in range(20):
                detector._nli_model.predict([('context', 'claim')] * 5, batch_size=32, apply_softmax=True)
    return dict(status='success', checksum=contrast.hash_work(buffer, 1)['checksum'])


def run_contrast(root):
    root = Path(root)
    sampler = observe.Sampler()
    buffer = bytes(4 * 1024 * 1024)
    synthetic_work(buffer, 1, False)  # imports/initialization outside calibration
    started = time.perf_counter()
    synthetic_work(buffer, 1, False)
    iterations = max(1, math.ceil(10 / (time.perf_counter() - started)))
    gate.write_new(root / 'protocol.json', dict(at=gate.now(), build=gate.git('rev-parse', 'HEAD'),
        iterations=iterations, buffer_bytes=len(buffer), pairs=10, threshold_percent=5,
        observed_arm='ETW observer plus PipelineTrace', models_executed=False,
        source_sha256=gate.digest(__file__)))
    counter = iter(arm == 'observed' for i in range(10) for arm in contrast.order(i))
    rows = contrast.execute_pairs(root, lambda: synthetic_work(buffer, iterations, next(counter)), sampler)
    result = contrast.summarize(rows)
    gate.write_new(root / 'result.json', dict(at=gate.now(), **result))
    if not result['passed']:
        raise RuntimeError('Combined observer interference exceeds bound')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'run', 'report', 'contrast'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.output)
    elif args.command == 'contrast':
        run_contrast(args.output)
    elif args.command == 'report':
        print(report(gate.local_records(args.output)))
    else:
        try:
            run(args.output)
        except Exception as exc:
            gate.write_new(args.output / 'failures' / f'{uuid.uuid4().hex}.json',
                           dict(at=datetime.now(timezone.utc).isoformat(), error=f'{type(exc).__name__}: {exc}'))
            raise
