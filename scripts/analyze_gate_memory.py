"""Read-only, hash-checked memory/phase analysis of one controlled gate cohort."""
import argparse
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate

GIB = 1024 ** 3


def stats(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite observation')
    return dict(n=len(values), min=min(values) if values else None,
                median=gate.percentile(values, .5), p95=gate.percentile(values, .95),
                max=max(values) if values else None, mean=sum(values) / len(values) if values else None)


def stamp(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError('Timestamp lacks timezone')
    return value.timestamp()


def window_metrics(samples, faults, start, elapsed):
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Invalid interval')
    origin = stamp(start)
    selected = [r for r in samples if origin <= stamp(r['at']) < origin + elapsed]
    hard = [r for r in faults if origin <= r['timestamp_s'] < origin + elapsed]
    return dict(duration_s=elapsed, samples=len(selected),
        available_gib=stats(r['ram_available_bytes'] / GIB for r in selected),
        committed_gib=stats(r['committed_bytes'] / GIB for r in selected),
        commit_fraction=stats(r['committed_bytes'] / r['commit_limit_bytes'] for r in selected),
        gpu_util_percent=stats(float(r['gpu']['utilization.gpu']) for r in selected),
        gpu_temperature_c=stats(float(r['gpu']['temperature.gpu']) for r in selected
                                if r['gpu']['temperature.gpu'] not in ('[N/A]', 'N/A')),
        gpu_free_mib=stats(float(r['gpu']['memory.free']) for r in selected),
        gpu_limit_flags=dict(Counter(f'{k}={v}' for r in selected for k, v in r['gpu'].items()
            if k.startswith('clocks_event_reasons.') and k != 'clocks_event_reasons.active')),
        hard_fault_count=len(hard), hard_faults_per_s=len(hard) / elapsed,
        hard_fault_bytes=sum(r['bytes_read'] for r in hard),
        hard_fault_processes=dict(Counter(r.get('process') or 'unknown' for r in hard)),
        hard_fault_files=dict(Counter(r.get('file') or 'unknown' for r in hard).most_common(12)),
        offload_reports=sorted({r['ollama_ps_cli'] for r in selected}),
        page_input_per_s=stats(r.get('paging', {}).get('pages_input_per_s') for r in selected),
        all_sampled_ac=all(r.get('ac') is True for r in selected) if selected else None)


def analyze(root):
    root = Path(root).resolve()
    sources = {}

    def source(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Evidence path escapes cohort')
        checksum = gate.digest(path)
        if expected and checksum != expected:
            raise ValueError(f'Evidence hash mismatch: {path.name}')
        sources[path.relative_to(root).as_posix()] = checksum
        return path

    manifest = gate.read_json(source(root / 'source-manifest.json'))
    rows = gate.all_records(root)
    report = gate.summarize(rows, gate.selected_systems(manifest['protocol']))
    attempts = []
    for row in rows:
        directory = root / 'attempts' / row['attempt_id']
        source(directory / 'result.json')
        source(directory / 'events.jsonl', row['journal_sha256'])
        item = {k: row.get(k) for k in ('attempt_id', 'system', 'phase', 'index', 'warmup', 'status', 'elapsed_s')}
        if row['status'] != 'success' or row.get('conditions_invalid') or row.get('environment_invalid'):
            attempts.append(dict(item, excluded_from_performance=True))
            continue
        samples = gate.read_events(source(row['telemetry_path']))
        decoded_path = source(row['hard_fault_evidence'], row['hard_fault_evidence_sha256'])
        decoded = gate.read_json(decoded_path)
        if decoded['events_lost'] or decoded['buffers_lost']:
            raise ValueError('Incomplete ETW capture')
        source(decoded_path.with_name('trace.etl'), decoded['trace_sha256'])
        calls = [gate.read_json(source(p)) for p in Path(row['http_trace_path']).glob('*.json')]
        chats = [c for c in calls if c['method'] == 'chat']
        if len(chats) != 1 or chats[0]['status'] != 'success':
            raise ValueError('Expected one successful chat call per response')
        chat = chats[0]
        if abs(stamp(chat['finished_at']) - stamp(chat['started_at']) - chat['elapsed_s']) > 1:
            raise ValueError('Wall/monotonic chat clocks disagree')
        # request.started_at precedes observer setup, so it is not the response origin.
        source(directory / 'request.json')
        trace_start = gate.read_json(source(decoded_path.with_name('trace-active.json')))['at']
        # The observer's first sample is taken immediately before the response clock.
        # Use persisted HTTP boundaries for chat; whole-observer metrics are explicitly labeled.
        whole_start = samples[0]['at']
        whole_elapsed = stamp(samples[-1]['at']) - stamp(whole_start)
        if whole_elapsed <= 0 or stamp(trace_start) > stamp(whole_start):
            raise ValueError('Invalid observer bounds')
        stages = {k.removesuffix('_ms') + '_s': v / 1000 for k, v in row['response']['latency'].items()}
        server = chat.get('server', {})
        item.update(query_id=row['query']['query_id'], stages=stages,
            outside_timed_stages_s=row['elapsed_s'] - stages['total_s'],
            chat_s=chat['elapsed_s'],
            observer_window=window_metrics(samples, decoded['hard_faults'], whole_start, whole_elapsed),
            chat_window=window_metrics(samples, decoded['hard_faults'], chat['started_at'], chat['elapsed_s']),
            server_seconds={k: v / 1e9 for k, v in server.items() if k.endswith('_duration') and v is not None},
            output_tokens=server.get('eval_count'),
            generated_tokens_per_s=(server['eval_count'] * 1e9 / server['eval_duration']
                if server.get('eval_duration') and server.get('eval_count') is not None else None))
        attempts.append(item)
    groups = []
    for system in gate.selected_systems(manifest['protocol']):
        for phase in gate.PHASES:
            selected = [r for r in attempts if r['system'] == system and r['phase'] == phase
                        and not r['warmup'] and not r.get('excluded_from_performance')]
            groups.append(dict(system=system, phase=phase, n=len(selected),
                elapsed_s=stats(r['elapsed_s'] for r in selected),
                stages={k: stats(r['stages'][k] for r in selected) for k in (selected[0]['stages'] if selected else [])},
                chat_min_available_gib=stats(r['chat_window']['available_gib']['min'] for r in selected),
                chat_hard_faults_per_s=stats(r['chat_window']['hard_faults_per_s'] for r in selected),
                tokens_per_s=stats(r['generated_tokens_per_s'] for r in selected),
                outside_timed_stages_s=stats(r['outside_timed_stages_s'] for r in selected)))
    return dict(at=gate.now(), analysis_build=gate.git('rev-parse', 'HEAD'), analysis_sha256=gate.digest(__file__),
        measured_build=manifest['protocol']['build_id'], source_root=str(root), source_hashes=sources,
        latency_report=report, groups=groups, attempts=attempts,
        limitations=['Observational: does not establish exclusive cause or a RAM-upgrade speedup.',
            'No historical cohorts combined; warmup excluded from group statistics.',
            'Observer window includes boundary sampling overhead; chat uses exact HTTP timestamps.',
            'Hard-fault rate counts ETW completion events per second, not PDH total page faults.',
            'Per-stage percentiles are descriptive and must not be added.',
            'No CPU temperature sensor; GPU flags cannot rule out CPU thermal throttling.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.cohort.resolve()) or args.output.resolve().is_relative_to(gate.PROJECT.parent.parent):
        parser.error('Use a new output outside cohort and checkout')
    gate.write_new(args.output, analyze(args.cohort))
