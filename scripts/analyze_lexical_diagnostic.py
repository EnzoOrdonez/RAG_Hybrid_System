"""Read-only reconstruction of warm historical workload; never invent NLI calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate
from scripts.analyze_gate_memory import stats


def verified_path(root, path, inventory):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes cohort')
    relative = path.relative_to(root).as_posix()
    if relative not in inventory or gate.digest(path) != inventory[relative]:
        raise ValueError(f'Missing or changed evidence hash: {relative}')
    return path


def workload(row, calls):
    response = row['response']
    llm, report = response.get('llm_response') or {}, response.get('hallucination_report') or {}
    chats = [r for r in calls if r['method'] == 'chat']
    if len(chats) != 1 or chats[0]['status'] != 'success':
        raise ValueError('Expected exactly one successful chat')
    messages = chats[0]['request']['messages']
    prompt = [m['content'] for m in messages if m['role'] == 'user']
    if len(prompt) != 1 or not isinstance(prompt[0], str):
        raise ValueError('Expected one text user prompt')
    stages = {k.replace('_ms', '_s'): v / 1000 for k, v in response['latency'].items()}
    measured = sum(v for k, v in stages.items() if k != 'total_s')
    return dict(attempt_id=row['attempt_id'], query_id=row['query']['query_id'], index=row['index'],
        system=row['system'], started_at=row['started_at'], elapsed_s=row['elapsed_s'], stages=stages,
        other_s=row['elapsed_s'] - measured, chunks=len(response['retrieved_chunks']),
        chunk_ids=[c.get('chunk_id') for c in response['retrieved_chunks']],
        prompt_chars=len(prompt[0]), system_prompt_chars=sum(len(m['content']) for m in messages if m['role'] == 'system'),
        tokens_input=llm.get('tokens_input'), tokens_output=llm.get('tokens_output'),
        answer_chars=len(llm.get('text', '')), claims=report.get('total_claims'),
        artifacts=report.get('not_a_claim_claims'), faithfulness=report.get('faithfulness_score'),
        nli_predict_calls=None, nli_pairs_attempted=None,
        nli_count_certainty='UNMEASURED: historical records have claim outcomes, not actual predict calls')


def summarize(rows):
    groups = []
    for system in gate.SYSTEMS:
        selected = [r for r in rows if r['system'] == system]
        if not selected:
            continue
        numeric = ('elapsed_s', 'other_s', 'chunks', 'prompt_chars', 'system_prompt_chars',
                   'tokens_input', 'tokens_output', 'answer_chars', 'claims', 'artifacts')
        groups.append(dict(system=system, n=len(selected),
            metrics={k: stats(r[k] for r in selected) for k in numeric},
            stages={k: stats(r['stages'][k] for r in selected) for k in selected[0]['stages']}))
    pairs = []
    by_slot = {(r['system'], r['index']): r for r in rows}
    if len(by_slot) != len(rows):
        raise ValueError('Duplicate warm logical slot')
    for index in range(20):
        hybrid, lexical = by_slot.get(('hybrid', index)), by_slot.get(('lexical', index))
        if hybrid is None or lexical is None:
            continue
        if hybrid['query_id'] != lexical['query_id']:
            raise ValueError('Paired positions contain different queries')
        pairs.append(dict(index=index, query_id=hybrid['query_id'],
            lexical_minus_hybrid_s=lexical['elapsed_s'] - hybrid['elapsed_s'],
            lexical_attempt=lexical['attempt_id'], hybrid_attempt=hybrid['attempt_id']))
    return dict(groups=groups, pairs=pairs)


def analyze(root, analysis_path, expected_analysis_sha256):
    if gate.digest(analysis_path) != expected_analysis_sha256:
        raise ValueError('Analysis identity mismatch')
    analysis = gate.read_json(analysis_path)
    inventory = analysis['source_hashes']
    root = Path(root).resolve()
    verified_path(root, root / 'source-manifest.json', inventory)
    rows, excluded = [], []
    for path in sorted(root.glob('attempts/*/result.json')):
        row = gate.read_json(verified_path(root, path, inventory))
        if row.get('warmup') or row['phase'] != 'warm':
            continue
        if row['status'] != 'success' or row.get('conditions_invalid') or row.get('environment_invalid'):
            excluded.append({k: row.get(k) for k in ('attempt_id', 'system', 'index', 'status', 'error')})
            continue
        calls = [gate.read_json(verified_path(root, p, inventory))
                 for p in sorted(Path(row['http_trace_path']).glob('*.json'))]
        rows.append(workload(row, calls))
    return dict(at=gate.now(), build=gate.git('rev-parse', 'HEAD'), source_root=str(root),
        script_sha256=gate.digest(__file__), analysis_sha256=expected_analysis_sha256,
        rows=rows, excluded=excluded, **summarize(rows),
        limitations=['Historical, non-counterbalanced executions; no exclusive causal attribution.',
                     'NLI calls/pairs were not measured; missing counts remain null.',
                     'Stage percentiles are not additive; other_s includes external overhead.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--analysis-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    checkout = (gate.PROJECT / gate.git('rev-parse', '--git-common-dir')).resolve().parent
    if output.is_relative_to(checkout) or output.is_relative_to(args.cohort.resolve()):
        raise ValueError('Output must be outside checkout and historical cohort')
    gate.write_new(output, analyze(args.cohort, args.analysis, args.analysis_sha256))
