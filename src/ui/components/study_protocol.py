"""Frozen study configuration; no models, sessions or historical data mutations."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVALID_TASKS = frozenset(('q005', 'q007', 'q013', 'q017', 'q028', 'q066', 'q073', 'q074'))
CELLS = {1: (('A', 'T1'), ('B', 'T2')), 2: (('A', 'T2'), ('B', 'T1')),
         3: (('B', 'T1'), ('A', 'T2')), 4: (('B', 'T2'), ('A', 'T1'))}
QUOTAS = {1: (3, 2), 2: (2, 3), 3: (3, 2), 4: (2, 3)}
PROFILES = ('without_experience', 'with_experience')
LIKERT_IDS = ('F1', 'F2', 'F3', 'F4', 'U1', 'U2', 'U3', 'R1', 'R2', 'I1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(values, size):
    if len(values) != size or any(type(v) is not int or not 1 <= v <= 5 for v in values):
        raise ValueError('Incomplete or invalid instrument')
    return values


def sus_score(values):
    scores(values, 10)
    return 2.5 * sum(v - 1 if i % 2 == 0 else 5 - v for i, v in enumerate(values))


def load_protocol(config_path, assignment_path):
    config_path, assignment_path = Path(config_path), Path(assignment_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    reference = json.loads((ROOT / 'config/study.example.json').read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or set(config.get('labels', {})) != {'A', 'B'} or set(config['labels'].values()) != {'hybrid', 'no_rag'}:
        raise ValueError('Fill the fixed A/B mapping')
    items = config.get('sus', {}).get('items', [])
    if len(items) != 10 or any(not isinstance(v, str) or not v.strip() for v in items):
        raise ValueError('Fill all ten literal SUS items before admission')
    # These texts are transcribed from the specified protocol, never freely reworded.
    for name in ('likert', 'comparative', 'free_instruction', 'blinding_question', 'blinding_choices', 'blinding_reason'):
        if config.get(name) != reference[name]:
            raise ValueError('Protocol instrument text changed: ' + name)
    queries_path = ROOT / 'data/evaluation/test_queries.json'
    queries = json.loads(queries_path.read_text(encoding='utf-8'))
    catalog = {q['query_id']: q for q in queries}
    tasks = config.get('tasks', {})
    if set(tasks) != {'T1', 'T2'} or any(len(v) != 3 for v in tasks.values()):
        raise ValueError('Require two sets of three tasks')
    ids = tasks['T1'] + tasks['T2']
    if len(set(ids)) != 6 or any(q not in catalog or q in INVALID_TASKS for q in ids):
        raise ValueError('Unknown, duplicate or invalid-premise task')
    for slot, kind in enumerate(('factual', 'procedural', 'comparative')):
        a, b = (catalog[tasks[t][slot]] for t in ('T1', 'T2'))
        providers = [set(q['cloud_providers']) for q in (a, b)]
        if any(q['query_type'] != kind for q in (a, b)) or a['difficulty'] != b['difficulty']:
            raise ValueError('Task type/difficulty mismatch')
        if kind != 'comparative' and (providers[0] != providers[1] or len(providers[0]) != 1):
            raise ValueError('Single-provider slots must share provider')
        if kind == 'comparative' and (len(providers[0]) < 2 or len(providers[0]) != len(providers[1])):
            raise ValueError('Comparative slots must match provider count')
    practice = config.get('familiarization')
    forbidden = {catalog[q]['question'].strip().casefold() for q in set(ids) | (INVALID_TASKS & catalog.keys())}
    if not isinstance(practice, str) or not practice.strip() or practice.strip().casefold() in forbidden:
        raise ValueError('Practice must be fixed, nonempty and outside task/invalid sets')
    with assignment_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['participant_id', 'role', 'cell', 'profile']:
            raise ValueError('Assignment columns must not include personal identifiers')
        assignments = list(reader)
    seen = set()
    primary, reserves = [], []
    for row in assignments:
        pid = row['participant_id']
        if pid in seen or pid not in {f'P{i:02d}' for i in range(1, 25)}:
            raise ValueError('Invalid or duplicate participant')
        seen.add(pid)
        try:
            row['cell'] = int(row['cell'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Fill assignment cell') from exc
        if row['cell'] not in CELLS or row['profile'] not in PROFILES:
            raise ValueError('Invalid cell/profile')
        expected_role = 'primary' if int(pid[1:]) <= 20 else 'reserve'
        if row['role'] != expected_role:
            raise ValueError('P01–P20 primary; P21–P24 reserve')
        (primary if expected_role == 'primary' else reserves).append(row)
    if {r['participant_id'] for r in primary} != {f'P{i:02d}' for i in range(1, 21)} or len(reserves) > 4:
        raise ValueError('Require exactly twenty primary assignments')
    for cell, quota in QUOTAS.items():
        if tuple(sum(r['cell'] == cell and r['profile'] == p for r in primary) for p in PROFILES) != quota:
            raise ValueError('Stratified cell quotas differ from protocol')
    # Also validate actual conditions, not just the visible A/B labels.
    for condition in ('hybrid', 'no_rag'):
        for block in (0, 1):
            if sum(config['labels'][CELLS[r['cell']][block][0]] == condition for r in primary) != 10:
                raise ValueError('Condition order imbalance')
        for task_set in ('T1', 'T2'):
            if sum(config['labels'][label] == condition and tasks_name == task_set
                   for r in primary for label, tasks_name in CELLS[r['cell']]) != 10:
                raise ValueError('Task/condition imbalance')
    hashes = dict(config=digest(config_path), assignments=digest(assignment_path), queries=digest(queries_path))
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(config=config, assignments={r['participant_id']: r for r in assignments},
                queries=catalog, hashes=hashes, fingerprint=fingerprint)
