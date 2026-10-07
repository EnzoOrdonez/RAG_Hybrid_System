"""Preregistered synthetic census: complete histories, never response-based selection."""
from collections import Counter, defaultdict
import json

TASKS = ('q001', 'q068', 'q180', 'q016', 'q070', 'q172')
CONDITIONS = ('hybrid', 'no_rag')
FREE_HISTORY = ('Describe a fictional cloud backup plan for a small synthetic research dataset.',
                'Which factors should a fictional team consider when choosing a managed database?')
V2_VERSION = 'faithfulness_v2_28_patterns_300_chars'
CELLS = {1: (('A', 'T1'), ('B', 'T2')), 2: (('A', 'T2'), ('B', 'T1')),
         3: (('B', 'T1'), ('A', 'T2')), 4: (('B', 'T2'), ('A', 'T1'))}


def calendar(config):
    if (tuple(config['tasks']['T1']+config['tasks']['T2']) != TASKS
            or set(config['labels']) != {'A', 'B'} or set(config['labels'].values()) != set(CONDITIONS)):
        raise ValueError('Reviewed task set or label mapping changed')
    boots = {i: [] for i in range(1, 13)}
    combinations = [(q, c) for q in TASKS for c in CONDITIONS]
    for boot, (q, c) in enumerate(combinations, 1):
        boots[boot].append(dict(role='target', history='first_after_boot', query_id=q, condition=c))
    for repetition in range(1, 4):
        for pos, q in enumerate(TASKS):
            for c in (CONDITIONS if (repetition+pos) % 2 else CONDITIONS[::-1]):
                boots[1].append(dict(role='target', history='gate_calendar', query_id=q,
                                     condition=c, repetition=repetition))
    for repetition, boot in [(1, 2), (2, 3)]:
        for cell, blocks in CELLS.items():
            for label, taskset in blocks:
                for q in config['tasks'][taskset]:
                    boots[boot].append(dict(role='target', history='latin_cell', query_id=q,
                        condition=config['labels'][label], cell=cell, repetition=repetition))
    for history_number, question in enumerate(FREE_HISTORY, 1):
        for q, c in combinations:
            boots[4].append(dict(role='antecedent', history='arbitrary_free', question=question,
                                condition=c, history_number=history_number))
            boots[4].append(dict(role='target', history='after_arbitrary_free', query_id=q,
                                condition=c, history_number=history_number))
    counts = Counter((r['query_id'], r['condition']) for rows in boots.values() for r in rows if r['role'] == 'target')
    if len(counts) != 12 or set(counts.values()) != {10}:
        raise ValueError('Census construction incomplete')
    return boots


def analyze(rows, config, *, synthetic=False):
    """Reject missing predecessors, repeated slots/boots and mixed software first."""
    expected = {(boot, index): slot for boot, slots in calendar(config).items()
                for index, slot in enumerate(slots, 1)}
    seen, boot_ids, software, groups = set(), {}, set(), defaultdict(list)
    if len(rows) != len(expected):
        raise ValueError('Incomplete 144-call census, including free predecessors')
    for row in rows:
        key = row['boot_index'], row['index']
        if key in seen or expected.get(key) != row['slot'] or row['status'] != 'success' or row['valid'] is not True:
            raise ValueError('Duplicate, altered, failed or invalid stimulus slot')
        seen.add(key)
        boot = boot_ids.setdefault(row['boot_index'], row['boot_id'])
        if boot != row['boot_id'] or not boot:
            raise ValueError('A census boot contains mixed boot identities')
        software.add(row['software_sha256'])
        if row['response_class_version'] != V2_VERSION:
            raise ValueError('Stimulus class is not v2')
        if row['slot']['role'] == 'target':
            groups[row['slot']['query_id'], row['slot']['condition']].append(row)
    if seen != set(expected) or len(set(boot_ids.values())) != 12 or len(software) != 1:
        raise ValueError('Mixed software, missing slots or reused cold boot')
    result = {}
    for q in TASKS:
        for c in CONDITIONS:
            group = groups[q, c]
            variants = {key: len({json.dumps(row[key], sort_keys=True, ensure_ascii=False) for row in group})
                        for key in ('raw_text', 'answer', 'response_class', 'citations')}
            result[q+'|'+c] = dict(n=len(group), boots=sorted({row['boot_id'] for row in group}),
                variants=variants, accepted=len(group) == 10 and len({row['boot_id'] for row in group}) >= 2
                    and set(variants.values()) == {1})
    accepted = sum(group['accepted'] for group in result.values())
    return dict(status='SYNTHETIC_NOT_ACCEPTANCE' if synthetic else 'ACCEPTED' if accepted == 12 else 'BLOQUEADO-HUMANO',
        accepted_combinations=accepted, groups=result, calls=len(rows), targets=120, cold_boots=12,
        synthetic=synthetic, software_sha256=next(iter(software)))
