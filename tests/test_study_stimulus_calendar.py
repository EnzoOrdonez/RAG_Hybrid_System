import copy
from collections import Counter

import pytest

from scripts.study_operator.stimulus_calendar import TASKS, V2_VERSION, analyze, calendar


CONFIG = dict(tasks=dict(T1=list(TASKS[:3]), T2=list(TASKS[3:])), labels=dict(A='hybrid', B='no_rag'))


def census():
    rows = []
    for boot, slots in calendar(CONFIG).items():
        for index, slot in enumerate(slots, 1):
            key = str(slot.get('query_id', 'synthetic free'))+slot['condition']
            rows.append(dict(boot_index=boot, index=index, slot=slot, boot_id='cold-'+str(boot),
                software_sha256='a'*64, status='success', valid=True, response_class_version=V2_VERSION,
                raw_text='synthetic raw '+key, answer='synthetic answer '+key,
                response_class='supported', citations=['synthetic citation '+key]))
    return rows


def test_calendar_has_all_literal_tasks_histories_and_120_targets_144_calls():
    plan = calendar(CONFIG)
    rows = [r for slots in plan.values() for r in slots]
    assert len(rows) == 144
    assert all(slots[0]['history'] == 'first_after_boot' for slots in plan.values())
    assert Counter(r['history'] for r in rows) == dict(first_after_boot=12, gate_calendar=36,
        latin_cell=48, arbitrary_free=24, after_arbitrary_free=24)
    assert {r['cell'] for r in rows if r['history'] == 'latin_cell'} == {1, 2, 3, 4}
    for boot, slots in plan.items():
        for i, row in enumerate(slots):
            if row['history'] == 'after_arbitrary_free':
                assert slots[i-1]['role'] == 'antecedent' and slots[i-1]['condition'] == row['condition']


def test_complete_dry_census_cannot_concede_real_acceptance():
    result = analyze(census(), CONFIG, synthetic=True)
    assert result['status'] == 'SYNTHETIC_NOT_ACCEPTANCE' and result['accepted_combinations'] == 12


@pytest.mark.parametrize('tamper', ['missing_predecessor', 'duplicate', 'history', 'software', 'boot', 'class', 'failure', 'invalid'])
def test_census_rejects_defects_before_variant_decision(tamper):
    rows = census()
    if tamper == 'missing_predecessor':
        rows.pop(next(i for i, r in enumerate(rows) if r['slot']['role'] == 'antecedent'))
    elif tamper == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif tamper == 'history':
        rows[0]['slot'] = dict(rows[0]['slot'], history='unregistered')
    elif tamper == 'software':
        rows[-1]['software_sha256'] = 'b'*64
    elif tamper == 'boot':
        for row in rows:
            if row['boot_index'] == 12:
                row['boot_id'] = 'cold-1'
    elif tamper == 'class':
        rows[0]['response_class_version'] = 'old'
    elif tamper == 'failure':
        rows[0]['status'] = 'error'
    else:
        rows[0]['valid'] = False
    with pytest.raises(ValueError):
        analyze(rows, CONFIG, synthetic=True)


@pytest.mark.parametrize('field,value', [('raw_text', 'changed raw'), ('answer', 'changed answer'),
                                      ('response_class', 'changed class'), ('citations', ['changed citation'])])
def test_variation_blocks_real_acceptance(field, value):
    rows = census()
    rows[0][field] = value
    result = analyze(rows, CONFIG)
    assert result['status'] == 'BLOQUEADO-HUMANO' and result['accepted_combinations'] == 11


def test_calendar_never_accepts_task_or_condition_substitution():
    changed = copy.deepcopy(CONFIG)
    changed['tasks']['T1'][2] = 'q178'
    with pytest.raises(ValueError):
        calendar(changed)
