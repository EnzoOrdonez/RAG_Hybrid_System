import copy
import json

import pytest

from scripts.study_operator.evidence import add
from scripts.study_operator.final_report import CRITERIA, SECTIONS, render


def fixture(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(agent='Codex', model='fixture')))
    (tmp_path/'source.json').write_text('{"synthetic":true}')
    (tmp_path/'receipt.json').write_text(json.dumps(dict(command=['synthetic-test'], exit_code=0)))
    specs = [dict(key='fixture-'+str(i), statement='Synthetic fixture evidence '+str(i), certainty='VERIFICADO',
                  evidence=['source.json'], command_receipts=['receipt.json']) for i in range(1, 8)]
    add(tmp_path, specs)
    plan = dict(sections={section: [] for section in SECTIONS},
                attributes={a: dict(status='NO_MEDIDO', claims=[]) for a in CRITERIA})
    plan['sections']['Resumen ejecutivo'] = ['I5-V'+str(i).zfill(3) for i in range(1, 8)]
    return plan


def test_report_has_exact22_sections11_attributes_and_no_synthetic_acceptance(tmp_path):
    plan = fixture(tmp_path)
    before = copy.deepcopy(plan)
    text = render(tmp_path, plan)
    assert [line[3:] for line in text.splitlines() if line.startswith('## ')] == list(SECTIONS)
    assert text.count('NO_MEDIDO') == 11 and 'GO' not in text and 'Synthetic fixture' in text
    assert plan == before


@pytest.mark.parametrize('defect', ['missing_section', 'extra_section', 'missing_attribute', 'summary_short',
    'unknown_claim', 'duplicate_claim', 'unbacked_observed', 'invented_GO', 'changed_evidence'])
def test_missing_forged_or_modified_report_inputs_rejected(tmp_path, defect):
    plan = fixture(tmp_path)
    if defect == 'missing_section':
        plan['sections'].pop(SECTIONS[-1])
    elif defect == 'extra_section':
        plan['sections']['Unrequested'] = []
    elif defect == 'missing_attribute':
        plan['attributes'].pop(next(iter(CRITERIA)))
    elif defect == 'summary_short':
        plan['sections']['Resumen ejecutivo'].pop()
    elif defect == 'unknown_claim':
        plan['sections']['Falta'] = ['I5-V999']
    elif defect == 'duplicate_claim':
        plan['sections']['Falta'] = ['I5-V001', 'I5-V001']
    elif defect == 'unbacked_observed':
        plan['attributes']['Rendimiento']['status'] = 'OBSERVADO'
    elif defect == 'invented_GO':
        plan['attributes']['Rendimiento']['status'] = 'GO'
    else:
        (tmp_path/'source.json').write_text('tampered')
    with pytest.raises(ValueError):
        render(tmp_path, plan)
