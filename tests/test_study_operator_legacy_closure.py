from datetime import datetime, timezone
import hashlib
import json

import pytest

from scripts.study_operator.legacy_closure import closure_guard, finalize_report, permitted_task_preservation, removable_tasks, report_sections, verify_claims


def test_all_four_real_closure_receipts_required():
    rows = [dict(status='SAFE_RESOURCES_VERIFIED', errors=[]),
            dict(status='SAFE_RESOURCE_CLOSURE_AUDITED_NOT_PARTICIPANT_ACCEPTANCE'),
            dict(status='PASS_FINAL_FUNCTIONAL_CONTROLS_STATIC_FREEZE_ONLY'),
            dict(status='ABSENT_OWNED_IP_RECONCILED_ONCE_AFTER_SAFE_CLOSURE')]
    closure_guard(*rows)
    for i in range(4):
        changed = list(rows)
        changed[i] = dict(rows[i], status='FAILED')
        with pytest.raises(ValueError):
            closure_guard(*changed)
    with pytest.raises(ValueError):
        closure_guard(dict(rows[0], errors=['failure']), *rows[1:])


def test_ready_manual_task_is_terminal_but_future_or_running_is_pending():
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    row = dict(name='owned', state='Ready', next_run_utc=None)
    assert removable_tasks([row], {'owned'}, now) == ['owned']
    for bad in [dict(row, state='Running'), dict(row, next_run_utc='2026-10-08T00:00:00+00:00')]:
        with pytest.raises(ValueError):
            removable_tasks([bad], {'owned'}, now)
    with pytest.raises(ValueError):
        removable_tasks([row], set(), now)
    with pytest.raises(ValueError):
        removable_tasks([row, row], {'owned'}, now)


def test_claim_hash_verifier_detects_same_size_tampering_without_repair(tmp_path):
    evidence = tmp_path / 'evidence'
    evidence.write_bytes(b'pass')
    sha = hashlib.sha256(b'pass').hexdigest()
    ledger = tmp_path / 'CLAIMS_LEDGER.md'
    ledger.write_text(f'V001 | original `quoted description` | `{evidence}` | SHA-256 `{sha}` | `verify`\n', encoding='utf-8')
    assert verify_claims(tmp_path)['checked'] == 1
    evidence.write_bytes(b'fail')
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        verify_claims(tmp_path)
    assert evidence.read_bytes() == b'fail' and ledger.read_bytes() == before


def test_preservation_requires_proven_limited_denial_never_infers_removal():
    denied = b'Unregister-ScheduledTask HRESULT 0x80070005'
    result = permitted_task_preservation(dict(exit_code=1), denied)
    assert result['status'] == 'BLOQUEADO-HUMANO' and result['no_elevation_or_acl_change']
    for code, stderr in [(0, denied), (1, b'Unexpected parser failure'), (1, b'0x80070005 unrelated')]:
        with pytest.raises(ValueError):
            permitted_task_preservation(dict(exit_code=code), stderr)


def test_twenty_multiline_sections_are_separate_not_one_greedy_match():
    text = '# Cierre\n\n' + ''.join(f'## Sección {i}\n\nLínea uno\nLínea dos\n\n' for i in range(20))
    sections = report_sections(text)
    assert len(sections) == 20 and sections[0][0] == 'Sección 0'
    assert 'Línea dos' in sections[0][1] and sections[-1][0] == 'Sección 19'
    with pytest.raises(ValueError):
        report_sections(text.replace('## Sección 19', '# Sección 19'))


def test_partial_report_resumes_without_repeating_cloud_effects_or_duplicate_claims(tmp_path):
    root, package = tmp_path / 'old', tmp_path / 'new'
    root.mkdir()
    package.mkdir()
    audit = dict(at='2026-10-06T20:50:00+00:00', cost_estimate_usd=12,
                 image_egress_upper_separate_usd=.5, initial_margin_separate_usd=2.72,
                 current_no_IP_idle_upper_usd_day=1.15)
    audit_file = root / 'closure-audit01.json'
    audit_file.write_text(json.dumps(audit))
    sha = hashlib.sha256(audit_file.read_bytes()).hexdigest()
    (root / 'CLAIMS_LEDGER.md').write_text(f'V001 | fixture | `{audit_file}` | SHA-256 `{sha}` | `verify`\n')
    (root / 'STATE.json').write_text(json.dumps(dict(deadline_utc='2026-10-06T22:28:29+00:00')))
    (root / 'RECOVERY_ITERATION5.json').write_text(json.dumps(dict(at='2026-10-07T02:45:00+00:00',
        inherited_receipts={'closure-audit01.json': sha}, old_tasks_quiescent_no_future_trigger=True,
        original_deadline_utc='2026-10-06T22:28:29+00:00', task_retirement_block=None,
        inherited_claims_verified=1)))
    (root / 'phase-command-supervisor-recovery-i5-03.json').write_text('{}')
    (root / 'REPORT_WORKING05.md').write_text(''.join(f'## Sección {i}\n\nCheckpoint\n' for i in range(20)), encoding='utf-8')
    (root / 'DATOS_B4_V6_WORKING04.md').write_text('Technical fixture')
    finalize_report(root, package)
    assert len(report_sections((root / 'REPORT_FINAL.md').read_text(encoding='utf-8'))) == 20
    assert not json.loads((root / 'STATE.json').read_bytes())['original_deadline_met']
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    finalize_report(root, package)
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
