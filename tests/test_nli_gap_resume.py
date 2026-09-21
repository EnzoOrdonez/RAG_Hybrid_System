"""Interrupted paired observations remain immutable; never infer or impute a mate."""
import copy

import pytest

from scripts import measure_interview_gate as gate
from scripts import nli_batch_experiment as experiment
from scripts import unattended_diagnostic as unattended


def row(position):
    system, index, arm = experiment.schedule()[position]
    result = unattended.synthetic_row(system, index)
    result.update(arm=arm, attempt_id=f'a{position:03d}', window_id='first', started_at=f'{position:04d}')
    return result


def publish(root, result):
    path = root / 'cohort/attempts' / result['attempt_id'] / 'result.json'
    gate.write_new(path, result)
    return path


def initialize(root, restored=True):
    gate.write_new(root / 'cohort/source-manifest.json', dict(protocol=dict(
        nli_experiment=experiment.EXPERIMENT, interrupted_pair_policy=experiment.GAP_POLICY,
        quality_source_ids=[str(i) for i in range(40)])))
    gate.write_new(root / 'windows/first/window.json', dict(simulated=True))
    if restored:
        gate.write_new(root / 'windows/first/restored.json', dict(simulated=True))


def partial(root, end=3):
    initialize(root)
    for i in range(end):
        publish(root, row(i))
    return unattended.package(root)


def finish_with_gap(root):
    partial(root)
    for i in range(4, 120):
        publish(root, dict(row(i), window_id='second'))
    return unattended.package(root)


def test_gap_append_only_repack_preserves_attempt_and_marker_bytes(tmp_path):
    partial(tmp_path)
    originals = {p: p.read_bytes() for p in (tmp_path / 'cohort').rglob('*.json')}
    unattended.package(tmp_path)
    assert all(p.read_bytes() == content for p, content in originals.items())
    unattended.verify_package(tmp_path)
    gaps = experiment.load_gaps(tmp_path / 'cohort')
    assert len(gaps) == 1 and gaps[0]['status'] == 'INCOMPLETO_INTERRUMPIDO'
    assert gaps[0]['missing_arms'] == ['control']
    assert not gaps[0]['imputed']
    assert len(gate.local_records(tmp_path / 'cohort')) == 3


def test_unfinished_live_pair_cannot_be_sealed(tmp_path):
    initialize(tmp_path, restored=False)
    publish(tmp_path, row(0))
    with pytest.raises(RuntimeError, match='Restore window'):
        experiment.seal_interruptions(tmp_path / 'cohort')
    assert not (tmp_path / 'cohort/gaps').exists()


def test_prior_cohort_policy_not_silently_reinterpreted(tmp_path):
    gate.write_new(tmp_path / 'cohort/source-manifest.json', dict(protocol=dict(nli_experiment=experiment.EXPERIMENT)))
    publish(tmp_path, row(0))
    with pytest.raises(ValueError, match='not registered'):
        experiment.seal_interruptions(tmp_path / 'cohort')


@pytest.mark.parametrize('second_started', [False, True])
def test_recovered_abort_has_no_duration_and_no_fake_missing_attempt(tmp_path, second_started):
    initialize(tmp_path)
    if second_started:
        publish(tmp_path, row(0))
    dead = row(int(second_started))
    request = {k: v for k, v in dead.items() if k not in ('response', 'diagnostic_trace', 'status', 'elapsed_s')}
    request_path = tmp_path / 'cohort/attempts' / dead['attempt_id'] / 'request.json'
    gate.write_new(request_path, request)
    original = request_path.read_bytes()
    report = unattended.package(tmp_path)
    recovered = gate.read_json(request_path.with_name('result.json'))
    assert request_path.read_bytes() == original
    assert recovered['status'] == 'aborted' and recovered['elapsed_s'] is None
    marker = report['interrupted_pairs'][0]
    assert len(marker['attempts']) == (2 if second_started else 1)
    assert len(marker['missing_arms']) == (0 if second_started else 1)
    assert not report['pairs']
    assert sum(g['aborted'] for g in report['groups']) == 1
    assert experiment.remaining(gate.local_records(tmp_path / 'cohort'), experiment.load_gaps(tmp_path / 'cohort'))[0] == experiment.schedule()[2]


def test_missing_arm_never_executed_and_orphan_excluded_from_paired_metrics(tmp_path):
    initialize(tmp_path)
    for i in range(3):
        publish(tmp_path, dict(row(i), elapsed_s=999 if i == 2 else .001))
    unattended.package(tmp_path)
    for i in range(4, 120):
        publish(tmp_path, dict(row(i), window_id='second'))
    report = unattended.package(tmp_path)
    assert report['complete'] and report['executed_attempts'] == 119
    assert report['not_executed_slots'] == [experiment.schedule()[3]]
    assert len(report['pairs']) == 59
    assert report['complete_valid_pairs_by_system'] == dict(hybrid=20, lexical=19, semantic=20)
    assert report['sufficiency'] == 'INSUFICIENTE' and not report['latency_pass_candidate']
    paired = next(g for g in report['paired_groups'] if g['system'] == 'lexical' and g['arm'] == 'candidate')
    individual = next(g for g in report['groups'] if g['system'] == 'lexical' and g['arm'] == 'candidate')
    assert paired['n'] == 19 and paired['metrics']['total_s']['maximum'] == .001
    assert individual['metrics']['total_s']['maximum'] == 999
    interval = next(b for b in report['bootstrap'] if b['system'] == 'lexical')
    assert interval['n'] == 19 and interval['status'] == 'insufficient'
    assert interval['fields']['total_s']['delta_p95'] == 0


def test_contaminated_pair_and_interrupted_pair_are_distinct(tmp_path):
    partial(tmp_path)
    for i in range(4, 120):
        r = dict(row(i), window_id='second')
        if i == 8:
            r.update(conditions_invalid=True, control_reasons=['prohibited_process'])
        publish(tmp_path, r)
    report = unattended.package(tmp_path)
    states = [r['status'] for r in report['pair_states']]
    assert states.count(experiment.GAP_STATUS) == 1
    assert states.count('INVALIDO_CONTAMINACION') == 1
    assert states.count('COMPLETO_VALIDO') == 58
    assert len(report['pairs']) == 58


def test_cannot_fill_omitted_arm_after_sealing(tmp_path):
    partial(tmp_path)
    publish(tmp_path, row(3))
    with pytest.raises(ValueError, match='interrupted registered pair|filled'):
        experiment.load_gaps(tmp_path / 'cohort')


def test_hash_change_of_original_attempt_rejected(tmp_path):
    partial(tmp_path)
    source = tmp_path / 'cohort/attempts/a002/result.json'
    data = gate.read_json(source)
    data['elapsed_s'] = 1.5
    unattended.replace_view(source, data)
    with pytest.raises(ValueError, match='hash changed'):
        experiment.load_gaps(tmp_path / 'cohort')
    with pytest.raises(ValueError, match='changed'):
        unattended.verify_package(tmp_path)


def test_markers_cannot_invent_or_duplicate_pairs(tmp_path):
    partial(tmp_path)
    rows = gate.local_records(tmp_path / 'cohort')
    marker = experiment.load_gaps(tmp_path / 'cohort')[0]
    with pytest.raises(ValueError, match='Duplicate'):
        experiment.remaining(rows, [marker, marker])
    invented = dict(marker, system='semantic', index=19)
    with pytest.raises(ValueError, match='registered pair'):
        experiment.remaining(rows, [invented])
    with pytest.raises(ValueError, match='prefix'):
        experiment.remaining([*rows, row(4)])  # hole without its durable marker


def test_quality_uses_only_complete_pairs_and_exhausted_insufficient_cohort_does_not_reopen(tmp_path, monkeypatch, capsys):
    finish_with_gap(tmp_path)
    rows = gate.local_records(tmp_path / 'cohort')
    gaps = experiment.load_gaps(tmp_path / 'cohort')
    for i in range(40):
        gate.write_new(tmp_path / 'cohort/quality' / f'base-{i}.json', dict(passed=True))
    for pair in experiment.complete_pairs(rows, gaps):
        for r in pair:
            gate.write_new(tmp_path / 'cohort/quality' / ('new-'+r['attempt_id']+'.json'), dict(passed=True))
    gate.write_new(tmp_path / 'cohort/quality/new-a002.json', dict(passed=False))
    quality = experiment.quality_summary(tmp_path / 'cohort', rows)
    assert quality['new_expected'] == quality['new_passed'] == 118
    assert quality['replay_complete'] and quality['available_pairs_equivalent'] and not quality['passed']
    assert not quality['failures'] and quality['excluded_failed_replays'] == ['new-a002']
    unattended.package(tmp_path)
    unattended.verify_package(tmp_path)
    monkeypatch.setattr(unattended, 'external_root', lambda p: p)
    monkeypatch.setattr(gate, 'git', lambda *args: 'fix/interview-readiness' if args[0] == 'branch' else '')
    monkeypatch.setattr(unattended, 'preflight', lambda *args, **kwargs: pytest.fail('must not open another window'))
    assert unattended.prepare(tmp_path, resume=True, authorize=True, nli_experiment=True) is None
    assert 'COMPLETE' in capsys.readouterr().out


def test_gap_marker_tamper_is_detected_by_manifest(tmp_path):
    partial(tmp_path)
    marker = next((tmp_path / 'cohort/gaps').glob('*.json'))
    payload = copy.deepcopy(gate.read_json(marker))
    payload['at'] = 'changed'
    unattended.replace_view(marker, payload)
    with pytest.raises(ValueError, match='changed'):
        unattended.verify_package(tmp_path)


def test_synthetic_interruption_resume_authorization_and_integrity(tmp_path):
    checks = experiment.synthetic_gap_run(tmp_path / 'scenario')
    assert all(checks.values())
    assert checks['no_imputation_or_duplicates'] and checks['paired_n_59']
    unattended.verify_package(tmp_path / 'scenario')
