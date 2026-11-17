"""A future GO belongs to its image/source; an earlier GO cannot follow a rebuild."""
import subprocess

import pytest

from tests.test_study_i7_human import closed_phase_b, gate_receipt


@pytest.mark.parametrize('field,replacement', [('image_id', 'sha256:' + 'e' * 64), ('commit', 'd' * 40)])
def test_previous_go_rejects_changed_installation_identity(tmp_path, field, replacement):
    op, cloud = closed_phase_b(tmp_path)
    assert len(op.verified_gate_go()) == 64
    op.config[field] = replacement
    with pytest.raises(ValueError, match='^VERIFIED_GATE_GO_REQUIRED$'):
        op.verified_gate_go()
    assert cloud.calls == []


def test_new_build_and_new_go_can_open_human_phase(tmp_path):
    op, cloud = closed_phase_b(tmp_path)
    old_go = op.verified_gate_go()
    op.config.update(image_id='sha256:' + 'e' * 64, commit='d' * 40)
    op.persist()
    gate_receipt(tmp_path, 'GO')
    new_go = op.verified_gate_go()
    assert new_go != old_go
    assert op.human_phase_open()['status'] == 'HUMAN_PHASE_OPEN'
    assert op.state['human_phase']['gate_receipt_sha256'] == new_go
    assert op.human_start_profile('rehearsal') == {'i7_profile': 'standard'}
    assert cloud.calls == []


def test_new_go_cannot_silently_replace_an_open_human_phase(tmp_path):
    op, cloud = closed_phase_b(tmp_path)
    op.human_phase_open()
    op.config.update(image_id='sha256:' + 'e' * 64, commit='d' * 40)
    op.persist()
    gate_receipt(tmp_path, 'GO')
    assert len(op.verified_gate_go()) == 64
    with pytest.raises(ValueError, match='^GATE_EVIDENCE_CHANGED_SINCE_OPEN$'):
        op.human_start_profile('rehearsal')
    assert cloud.calls == []


@pytest.mark.parametrize('changed,allowed', [
    ('scripts/study_operator/i7.py', False), ('.harness/progress.md', False), ('docs/notes.md', True)])
def test_owner_source_admission_is_separate_from_go(tmp_path, monkeypatch, changed, allowed):
    from scripts.study_operator import i7_workflows
    op, cloud = closed_phase_b(tmp_path)
    assert len(op.verified_gate_go()) == 64
    def git(argv, **kwargs):
        if argv[3:] == ['rev-parse', 'HEAD']:
            return 'd' * 40
        assert argv[3:6] == ['diff', '--name-only', op.config['commit']]
        return changed + '\n'
    monkeypatch.setattr(i7_workflows.subprocess, 'check_output', git)
    monkeypatch.setattr(i7_workflows.subprocess, 'run', lambda *a, **kw:
        subprocess.CompletedProcess([], 0, stdout=b''))
    archived = []
    def archive(source, destination, commit):
        archived.append(commit)
        return {'commit': commit}
    monkeypatch.setattr(i7_workflows, 'archive_tools', archive)
    if allowed:
        assert i7_workflows.verify_tool_source(op) == {'commit': 'd' * 40}
        assert archived == ['d' * 40]
    else:
        with pytest.raises(ValueError, match='^OPERATOR_SOURCE_COMMIT_DIFFERS$'):
            i7_workflows.verify_tool_source(op)
        assert archived == []
    assert cloud.calls == []
