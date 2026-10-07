from scripts.study_operator.quality_baseline import compare


def test_no_global_pass_when_baseline_is_red(tmp_path):
    row = dict(filename=str(tmp_path / 'frozen.py'), code='F401', message='unused')
    result = compare([row], [row], tmp_path)
    assert result['status'] == 'NO_NEW_FINDINGS'
    assert result['global_ruff'] == 'FAILED'
    assert result['inherited_findings'] == 1


def test_new_occurrence_of_same_message_not_hidden(tmp_path):
    row = dict(filename=str(tmp_path / 'file.py'), code='F401', message='unused')
    result = compare([row], [row, row], tmp_path)
    assert result['status'] == 'NEW_FINDINGS_FAILED' and result['new_findings'] == 1


def test_new_file_and_removed_finding_visible(tmp_path):
    old = dict(filename=str(tmp_path / 'a.py'), code='F401', message='unused')
    new = dict(old, filename=str(tmp_path / 'b.py'))
    result = compare([old], [new], tmp_path)
    assert result['new_findings'] == result['removed_findings'] == 1
