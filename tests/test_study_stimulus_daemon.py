import io
import json

import pytest

from scripts.study_operator.stimulus_daemon import serve
from scripts.study_operator.stimulus_evidence import OPTIONS

ADMIT = dict(operation='admit', receipt_sha256='a'*64)


def prepared(deployment, boot_index):
    # Import the reusable measurement fixtures through pytest's test path,
    # not the product bundle. No real pipeline or backend is initialized.
    from test_study_stimulus_collection import setup_collector
    collector, _, _ = setup_collector(boot_index)
    return collector


def execute(messages, **kwargs):
    incoming = io.StringIO(''.join(json.dumps(row)+'\n' for row in messages))
    outgoing = io.StringIO()
    status = serve(dict(purpose='technical'), 5, incoming=incoming, outgoing=outgoing,
                   prepare=prepared, **kwargs)
    return status, [json.loads(line) for line in outgoing.getvalue().splitlines()]


def test_complete_private_pipe_is_synthetic_and_never_acceptance():
    status, messages = execute([ADMIT,dict(index=1),dict(operation='complete')])
    assert status == 0 and messages[0]['status'] == 'PREPARED_NOT_ADMITTED'
    assert messages[1]['status'] == 'HOST_ADMISSION_ACKNOWLEDGED'
    assert messages[2]['row']['generation_options'] == OPTIONS
    assert messages[-1]['status'] == 'COMPLETE_NOT_ACCEPTANCE'
    assert messages[-1]['proof']['mode'] == 'SYNTHETIC'


@pytest.mark.parametrize('messages', [[],[dict(index=2)],[dict(index=True)],
    [dict(operation='complete')],[dict(operation='replay')],[ADMIT,dict(index=1),dict(index=1)],
    [ADMIT,dict(operation='complete')]])
def test_incomplete_or_altered_calendar_is_terminal(messages):
    status, result = execute(messages)
    assert status == 1 and result[-1]['status'] == 'INCOMPLETE_TERMINAL'
    assert result[-1]['replay_allowed'] is False
    assert not any(row['status']=='COMPLETE_NOT_ACCEPTANCE' for row in result)


def test_study_purpose_cannot_start_measurement_or_print_private_error():
    output = io.StringIO()
    def wrong(*args):
        pytest.fail('Study must reject before preparation')
    assert serve(dict(purpose='study'),5,incoming=io.StringIO(),outgoing=output,prepare=wrong)==1
    assert 'study' not in output.getvalue().lower()
    assert json.loads(output.getvalue())['error_type'] == 'ValueError'
