"""Private JSON pipe for one cold boot; no public route and no model replay."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
import re
import sys

from scripts.study_operator.stimulus_collection import live_prepare
from scripts.study_operator.stimulus_evidence import verify_boot


def serve(deployment, boot_index, *, incoming=None, outgoing=None, prepare=live_prepare):
    incoming, outgoing = incoming or sys.stdin, outgoing or sys.stdout
    disabled = logging.root.manager.disable
    terminal = False
    admitted = False
    admission_sha = None

    def send(value):
        outgoing.write(json.dumps(value, ensure_ascii=False)+'\n')
        outgoing.flush()

    try:
        if deployment.get('purpose') != 'technical':
            raise ValueError('Synthetic technical purpose required')
        # Silence library output during loading as well as each content query.
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            collector = prepare(deployment, boot_index)
        send(dict(status='PREPARED_NOT_ADMITTED', inventory=collector.inventory,
                  initial_service_state=collector.initial, slots=len(collector.slots)))
        for line in incoming:
            if len(line) > 128:
                raise ValueError('Control message exceeds bound')
            request = json.loads(line)
            if (not admitted and set(request) == {'operation','receipt_sha256'}
                    and request['operation'] == 'admit'
                    and re.fullmatch('[a-f0-9]{64}', request['receipt_sha256'])):
                from scripts.study_operator.service_transition import cold_state

                cold_state(collector.read_state(), collector.boot)
                admitted = True
                admission_sha = request['receipt_sha256']
                send(dict(status='HOST_ADMISSION_ACKNOWLEDGED', receipt_sha256=request['receipt_sha256']))
            elif admitted and set(request) == {'index'} and type(request['index']) is int:
                row = collector.call(request['index'])
                send(dict(status='OBSERVATION', row=row))
            elif admitted and request == {'operation':'complete'}:
                proof = collector.proof()
                proof['host_admission_sha256'] = admission_sha
                verify_boot(proof, collector.protocol['config'], require_host=False)
                send(dict(status='COMPLETE_NOT_ACCEPTANCE', proof=proof))
                return 0
            else:
                raise ValueError('Only ordered indices and verified completion are allowed')
        raise ValueError('EOF before complete boot is terminal')
    except BaseException as error:
        terminal = True
        send(dict(status='INCOMPLETE_TERMINAL', error_type=type(error).__name__, replay_allowed=False))
        return 1
    finally:
        logging.disable(disabled)
        if terminal:
            # The owning host retains all observations received before failure.
            # No query or private exception message is echoed by this process.
            outgoing.flush()
