"""Private gate worker; same isolated query/service boundary as the stimulus."""
import io
import json
import logging
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout

from scripts.study_operator.i7_gate import calendar
from scripts.study_operator.stimulus_collection import live_prepare
from scripts.study_operator.stimulus_host_evidence import sha


def serve(deployment, *, incoming=None, outgoing=None, prepare=live_prepare):
    incoming, outgoing = incoming or sys.stdin, outgoing or sys.stdout
    from scripts.study_operator.generation_observer import observe as generation
    from scripts.study_operator.rerank_observer import observe as rerank
    from scripts.study_operator.rerank_observer import report
    from scripts.study_operator.stimulus_evidence import OPTIONS, software_projection
    from src.ui.components.study_service import execute_query
    def send(value):
        outgoing.write(json.dumps(value, ensure_ascii=False)+'\n')
        outgoing.flush()
    disabled = logging.root.manager.disable
    try:
        if deployment.get('purpose') != 'technical':
            raise ValueError('TECHNICAL_GATE_ONLY')
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            base = prepare(deployment, 1)
        tasks = base.protocol['config']['tasks']
        slots = calendar(tasks['T1']+tasks['T2'])
        send({'status': 'PREPARED_NOT_ADMITTED', 'inventory': base.inventory, 'slots': 140,
                  'initial_service_state': base.initial})
        admission = json.loads(incoming.readline(129))
        import re
        if (set(admission) != {'operation','receipt_sha256'} or admission['operation'] != 'admit'
                or not re.fullmatch('[a-f0-9]{64}', admission['receipt_sha256'])):
            raise ValueError('HOST_ADMISSION_REQUIRED')
        send({'status': 'HOST_ADMISSION_ACKNOWLEDGED', 'receipt_sha256': admission['receipt_sha256']})
        rows = []
        identity = {'boot_id': base.boot, 'image_id': base.inventory['image']['image_id'],
                        'software_sha256': sha(software_projection(base.inventory))}
        for index, slot in enumerate(slots, 1):
            if json.loads(incoming.readline(129)) != {'index': index}:
                raise ValueError('ORDERED_GATE_INDEX_REQUIRED')
            pipeline = base.pipelines[slot['condition']]
            requests, reranks, captured = [], [], []
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), ExitStack() as stack:
                stack.enter_context(generation(pipeline.llm, requests))
                stack.enter_context(rerank(pipeline, reranks))
                payload, elapsed_ms = execute_query(slot['condition'],
                    base.protocol['queries'][slot['query_id']]['question'],
                    lambda condition: base.pipelines[condition], capture=captured.append)
            if len(requests) != 1 or requests[0]['generation_options'] != OPTIONS or len(captured) != 1:
                raise ValueError('FROZEN_GATE_GENERATION_REQUIRED')
            response = captured[0]
            row = dict(slot, index=index, status='success', valid=True, identity=identity,
                elapsed_s=elapsed_ms/1000, service_after=base.read_state(), **requests[0],
                raw_text=response.llm_response.text, answer=payload['answer'], citations=payload['sources'],
                contexts=[item['chunk_id'] for item in response.retrieved_chunks],
                stages=response.latency.model_dump(mode='json'), rerank=report(reranks, response))
            state = row['service_after']
            if state['phase'] != 'RESIDENT' or state['sequence'] != 1+4*index or state['boot_id'] != base.boot:
                raise ValueError('GATE_SERVICE_TRANSITION_DIFFERS')
            rows.append(row)
            send({'status': 'OBSERVATION', 'row': row})
        if json.loads(incoming.readline(129)) != {'operation': 'complete'}:
            raise ValueError('GATE_COMPLETION_REQUIRED')
        send({'status': 'GATE_COMPLETE', 'rows_sha256': sha(rows)})
        return 0
    except BaseException as error:  # noqa: BLE001 - terminal pipe reply also on interruption
        send({'status': 'INCOMPLETE_TERMINAL', 'error_type': type(error).__name__, 'replay_allowed': False})
        return 1
    finally:
        logging.disable(disabled)
