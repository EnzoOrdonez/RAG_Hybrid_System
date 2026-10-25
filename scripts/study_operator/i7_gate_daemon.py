"""Private gate worker; same isolated query/service boundary as the stimulus."""
import io
import json
import logging
import os
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from scripts.study_operator.i7_gate import calendar
from scripts.study_operator.stimulus_host_evidence import sha


def answer_slot(store, slot, index, factory, backup, *, clock=None, capture=None):
    """Use the unchanged application timer, including durable begin and readiness."""
    from src.ui.components import study_service
    from src.ui.components.study_protocol import CELLS
    protocol=store.protocol
    task_set=next(k for k,v in protocol['config']['tasks'].items() if slot['query_id'] in v)
    cell=next(k for k,blocks in CELLS.items() if blocks[0][1]==task_set
              and protocol['config']['labels'][blocks[0][0]]==slot['condition'])
    session=store.admit(store.issue(f'P{900000+index}',cell=cell,profile='without_experience'))
    session.familiarization_done()
    session.data['task_index']=protocol['config']['tasks'][task_set].index(slot['query_id'])
    session.save()
    kwargs={'capture':capture}
    if clock is not None:
        kwargs['clock']=clock
    try:
        study_service.answer(session,factory,**kwargs)
        attempt=dict(session.pending)
    finally:
        # Technical exports deliberately require an explicit host-agent backup.
        store.abandon(session.session_id)
    receipt=backup(store.root/session.session_id)
    good=attempt['error'] is None
    return {'status':'success' if good else 'failed','valid':good,'error':attempt['error'],
            'elapsed_s':attempt['elapsed_ms']/1000 if attempt['elapsed_ms'] is not None else None,
            'backup':receipt,'session_id':session.session_id,
            'answer':attempt.get('answer'), 'citations':attempt.get('sources',[])}


def live_prepare(deployment):
    if os.name!='posix' or os.environ.get('CLOUDRAG_ISOLATED_APP')!='1':
        raise ValueError('ISOLATED_LINUX_GATE_REQUIRED')
    from scripts import cloud_entrypoint
    from scripts.environment_identity import load, verify
    from src.ui.components.interview_preparation import Preparation
    from src.ui.components.study_pipeline import build_study_pipeline, configure_study_device
    from src.ui.components.study_sessions import StudyStore
    cloud_entrypoint.configure(deployment)
    protocol=cloud_entrypoint.verify(deployment)
    before=verify(deployment)
    inventory=load(deployment)
    configure_study_device()
    store=StudyStore('/sessions',protocol,purpose='technical')
    store.freeze()
    scope='i7-gate-'+inventory['observed']['boot_id']
    preparation=Preparation(store.root/'_preparation',factory=build_study_pipeline,
                            systems=('hybrid','no_rag'),nli_systems=('hybrid',))
    preparation.prepare(scope)
    if verify(deployment)!=before or load(deployment)!=inventory:
        raise ValueError('GATE_PREPARATION_IDENTITY_CHANGED')
    events=[json.loads(p.read_bytes()) for p in sorted(preparation.root.glob('*.json'))]
    events=[e for e in events if e['scope']==scope]
    def read_state():
        return json.loads(Path(deployment['service_state_path']).read_bytes())
    state=read_state()
    if state.get('phase')!='RESIDENT' or state.get('sequence')!=9:
        raise ValueError('EXACTLY_TWO_WARMUP_GENERATIONS_REQUIRED')
    from scripts.cloud_storage import Bucket, backup_session
    if not os.environ.get('CLOUDRAG_BACKUP_SOCKET'):
        raise ValueError('ISOLATED_HOST_BACKUP_AGENT_REQUIRED')
    return SimpleNamespace(protocol=protocol,inventory=inventory,boot=inventory['observed']['boot_id'],
        store=store,preparation=preparation,events=events,initial=state,read_state=read_state,
        validate=lambda: verify(deployment)==before and load(deployment)==inventory,
        factory=lambda c:preparation.pipeline(c,scope),
        backup=lambda folder:backup_session(folder,Bucket(deployment['bucket']),deployment['backup_prefix']))


def serve(deployment, *, incoming=None, outgoing=None, prepare=live_prepare):
    incoming, outgoing = incoming or sys.stdin, outgoing or sys.stdout
    from scripts.study_operator.generation_observer import observe as generation
    from scripts.study_operator.rerank_observer import observe as rerank
    from scripts.study_operator.rerank_observer import report
    from scripts.study_operator.stimulus_evidence import OPTIONS, software_projection
    def send(value):
        outgoing.write(json.dumps(value, ensure_ascii=False)+'\n')
        outgoing.flush()
    disabled = logging.root.manager.disable
    try:
        if deployment.get('purpose') != 'technical':
            raise ValueError('TECHNICAL_GATE_ONLY')
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            base = prepare(deployment)
        tasks = base.protocol['config']['tasks']
        slots = calendar(tasks['T1']+tasks['T2'])
        send({'status': 'PREPARED_NOT_ADMITTED', 'inventory': base.inventory, 'slots': 140,
                  'initial_service_state': base.initial, 'preparation_events':base.events})
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
            if index in (21,81):
                if json.loads(incoming.readline(129))!={'operation':'revalidate','index':index}:
                    raise ValueError('WINDOW_REVALIDATION_REQUIRED')
                if not base.validate():
                    raise ValueError('WINDOW_IDENTITY_CHANGED')
                for condition in ('hybrid','no_rag'):
                    base.factory(condition)
                send({'status':'WINDOW_REVALIDATED','index':index,'inventory_sha256':sha(base.inventory)})
            if json.loads(incoming.readline(129)) != {'index': index}:
                raise ValueError('ORDERED_GATE_INDEX_REQUIRED')
            pipeline = base.preparation.pipelines[slot['condition']]
            requests, reranks, captured = [], [], []
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), ExitStack() as stack:
                stack.enter_context(generation(pipeline.llm, requests))
                stack.enter_context(rerank(pipeline, reranks))
                attempt=answer_slot(base.store,slot,index,base.factory,base.backup,capture=captured.append)
            if not attempt['valid']:
                rows.append(dict(slot,index=index,identity=identity,**attempt))
                send({'status':'OBSERVATION','row':rows[-1]})
                return 1
            if len(requests) != 1 or requests[0]['generation_options'] != OPTIONS or len(captured) != 1:
                raise ValueError('FROZEN_GATE_GENERATION_REQUIRED')
            response = captured[0]
            row = dict(slot, index=index, identity=identity, **attempt,
                service_after=base.read_state(), **requests[0],
                raw_text=response.llm_response.text,
                contexts=[item['chunk_id'] for item in response.retrieved_chunks],
                stages=response.latency.model_dump(mode='json'), rerank=report(reranks, response))
            state = row['service_after']
            if state['phase'] != 'RESIDENT' or state['sequence'] != 9+4*index or state['boot_id'] != base.boot:
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
