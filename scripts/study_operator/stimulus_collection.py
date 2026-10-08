"""Cold-history collection with the frozen query path and no generation warmup."""
from contextlib import ExitStack
import io
import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import time

from scripts.study_operator.generation_observer import observe as observe_generation
from scripts.study_operator.rerank_observer import observe as observe_rerank, report
from scripts.study_operator.stimulus_calendar import V2_VERSION, calendar
from scripts.study_operator.stimulus_evidence import OPTIONS, digest, software_projection, verify_boot
from src.evaluation.decline_classifier import classify_response
from src.ui.components.session_storage import atomic_json
from src.ui.components.study_service import execute_query

_LIVE_BINDING = object()


def cold_state(state, boot):
    if (state.get('schema_version') != 1 or state.get('mode') != 'fresh_runner'
            or state.get('phase') != 'STARTING' or state.get('sequence') != 1
            or state.get('boot_id') != boot or state.get('request_id') is not None
            or state.get('deadline_monotonic_s') is not None):
        raise ValueError('Cold service history was already consumed or changed')
    return dict(mode='fresh_runner', phase='STARTING', sequence=1, boot_id=boot)


def prepare_without_generation(factory, read_state, boot):
    """Initialize pipelines; the first LLM generation remains a calendar target."""
    cold_state(read_state(), boot)
    pipelines = {condition: factory(condition) for condition in ('hybrid', 'no_rag')}
    cold_state(read_state(), boot)
    return pipelines


class Collector:
    def __init__(self, protocol, inventory, boot_index, pipelines, read_state, *,
                 synthetic=True, clock=time.perf_counter, live_binding=None):
        if not synthetic and live_binding is not _LIVE_BINDING:
            raise ValueError('LIVE requires preparation bound to verified Linux identity')
        self.protocol, self.inventory, self.boot_index = protocol, inventory, boot_index
        self.pipelines, self.read_state, self.clock = pipelines, read_state, clock
        self.boot = inventory['observed']['boot_id']
        self.software = digest(software_projection(inventory))
        self.slots = calendar(protocol['config'])[boot_index]
        self.initial = cold_state(read_state(), self.boot)
        self.rows, self.terminal, self.synthetic = [], False, synthetic

    def call(self, index):
        if self.terminal or index != len(self.rows)+1 or not 1 <= index <= len(self.slots):
            raise ValueError('Terminal boot or out-of-order slot cannot resume')
        slot = self.slots[index-1]
        question = (slot['question'] if slot['role'] == 'antecedent'
                    else self.protocol['queries'][slot['query_id']]['question'])
        requests, reranks, captured = [], [], []
        pipeline = self.pipelines[slot['condition']]
        began = self.clock()
        disabled = logging.root.manager.disable
        try:
            # Raw synthetic text belongs only in the private, coded boot record.
            logging.disable(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), ExitStack() as stack:
                stack.enter_context(observe_generation(pipeline.llm, requests))
                stack.enter_context(observe_rerank(pipeline, reranks))
                payload, _ = execute_query(slot['condition'], question,
                    lambda condition: self.pipelines[condition], clock=self.clock, capture=captured.append)
            elapsed = self.clock()-began
            if len(requests) != 1 or requests[0]['generation_options'] != OPTIONS or len(captured) != 1:
                raise ValueError('Exactly one frozen generation must be observed')
            response = captured[0]
            raw = response.llm_response.text
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError('Raw generated stimulus is missing')
            row = dict(boot_index=self.boot_index, index=index, slot=slot, boot_id=self.boot,
                software_sha256=self.software, synthetic=self.synthetic, status='success', valid=True,
                raw_text=raw, answer=payload['answer'], citations=payload['sources'],
                response_class=classify_response(payload['answer']), response_class_version=V2_VERSION,
                elapsed_s=elapsed, service_after=self.read_state(), **requests[0],
                contexts=[item['chunk_id'] for item in response.retrieved_chunks],
                stages=response.latency.model_dump(mode='json'), rerank=report(reranks, response))
            self.rows.append(row)
            # The verifier binds the state sequence, boot and runner PID on every call.
            state = row['service_after']
            if (state['phase'] != 'RESIDENT' or state['sequence'] != 1+4*index
                    or state['boot_id'] != self.boot or not 0 < elapsed <= 600):
                raise ValueError('Unconfirmed service transition or call deadline')
            return row
        except BaseException:
            self.terminal = True
            raise
        finally:
            logging.disable(disabled)

    def proof(self):
        return dict(boot_index=self.boot_index, mode='SYNTHETIC' if self.synthetic else 'LIVE',
            status='COMPLETE' if len(self.rows) == len(self.slots) and not self.terminal else 'INCOMPLETE_TERMINAL',
            inventory=self.inventory, rows=self.rows, initial_service_state=self.initial,
            identity_verified_live=not self.synthetic)

    def save_complete(self, destination):
        proof = self.proof()
        verify_boot(proof, self.protocol['config'])
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError('Completed boot evidence must never be replaced')
        atomic_json(destination, proof)
        return dict(status='BOOT_COMPLETE_NOT_ACCEPTANCE', calls=len(self.rows),
                    proof_sha256=digest(proof), synthetic=self.synthetic)


def live_prepare(deployment, boot_index):
    """Linux VM only; no model execution on the investigator's Windows computer."""
    import os
    if os.name != 'posix' or os.environ.get('CLOUDRAG_ISOLATED_APP') != '1':
        raise ValueError('Isolated Linux study container required')
    from scripts import cloud_entrypoint
    from scripts.environment_identity import load, verify
    from src.ui.components.study_pipeline import build_study_pipeline, configure_study_device

    cloud_entrypoint.configure(deployment)
    protocol = cloud_entrypoint.verify(deployment)
    observed = verify(deployment)
    inventory = load(deployment)
    configure_study_device()
    def read_state():
        return json.loads(Path(deployment['service_state_path']).read_bytes())
    pipelines = prepare_without_generation(build_study_pipeline, read_state, inventory['observed']['boot_id'])
    if verify(deployment) != observed or load(deployment) != inventory:
        raise ValueError('Live identity changed while preparing')
    return Collector(protocol, inventory, boot_index, pipelines, read_state,
                     synthetic=False, live_binding=_LIVE_BINDING)
