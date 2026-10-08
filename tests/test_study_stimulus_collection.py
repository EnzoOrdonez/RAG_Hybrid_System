import copy
import json
from types import SimpleNamespace

import pytest

from scripts.study_operator.generation_observer import observe
from scripts.study_operator.stimulus_collection import Collector, cold_state, live_prepare, prepare_without_generation
from scripts.study_operator.stimulus_evidence import OPTIONS
from scripts.study_operator.stimulus_calendar import TASKS
from src.generation.llm_manager import LLMManager

CONFIG = dict(tasks=dict(T1=list(TASKS[:3]), T2=list(TASKS[3:])), labels=dict(A='hybrid', B='no_rag'))


def inventory_fixture(boot_index):
    return dict(source=dict(commit='a'*40), recipes={}, dependencies=[], execution_environment={},
        locks={}, vendor={}, ollama={}, artifacts={}, protocol=dict(fingerprint='b'*64),
        image=dict(image_id='sha256:'+'c'*64, container_image_id='sha256:'+'c'*64),
        service=dict(mode='fresh_runner', policy=dict(generation_options=OPTIONS)),
        observed=dict(boot_id='fixture-boot-'+str(boot_index), gpu='GPU-fixture, NVIDIA L4, fixture, 23034 MiB',
            device='1', platform='fixture', preregistration='d'*64, **{'machine-type':'fixture/g2-standard-4'}))


class Client:
    def __init__(self):
        self.requests = []
        self.failure = False

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        if self.failure:
            raise RuntimeError('private synthetic query must not reach logs')
        return dict(message=dict(content='synthetic raw response'))


def manager():
    value = LLMManager(provider='ollama', model='granite4.1:8b', cache_enabled=False,
                       enforce_timeout=True, num_ctx=4096)
    value._ollama_client = Client()
    return value


def test_client_observer_passes_exact_generation_and_restores_real_client():
    value = manager()
    original = value._ollama_client
    requests = []
    options = dict(OPTIONS)
    options.pop('num_ctx')
    values = dict(model=value.model, messages=[dict(role='user', content='private synthetic text')],
                  options=options, keep_alive='30m')
    before = copy.deepcopy(values)
    with observe(value, requests):
        result = value._ollama_chat(None, **values)
    assert result == dict(message=dict(content='synthetic raw response'))
    assert values == before
    assert original.requests == [dict(values, options=OPTIONS)]
    assert value._ollama_client is original and '_ollama_chat' not in vars(value)
    assert requests[0]['generation_options'] == OPTIONS
    assert len(requests[0]['request_sha256']) == 64
    assert 'private synthetic text' not in json.dumps(requests)


def test_client_observer_failure_restores_method_and_client():
    value = manager()
    original = value._ollama_client
    original.failure = True
    with pytest.raises(RuntimeError), observe(value, []):
        value._ollama_chat(None, model=value.model, options=OPTIONS)
    assert value._ollama_client is original and '_ollama_chat' not in vars(value)


def initial(boot):
    return dict(schema_version=1, mode='fresh_runner', phase='STARTING', sequence=1,
                boot_id=boot, request_id=None, deadline_monotonic_s=None)


def test_preparation_never_generates_and_rejects_consumed_history():
    state = initial('fixture')
    calls = []
    result = prepare_without_generation(lambda condition: calls.append(condition) or object(),
                                        lambda: state, 'fixture')
    assert list(result) == ['hybrid', 'no_rag'] and calls == ['hybrid', 'no_rag']
    state['sequence'] = 5
    with pytest.raises(ValueError, match='consumed'):
        prepare_without_generation(lambda condition: pytest.fail('must reject before factory'),
                                    lambda: state, 'fixture')


def setup_collector(boot_index=5, clock=None):
    inventory = inventory_fixture(boot_index)
    state = initial(inventory['observed']['boot_id'])
    queries = {q: dict(question='synthetic '+q) for tasks in CONFIG['tasks'].values() for q in tasks}
    protocol = dict(config=CONFIG, queries=queries)
    calls = []

    class Pipeline:
        def __init__(self, condition):
            self.llm, self.reranker = manager(), None
            self.condition = condition

        def query(self, question):
            calls.append((self.condition, question))
            self.llm._ollama_chat(None, model=self.llm.model,
                messages=[dict(role='user', content=question)], options=OPTIONS)
            state.update(phase='RESIDENT', sequence=state['sequence']+4, runner_pids=[100+len(calls)])
            return SimpleNamespace(answer='A complete synthetic response. '*20, sources=[], error=None,
                confidence='HIGH', hallucination_report=None, retrieved_chunks=[],
                llm_response=SimpleNamespace(text='synthetic raw response'),
                latency=SimpleNamespace(model_dump=lambda **kwargs: dict(generation=0.01)))

    pipelines = prepare_without_generation(Pipeline, lambda: state, inventory['observed']['boot_id'])
    values = {} if clock is None else dict(clock=clock)
    return Collector(protocol, inventory, boot_index, pipelines, lambda: state, **values), calls, state


def test_complete_cold_boot_is_bound_but_never_accepts(tmp_path):
    collector, calls, state = setup_collector()
    assert calls == [] and cold_state(state, collector.boot)['sequence'] == 1
    row = collector.call(1)
    assert row['slot']['history'] == 'first_after_boot' and row['synthetic'] is True
    assert row['generation_options'] == OPTIONS and row['raw_text'] == 'synthetic raw response'
    receipt = collector.save_complete(tmp_path/'boot.json')
    assert receipt['synthetic'] is True and receipt['status'] == 'BOOT_COMPLETE_NOT_ACCEPTANCE'
    with pytest.raises(FileExistsError):
        collector.save_complete(tmp_path/'boot.json')
    with pytest.raises(ValueError):
        collector.call(1)


def test_free_predecessors_and_targets_use_exact_shared_query_path():
    collector, calls, _ = setup_collector(4)
    for index in range(1, len(collector.slots)+1):
        collector.call(index)
    for call, slot in zip(calls, collector.slots, strict=True):
        expected = (slot['question'] if slot['role'] == 'antecedent'
                    else collector.protocol['queries'][slot['query_id']]['question'])
        assert call == (slot['condition'], expected)


def test_missing_observation_aborts_and_cannot_resume():
    collector, calls, _ = setup_collector()
    collector.pipelines[collector.slots[0]['condition']].llm._ollama_client.failure = True
    with pytest.raises(RuntimeError):
        collector.call(1)
    assert collector.terminal and len(calls) == 1
    with pytest.raises(ValueError):
        collector.call(1)


def test_elapsed_service_wait_is_included_and_deadline_failure_is_terminal():
    ticks = iter([1.0, 1.1, 1.2, 602.0])
    collector, _, _ = setup_collector(clock=lambda: next(ticks))
    with pytest.raises(ValueError, match='deadline'):
        collector.call(1)
    assert collector.terminal


def test_unattested_live_collector_and_local_live_measurement_are_rejected(monkeypatch):
    monkeypatch.setenv('CLOUDRAG_ISOLATED_APP', '0')
    collector, _, _ = setup_collector()
    with pytest.raises(ValueError, match='LIVE'):
        Collector(collector.protocol, collector.inventory, 5, collector.pipelines,
                  collector.read_state, synthetic=False)
    with pytest.raises(ValueError, match='Linux'):
        live_prepare({}, 5)
