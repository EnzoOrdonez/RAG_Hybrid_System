import json
from types import SimpleNamespace

import pytest

from src.ui.components.study_protocol import load_protocol, sus_score, INVALID_TASKS
from src.ui.components.study_pipeline import build_study_pipeline
from src.pipeline.pipeline_config import LLM_ONLY_NO_RAG
from tests.study_helpers import configured


@pytest.mark.parametrize('values,expected', [([3]*10, 50), ([5, 1]*5, 100), ([1, 5]*5, 0)])
def test_sus_known_vectors(values, expected):
    assert sus_score(values) == expected


@pytest.mark.parametrize('values', [[True]*10, [0]*10, [6]*10, [3]*9, [3.0]*10])
def test_sus_rejects_incomplete_or_nonordinal(values):
    with pytest.raises(ValueError):
        sus_score(values)


def test_complete_assignment_and_paired_tasks(tmp_path):
    _, _, protocol = configured(tmp_path)
    assert len(protocol['assignments']) == 21
    assert protocol['config']['tasks']['T1'] == ['q001', 'q064', 'q171']
    assert len(protocol['fingerprint']) == 64


@pytest.mark.parametrize('invalid', sorted(INVALID_TASKS))
def test_all_invalid_premises_rejected(tmp_path, invalid):
    path, csv_path, protocol = configured(tmp_path)
    protocol['config']['tasks']['T1'][0] = invalid
    path.write_text(json.dumps(protocol['config']), encoding='utf-8')
    with pytest.raises(ValueError, match='invalid-premise'):
        load_protocol(path, csv_path)


@pytest.mark.parametrize('fault', ['sus', 'labels', 'duplicate', 'type', 'practice', 'quotas', 'pii'])
def test_protocol_prefight_rejects_invalid_inputs(tmp_path, fault):
    path, csv_path, p = configured(tmp_path)
    c = p['config']
    if fault == 'sus':
        c['sus']['items'][4] = ''
    elif fault == 'labels':
        c['labels']['B'] = 'hybrid'
    elif fault == 'duplicate':
        c['tasks']['T2'][0] = 'q001'
    elif fault == 'type':
        c['tasks']['T2'][0] = 'q071'
    elif fault == 'practice':
        c['familiarization'] = p['queries']['q001']['question']
    elif fault == 'quotas':
        csv_path.write_text(csv_path.read_text().replace('P01,primary,1,', 'P01,primary,2,'))
    else:
        csv_path.write_text(csv_path.read_text().replace('profile', 'email', 1))
    path.write_text(json.dumps(c), encoding='utf-8')
    with pytest.raises(ValueError):
        load_protocol(path, csv_path)


def test_both_conditions_share_generation_recipe_and_no_rag_never_loads_index():
    options = []
    def llm(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(**kwargs)
    def pipeline(**kwargs):
        return SimpleNamespace(**kwargs)
    control = build_study_pipeline('no_rag', index_factory=lambda: pytest.fail('index used'),
                                  llm_factory=llm, pipeline_factory=pipeline)
    hybrid = build_study_pipeline('hybrid', index_factory=lambda: 'index', llm_factory=llm, pipeline_factory=pipeline)
    assert options[0] == options[1]
    assert control.config.temperature == hybrid.config.temperature == 0
    assert control.config.llm_model == hybrid.config.llm_model == 'granite4.1:8b'
    assert control.config.retrieval_method == 'none' and control.config.reranker is None
    assert LLM_ONLY_NO_RAG.llm_model == 'llama3.1:8b-instruct-q4_K_M'


def test_no_rag_real_pipeline_path_does_not_retrieve_or_rerank(monkeypatch):
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.generation.llm_manager import LLMResponse
    calls = []
    llm = SimpleNamespace(generate=lambda **kw: (calls.append(kw) or LLMResponse(
        text='Synthetic answer', model='granite4.1:8b', provider='ollama', tokens_input=1,
        tokens_output=2, latency_ms=1, from_cache=False)))
    p = build_study_pipeline('no_rag', index_factory=lambda: pytest.fail('index loaded'),
                             llm_factory=lambda **kw: llm, pipeline_factory=RAGPipeline)
    p._retrieve = lambda *a: pytest.fail('retrieval called')
    p._rerank = lambda *a: pytest.fail('reranking called')
    response = p.query('Technical synthetic question')
    assert not response.error and not response.retrieved_chunks and len(calls) == 1
    assert calls[0]['temperature'] == 0
