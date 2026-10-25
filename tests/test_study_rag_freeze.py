import copy
import json

import pytest

from scripts.study_operator.rag_freeze import compare, inherited_contexts


def contexts_fixture(root):
    for position in range(1, 13):
        stem = f'w1-{position:03}'
        (root / f'deployment--candidate--gate-cohort-worker--requests--{stem}-pipeline.json').write_text(
            json.dumps(dict(retrieved_chunks=[dict(chunk_id=str(position))])))
        (root / f'deployment--candidate--gate-cohort--window-1--attempts--{position:03}-result.json').write_text(
            json.dumps(dict(query_id=f'q{(position-1)//2:03}', condition='hybrid' if position % 2 else 'no_rag')))


def test_twelve_historical_reads_are_hash_bound_but_not_live_proof(tmp_path):
    contexts_fixture(tmp_path)
    values, refs = inherited_contexts(tmp_path)
    assert len(values) == len(refs) == 12
    assert all(len(ref['sha256']) == len(ref['result_sha256']) == 64 for ref in refs)
    projection = dict(rag=dict(contexts=values), provenance=dict(live_contexts_verified=False))
    assert compare(projection, projection)['rag_projection_equal']
    with pytest.raises(ValueError, match='Live'):
        compare(projection, projection, require_live=True)
    final = copy.deepcopy(projection)
    final['rag']['contexts']['q000|hybrid'] = ['changed']
    with pytest.raises(ValueError, match='Frozen'):
        compare(projection, final)


def test_incomplete_and_variant_gate_reads_fail(tmp_path):
    with pytest.raises(ValueError, match='Twelve'):
        inherited_contexts(tmp_path)
    contexts_fixture(tmp_path)
    (tmp_path / 'deployment--candidate--gate-cohort-worker--requests--w2-001-pipeline.json').write_text(
        json.dumps(dict(retrieved_chunks=[dict(chunk_id='changed')])))
    (tmp_path / 'deployment--candidate--gate-cohort--window-2--attempts--001-result.json').write_text(
        json.dumps(dict(query_id='q000', condition='hybrid')))
    with pytest.raises(ValueError, match='variants'):
        inherited_contexts(tmp_path)
