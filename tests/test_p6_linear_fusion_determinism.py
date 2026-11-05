"""P6: _linear_fusion must not depend on PYTHONHASHSEED when scores tie (all_ids is a set).

RRF is checked too: it was already deterministic (dict insertion order) and must stay so.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = r'''
import json, sys
sys.path.insert(0, '.')
from src.embedding.index.hybrid_index import HybridIndex
ix = object.__new__(HybridIndex)
bm = [('doc%d' % i, 1.0) for i in range(6)]
de = [('doc%d' % (i + 3), 1.0) for i in range(6)]
print(json.dumps({'linear': [c for c, _ in ix._linear_fusion(bm, de, 0.5, 4)],
                  'rrf': [c for c, _ in ix._rrf_fusion(bm, de, 60, 4)]}))
'''


def fused_under(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, '-c', SNIPPET], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True, timeout=300).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_linear_fusion_ties_are_identical_across_hash_seeds():
    runs = [fused_under(seed) for seed in (0, 7, 42, 1)]
    assert all(r['linear'] == runs[0]['linear'] for r in runs), [r['linear'] for r in runs]
    assert all(r['rrf'] == runs[0]['rrf'] for r in runs)


def test_linear_fusion_breaks_ties_by_chunk_id_and_keeps_score_order():
    from src.embedding.index.hybrid_index import HybridIndex
    ix = object.__new__(HybridIndex)
    bm = [('b', 1.0), ('a', 1.0), ('z', 0.0)]
    de = [('c', 1.0), ('z', 1.0)]
    result = ix._linear_fusion(bm, de, 0.5, 4)
    # normalized: a=b=0.5 (bm25 only), c=0.5 (dense only), z=0.5 (bm25 min 0 + dense 1) -> all tie
    assert [c for c, _ in result] == ['a', 'b', 'c', 'z']
    higher = ix._linear_fusion([('y', 2.0), ('x', 0.0)], [('y', 1.0), ('x', 0.0)], 0.5, 2)
    assert [c for c, _ in higher] == ['y', 'x'] and higher[0][1] == pytest.approx(1.0)
