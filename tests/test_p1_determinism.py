"""P1 (R-05): el bootstrap de exp17 no puede depender de PYTHONHASHSEED ni del orden de inserción."""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/results/exp17_crosscloud_balanced'
SNIPPET = (
    "import json,sys; sys.path.insert(0,'.'); "
    "from scripts.compute_exp17_powered import load_counts, cluster_bootstrap; "
    "c=load_counts('faithfulness_rows__hhem.json'); "
    "print(json.dumps(cluster_bootstrap(c, n_boot=400), sort_keys=True))"
)


def bootstrap_under(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, '-c', SNIPPET], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True, timeout=300).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.skipif(not (EXP/'faithfulness_rows__hhem.json').exists(), reason='exp17 rows no presentes')
def test_exp17_cluster_bootstrap_is_identical_across_hash_seeds():
    results = [bootstrap_under(seed) for seed in (0, 7, 42)]
    assert results[0] == results[1] == results[2]


def test_exp17_cluster_bootstrap_ignores_dict_insertion_order():
    from scripts.compute_exp17_powered import cluster_bootstrap
    counts = {f'q{i:02d}': {'baseline': (i % 4, 4 + i % 3), 'balanced': ((i * 7) % 5, 5 + i % 2)} for i in range(25)}
    reversed_counts = dict(reversed(list(counts.items())))
    assert cluster_bootstrap(counts, n_boot=300) == cluster_bootstrap(reversed_counts, n_boot=300)


def test_exp17_load_counts_returns_sorted_query_ids():
    from scripts import compute_exp17_powered as module
    if not (EXP/'faithfulness_rows__hhem.json').exists():
        pytest.skip('exp17 rows no presentes')
    counts = module.load_counts('faithfulness_rows__hhem.json')
    assert list(counts) == sorted(counts)


def test_exp17_recompute_never_targets_the_signed_artifacts():
    from scripts import compute_exp17_powered as module
    source = inspect.getsource(module.main)
    assert 'powered_reanalysis_v2.json' in source and 'powered_reanalysis_v2.md' in source
    assert '"powered_reanalysis.json"' not in source and '"powered_reanalysis.md"' not in source
    assert 'guard_write(' in source
