"""Reproductores de la auditoría 2026-11-02: R-05 (exp17 depende del hash seed) y R-06 (escritores sin guarda).

Solo leen evidencia; no ejecutan escritores. Marcados xfail(strict=True) hasta aplicar
docs/review/2026-11-02/patches/ (DECISIONES.md D-04).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP17 = ROOT / "experiments" / "results" / "exp17_crosscloud_balanced"

PROBE = (
    "import json, scripts.compute_exp17_powered as m\n"
    "c = m.load_counts('faithfulness_rows__hhem.json')\n"
    "print(json.dumps(m.cluster_bootstrap(c, n_boot=2000), sort_keys=True))\n"
)


def _bootstrap_under(hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120, check=True)
    return out.stdout.strip()


@pytest.mark.skipif(not (EXP17 / "faithfulness_rows__hhem.json").exists(), reason="exp17 rows absent")
@pytest.mark.xfail(strict=True, reason="R-05 (S1): compute_exp17_powered.py:46,82 itera un set; el bootstrap depende de PYTHONHASHSEED")
def test_exp17_cluster_bootstrap_is_independent_of_hash_seed():
    assert _bootstrap_under("0") == _bootstrap_under("42")


# Escritores de nombre fijo sobre experiments/results que no pasan por guard_write (verificado por grep
# y por revisión adversarial; rescore_nli_v3 solo sobrescribe con --variant vb_agree).
UNGUARDED = [
    "scripts/recompute_retrieval_stats.py",
    "scripts/compute_exp17_powered.py",
    "scripts/compute_exp18_powered.py",
    "scripts/compute_exp18_diagnosis.py",
    "scripts/compute_exp18_selection_bound.py",
    "scripts/compute_tierA_arm_stats.py",
    "scripts/compute_exp15_ensemble_sweep.py",
    "scripts/compute_exp15_nli_sweep.py",
    "scripts/compute_exp15_hhem_analysis.py",
    "scripts/compute_exp16_guards.py",
    "scripts/compute_exp19b_stats.py",
    "scripts/rescore_nli_v3.py",
]


@pytest.mark.parametrize("script", UNGUARDED)
@pytest.mark.xfail(strict=True, reason="R-04/R-06 (S1): escritor de evidencia congelada sin guard_write")
def test_evidence_writer_calls_guard_write(script):
    assert "guard_write(" in (ROOT / script).read_text(encoding="utf-8")
