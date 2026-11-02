"""Auditoría 2026-11-02: hallazgos previos que siguen CONFIRMADOS en la base b470d3e.

Cada test expresa la propiedad deseada y lleva xfail(strict=True): hoy falla (defecto vigente) y,
cuando se corrija, pasa a XPASS y rompe la suite para que se quite el marcador.
Ninguno escribe evidencia ni ejecuta modelos.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY_SOURCES = [p for p in (ROOT/'scripts').rglob('*.py')]


@pytest.mark.xfail(strict=True, reason='R-03: la capa 1 no reconoce tokens GCP/Google/HF/GitHub fine-grained')
@pytest.mark.parametrize('token', [
    'ya29.' + 'a0Ae' + 'X'*60,                     # token OAuth de Google
    'AIza' + 'S'*35,                               # clave de API de Google
    'hf_' + 'A'*34,                                # token de Hugging Face
    'github_pat_' + '1'*22 + '_' + 'b'*59,          # token fine-grained de GitHub
])
def test_r03_layer1_detects_common_cloud_tokens(token):
    from scripts.scan_secrets import scan_text
    assert scan_text('value = "%s"\n' % token, Path('notes.md'))


@pytest.mark.xfail(strict=True, reason='R-03: .pem/.key/.env.* se omiten sin inspección ni aviso')
@pytest.mark.parametrize('name', ['deploy/service.pem', 'keys/id.key', 'config/.env.production'])
def test_r03_key_material_paths_are_inspected_not_skipped(name):
    from scripts.scan_secrets import should_exclude
    assert not should_exclude(Path(name))


@pytest.mark.xfail(strict=True, reason='R-04: recompute_retrieval_stats sobrescribe retrieval_metrics.json sin guard_write')
def test_r04_retrieval_recompute_guards_signed_evidence():
    source = (ROOT/'scripts'/'recompute_retrieval_stats.py').read_text(encoding='utf-8')
    assert 'guard_write' in source


@pytest.mark.xfail(strict=True, reason='R-05: el orden de qids del bootstrap de exp17 depende de PYTHONHASHSEED')
def test_r05_exp17_cluster_bootstrap_is_independent_of_hash_seed():
    data = ROOT/'experiments'/'results'/'exp17_crosscloud_balanced'/'faithfulness_rows__hhem.json'
    if not data.is_file():
        pytest.fail('Artefacto versionado ausente: '+str(data))
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from scripts.compute_exp17_powered import load_counts, cluster_bootstrap\n'
            'r = cluster_bootstrap(load_counts("faithfulness_rows__hhem.json"), n_boot=500)\n'
            'print(r["boot95"], r["p_one_sided_diff_le0"])') % str(ROOT)
    outputs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                              env=dict(os.environ, PYTHONHASHSEED=seed)).stdout for seed in ('42', '7')}
    assert len(outputs) == 1, outputs


WRITERS = ['compute_exp17_powered.py', 'compute_exp18_powered.py', 'compute_exp18_diagnosis.py',
           'compute_exp18_selection_bound.py', 'compute_tierA_arm_stats.py', 'compute_exp15_ensemble_sweep.py',
           'compute_exp15_nli_sweep.py', 'compute_exp15_hhem_analysis.py', 'compute_exp16_guards.py',
           'compute_exp19b_stats.py', 'rescore_nli_v3.py']


@pytest.mark.xfail(strict=True, reason='R-06: la guarda de evidencia es opt-in en los escritores de exp15..exp19b')
def test_r06_every_signed_evidence_writer_uses_guard_write():
    unguarded = [n for n in WRITERS if 'guard_write' not in (ROOT/'scripts'/n).read_text(encoding='utf-8')]
    assert not unguarded, unguarded


@pytest.mark.xfail(strict=True, reason='R-09: la rama fix/interview-readiness está escrita a mano en varios módulos')
def test_r09_operator_branch_name_has_one_source_of_truth():
    holders = sorted(str(p.relative_to(ROOT)) for p in PY_SOURCES
                     if 'fix/interview-readiness' in p.read_text(encoding='utf-8', errors='replace'))
    assert len(holders) <= 1, holders


def test_r09_every_hardcoded_branch_literal_is_identical_today():
    """[VERIFICADO] Hoy todas las copias coinciden: el riesgo es de mantenimiento (S2), no un fallo."""
    literals = set()
    for path in PY_SOURCES:
        literals.update(re.findall(r"refs/heads/([\w./-]+?)['\"]", path.read_text(encoding='utf-8', errors='replace')))
    assert literals <= {'fix/interview-readiness'}, literals
