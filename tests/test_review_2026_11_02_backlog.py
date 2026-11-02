"""Auditoría 2026-11-02, BACKLOG: test propuesto para el arreglo de R-05 (patches/005-*.patch, NO aplicado).

Hoy falla (xfail estricto): cluster_bootstrap indexa qids por posición en el orden de inserción del
dict, que load_counts construye iterando un set (dependiente de PYTHONHASHSEED). Con el parche
(qids ordenados) pasa a XPASS y hay que quitar el marcador. Datos sintéticos: no lee ni escribe evidencia.
"""
import pytest

from scripts.compute_exp17_powered import cluster_bootstrap

COUNTS = {'q%02d' % i: {'baseline': (i % 4, 4 + i % 3), 'balanced': ((i * 7) % 5, 5)} for i in range(25)}


@pytest.mark.xfail(strict=True, reason='R-05: el resultado depende del orden de inserción de las queries')
def test_cluster_bootstrap_is_invariant_to_query_insertion_order():
    forward = dict(COUNTS)
    backward = dict(reversed(list(COUNTS.items())))
    assert cluster_bootstrap(forward, n_boot=500) == cluster_bootstrap(backward, n_boot=500)


def test_cluster_bootstrap_point_estimate_does_not_depend_on_order():
    """Control: la diferencia observada (sin remuestreo) ya es invariante; solo el IC y p cambian."""
    a = cluster_bootstrap(dict(COUNTS), n_boot=50)
    b = cluster_bootstrap(dict(reversed(list(COUNTS.items()))), n_boot=50)
    assert (a['micro_baseline'], a['micro_balanced'], a['micro_diff']) == (b['micro_baseline'], b['micro_balanced'], b['micro_diff'])
