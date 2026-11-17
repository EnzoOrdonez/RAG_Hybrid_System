"""Auditoría Fable: detección de proveedores y balanceo de cobertura (camino de SURVEY_DEPLOY).

  - H-11 (clase A): fija el orden real que produce el relleno de `balance()`.
  - H-04 / FABLE-B4 y H-05 / FABLE-B5 (clase B): `xfail(strict=True)` a la espera de Enzo.
  - Caracterización del clasificador de consultas, que decide la plantilla del prompt.

Sin índice ni modelos: el mapa de servicios es una copia sintética de los nombres reales
(`data/corpus_stats.json`), suficiente para reproducir el defecto.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.coverage_balancer import balance, resolve_wanted_providers  # noqa: E402
from src.retrieval.query_processor import QueryProcessor  # noqa: E402

PRESENT = {"aws", "azure", "gcp"}
# Nombres de servicio tal como los deriva build_service_index() del índice construido.
SERVICE_INDEX = {
    "ec2": "aws", "eks": "aws", "iam": "aws", "lambda": "aws", "s3": "aws", "vpc": "aws",
    "aks": "azure", "azure functions": "azure", "blob storage": "azure",
    "virtual machines": "azure", "virtual network": "azure",
    "cloud functions": "gcp", "compute engine": "gcp", "gke": "gcp",
}


@pytest.fixture(scope="module")
def processor():
    return QueryProcessor()


# ------------------------------------------------------------------ H-11, clase A
def test_fill_pass_appends_leftovers_after_the_quota_picks():
    """El relleno va DESPUÉS de las cuotas: a4 rankea mejor que b1 y queda detrás.

    Es el comportamiento con el que se midió exp17; el test lo fija para que el docstring
    de `balance()` no prometa otra cosa.
    """
    pool = ["a1", "a2", "a3", "a4", "b1"]

    def provider_of(cid):
        return "aws" if cid.startswith("a") else "azure"

    assert balance(pool, provider_of, ["aws", "azure"], k=5) == ["a1", "a2", "a3", "b1", "a4"]


def test_quota_picks_keep_reranked_order_across_providers():
    pool = ["g1", "a1", "z1", "a2", "g2", "z2", "a3"]
    prov = {"a": "aws", "z": "azure", "g": "gcp"}
    out = balance(pool, lambda cid: prov[cid[0]], ["aws", "azure", "gcp"], k=5)
    assert out == ["g1", "a1", "z1", "a2", "g2"]          # cuota 2 por proveedor, sin relleno
    assert out == sorted(out, key=pool.index)


# ------------------------------------------------------- caracterización del ruteo
@pytest.mark.parametrize("question,providers,query_type,provider_filter", [
    ("What is Amazon S3?", ["aws"], "single_provider", ["aws"]),
    ("How do I configure AWS Lambda?", ["aws"], "procedural", None),
    ("Compare AWS and Azure object storage", ["aws", "azure"], "cross_cloud", None),
    ("What is the difference between a container and a virtual machine?", [], "cross_cloud", None),
    ("How does load balancing work?", [], "conceptual", None),
])
def test_query_classification_is_stable(processor, question, providers, query_type, provider_filter):
    processed = processor.process(question)
    assert processed.detected_providers == providers
    assert processed.query_type == query_type
    assert processed.provider_filter == provider_filter
    assert processed.semantic_query == question            # la pierna densa nunca se expande


# ------------------------------------------------------- propuestas de clase B
@pytest.mark.xfail(strict=True, reason="FABLE-B4: pendiente de decisión de Enzo")
@pytest.mark.parametrize("question,detected,expected", [
    ("Compare virtual machines on AWS and GCP", ["aws", "gcp"], ["aws", "gcp"]),
    ("How does a GCP VPC differ from an Azure virtual network?", ["azure", "gcp"], ["azure", "gcp"]),
    ("Compare IAM in Azure and GCP", ["azure", "gcp"], ["azure", "gcp"]),
])
def test_generic_service_names_do_not_add_an_unrequested_provider(question, detected, expected):
    """H-04: «virtual machines», «vpc» e «iam» son nombres de servicio de UN proveedor en el
    índice, pero términos comunes a los tres. Hoy añaden ese proveedor y le dan cuota."""
    assert resolve_wanted_providers(question, detected, SERVICE_INDEX, PRESENT) == expected


@pytest.mark.xfail(strict=True, reason="FABLE-B5: pendiente de decisión de Enzo")
@pytest.mark.parametrize("question", [
    "What data residency laws apply to object storage?",
    "What are common flaws in serverless function design?",
])
def test_provider_detection_does_not_match_inside_other_words(processor, question):
    """H-05: «laws» y «flaws» contienen «aws»; hoy activan el filtro single_provider=aws."""
    processed = processor.process(question)
    assert processed.detected_providers == []
    assert processed.provider_filter is None
