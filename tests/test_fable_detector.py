"""Auditoría Fable: verificador de fidelidad (P1) y utilidades de reproducibilidad (P4).

Caracteriza lo que `HallucinationDetector.check` calcula hoy: estados por afirmación,
denominador, casos vacuos (N9), camino sin evidencia (N3) y exclusión por `method`. El modelo
NLI es un doble que devuelve `[contradicción, implicación, neutral]` por par; nunca se carga
`nli-deberta` (D1).
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.hallucination_metrics import compute_hallucination_metrics  # noqa: E402
from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402
from src.utils import reproducibility  # noqa: E402

C1 = {"chunk_id": "c1", "text": "Amazon S3 stores objects inside buckets."}
C2 = {"chunk_id": "c2", "text": "Lambda runs code without servers."}
NEUTRAL = (0.05, 0.05, 0.90)


class FakeNLI:
    """`table` = {fragmento de la afirmación: [fila por chunk]}; lo demás es neutral."""

    def __init__(self, table=None, chunks=(C1, C2), error=None):
        self.table, self.error, self.calls = table or {}, error, []
        self.position = {c["text"]: i for i, c in enumerate(chunks)}

    def predict(self, pairs, batch_size=32, show_progress_bar=False, apply_softmax=False):
        self.calls.append({"pairs": len(pairs), "apply_softmax": apply_softmax})
        if self.error:
            raise self.error
        rows = []
        for chunk_text, claim in pairs:
            hit = next((v for k, v in self.table.items() if k in claim), None)
            rows.append(list(hit[self.position[chunk_text]]) if hit else list(NEUTRAL))
        return rows


def _detector(fake):
    detector = HallucinationDetector()
    detector._nli_model, detector._nli_available = fake, True     # la propiedad ya no carga nada
    return detector


# ------------------------------------------------- estados, denominador y rúbrica
def test_each_claim_gets_a_status_and_artifacts_leave_the_denominator():
    fake = FakeNLI({"Amazon S3 stores": [(0.01, 0.95, 0.04), (0.02, 0.10, 0.88)],
                    "Lambda needs": [(0.90, 0.02, 0.08), (0.92, 0.03, 0.05)]})
    response = ("Amazon S3 stores objects inside buckets. "
                "Lambda needs dedicated servers to run code. "
                "EC2 instances are billed per second of use. "
                "### Networking overview for the region.")

    report = _detector(fake).check(response, [C1, C2])

    assert [d.status for d in report.claim_details] == [
        "supported", "contradicted", "unsupported", "not_a_claim"]
    assert (report.total_claims, report.supported_claims, report.contradicted_claims,
            report.unsupported_claims, report.not_a_claim_claims) == (4, 1, 1, 1, 1)
    assert report.faithfulness_score == 0.3333          # 1 de 3: el artefacto no divide
    assert report.hallucination_rate == 0.6667 and report.suggested_rubric == 2
    assert report.method == "nli"
    assert report.claim_details[0].evidence_chunk_id == "c1"
    assert len(fake.calls) == 3, "el artefacto de formato nunca llega al verificador"
    assert all(call == {"pairs": 2, "apply_softmax": True} for call in fake.calls)


def test_a_single_chunk_can_never_mark_a_claim_contradicted_under_vb_agree():
    """H2 (N8): `contradicted` exige que al menos 2 chunks superen 0,7."""
    fake = FakeNLI({"Lambda needs": [(0.95, 0.01, 0.04)]}, chunks=(C2,))
    report = _detector(fake).check("Lambda needs dedicated servers to run code.", [C2])
    assert report.claim_details[0].status == "unsupported"
    assert report.claim_details[0].nli_score == 0.95 and report.contradicted_claims == 0


def test_a_response_made_only_of_format_artifacts_scores_a_vacuous_one():
    """N9: fila vacua. El runtime le pone 1,0; el agregador v4 la excluye fuera de línea."""
    fake = FakeNLI()
    report = _detector(fake).check(
        "### Overview of the storage services.\n| Service | Provider | Tier |", [C1, C2])
    assert (report.total_claims, report.not_a_claim_claims) == (2, 2)
    assert (report.faithfulness_score, report.hallucination_rate, report.suggested_rubric) == (1.0, 0.0, 5)
    assert report.method == "nli" and fake.calls == []


def test_responses_without_any_claim_are_tagged_method_none():
    """Flags 137/140: `none` lleva una fidelidad sintética de 1,0 que el agregador excluye."""
    detector = _detector(FakeNLI())
    for response in ("Here is a summary. See also the docs.", "   ", ""):
        report = detector.check(response, [C1, C2])
        assert (report.method, report.total_claims, report.faithfulness_score,
                report.suggested_rubric) == ("none", 0, 1.0, 5), repr(response)


def test_without_chunks_every_claim_is_unsupported_no_evidence():
    """N3: sin recuperación la fidelidad es 0 por construcción, salvo que no haya afirmaciones."""
    detector = _detector(FakeNLI())

    report = detector.check("Amazon S3 stores objects inside buckets.", [])
    assert (report.method, report.total_claims, report.unsupported_claims) == ("no_evidence", 1, 1)
    assert (report.faithfulness_score, report.suggested_rubric) == (0.0, 1)
    assert report.claim_details[0].status == "unsupported_no_evidence"

    declined = detector.check("I cannot answer that question today.", [])
    assert (declined.method, declined.total_claims, declined.faithfulness_score) == ("no_evidence", 0, 1.0)


def test_chunks_without_text_do_not_take_the_no_evidence_path():
    """Borde: hay chunks pero ninguno trae texto. Queda `unsupported` con method `nli`."""
    fake = FakeNLI()
    report = _detector(fake).check("Amazon S3 stores objects inside buckets.",
                                   [{"chunk_id": "c9", "text": ""}])
    assert (report.method, report.claim_details[0].status, report.faithfulness_score) == (
        "nli", "unsupported", 0.0)
    assert fake.calls == []


def test_an_nli_failure_falls_back_to_keywords_for_that_claim_and_is_flagged_mixed():
    fake = FakeNLI(error=RuntimeError("CUDA out of memory"))
    report = _detector(fake).check("Amazon S3 stores objects inside buckets.", [C1, C2])
    detail = report.claim_details[0]
    assert report.method == "mixed"
    assert (detail.status, detail.verification_method, detail.verification_error) == (
        "supported", "keyword_fallback", "RuntimeError")


def test_keyword_mode_supports_a_claim_at_half_overlap_and_never_contradicts():
    detector = HallucinationDetector(use_nli=False)
    report = detector.check("Amazon S3 stores objects inside buckets. "
                            "Kubernetes pods always share one network namespace.", [C1, C2])
    assert report.method == "keyword_fallback"
    assert [d.status for d in report.claim_details] == ["supported", "unsupported"]
    assert report.contradicted_claims == 0 and report.faithfulness_score == 0.5


# --------------------------------------------------------- extracción de claims
def test_bullet_lists_become_claims_with_their_header_attached():
    text = ("Here is the mapping across AWS, Azure, and GCP:\n"
            "* Networking:\n"
            "  * AWS uses VPC for isolated networks\n"
            "  * Azure uses Virtual Network for isolation\n"
            "* **Storage** is covered by S3 and Blob Storage\n")
    assert HallucinationDetector(use_nli=False)._extract_claims(text) == [
        "Networking: AWS uses VPC for isolated networks.",
        "Networking: Azure uses Virtual Network for isolation.",
        "Storage is covered by S3 and Blob Storage."]


def test_code_and_citations_are_removed_before_splitting_into_claims():
    text = ("Run `aws s3 ls` to list every bucket you own [Source: aws/S3/cli].\n"
            "```\naws s3 mb s3://bucket\n```\n"
            "Is that all? Too short.")
    assert HallucinationDetector(use_nli=False)._extract_claims(text) == [
        "Run [CODE] to list every bucket you own ."]


# --------------------------------------------------------- hallucination_metrics
def test_metrics_wrapper_maps_the_report_and_routes_missing_chunks_to_no_evidence():
    detector = HallucinationDetector(use_nli=False)
    scored = compute_hallucination_metrics("Amazon S3 stores objects inside buckets.", [C1], detector)
    assert scored["method"] == "keyword_fallback" and scored["faithfulness"] == 1.0
    assert {"contradicted_claims", "unsupported_claims", "processing_time_ms"} <= set(scored)
    assert compute_hallucination_metrics(
        "Amazon S3 stores objects inside buckets.", None, detector)["method"] == "no_evidence"


def test_empty_and_whitespace_answers_disagree_but_both_are_method_none():
    """Observación de la auditoría: "" da 0,0 y "   " da 1,0. Ambas son `none` y se excluyen."""
    detector = HallucinationDetector(use_nli=False)
    assert compute_hallucination_metrics("", [C1], detector) == {
        "faithfulness": 0.0, "hallucination_rate": 1.0, "suggested_rubric": 1,
        "total_claims": 0, "supported_claims": 0, "method": "none"}
    blank = compute_hallucination_metrics("   ", [C1], detector)
    assert (blank["faithfulness"], blank["method"]) == (1.0, "none")


def test_a_detector_crash_is_reported_as_method_error_not_raised():
    class Broken:
        def check(self, response, chunks):
            raise RuntimeError("modelo no disponible")

    out = compute_hallucination_metrics("Amazon S3 stores objects.", [C1], Broken())
    assert (out["method"], out["faithfulness"], out["error"]) == ("error", 0.0, "modelo no disponible")


# ------------------------------------------------------------ reproducibility (P4)
def test_set_all_seeds_makes_random_and_numpy_repeatable():
    """En un proceso aparte y sin torch, para no sembrar los RNG globales de la suite."""
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import os, random\n"
        "import numpy as np\n"
        "from src.utils.reproducibility import set_all_seeds\n"
        "def draw(seed):\n"
        "    set_all_seeds(seed)\n"
        "    return random.random(), float(np.random.rand()), os.environ['PYTHONHASHSEED']\n"
        "a, b, c = draw(7), draw(7), draw(8)\n"
        "print(a == b, a != c, a[2], c[2])\n"
        "try:\n"
        "    set_all_seeds(-1)\n"
        "except ValueError:\n"
        "    print('negativo-rechazado')\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, env=dict(os.environ, PYTHONUTF8="1"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "True", "7", "8", "negativo-rechazado"]


def test_hashseed_guard_only_re_executes_when_the_variable_is_missing(monkeypatch):
    execs = []
    monkeypatch.setattr(reproducibility.os, "execv", lambda exe, argv: execs.append((exe, argv)))
    # setenv antes de delenv: delenv sobre una variable ausente no registra nada que deshacer, y
    # la función bajo prueba escribe el centinela directamente en os.environ.
    monkeypatch.setenv("_CLOUDRAG_HASHSEED_REEXEC", "0")
    monkeypatch.delenv("_CLOUDRAG_HASHSEED_REEXEC")

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    reproducibility.ensure_hashseed_at_startup(42)
    assert execs == []                                       # ya fijada: no hace nada

    monkeypatch.setenv("PYTHONHASHSEED", "7")
    with pytest.warns(RuntimeWarning, match="already set"):
        reproducibility.ensure_hashseed_at_startup(42)
    assert execs == [] and os.environ["PYTHONHASHSEED"] == "7"   # no pisa otra semilla

    monkeypatch.delenv("PYTHONHASHSEED")
    reproducibility.ensure_hashseed_at_startup(42)
    assert execs == [(sys.executable, [sys.executable] + sys.argv)]
    assert os.environ["PYTHONHASHSEED"] == "42" and os.environ["_CLOUDRAG_HASHSEED_REEXEC"] == "1"

    monkeypatch.delenv("PYTHONHASHSEED")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reproducibility.ensure_hashseed_at_startup(42)       # centinela puesto: no hay bucle
    assert len(execs) == 1
