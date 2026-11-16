"""Auditoría Fable: `BenchmarkRunner` (P1), sin modelos, sin índice y sin tocar la evidencia.

Tres bloques:
  1. Caracterización: fija lo que el runner hace hoy (ejecución de una consulta y agregación).
  2. H-10 (clase A): el módulo no debe anunciar un timeout por consulta que no aplica.
  3. Propuestas de clase B, como `xfail(strict=True)`: muestran el defecto y quedan a la espera
     de la decisión de Enzo. Si alguien aplica el parche, el test pasa y `strict` obliga a
     quitar la marca.

El pipeline, el detector NLI y el cargador de índices son dobles. Todo se escribe en tmp_path.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import benchmark_runner as br_module  # noqa: E402
from src.evaluation.benchmark_runner import BenchmarkRunner, QueryResult  # noqa: E402
from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402
from src.pipeline.rag_pipeline import LatencyBreakdown, RAGResponse  # noqa: E402

CHUNK = {"chunk_id": "c1", "text": "Amazon S3 stores objects inside buckets in a region.",
         "cloud_provider": "aws", "service_name": "S3"}


class FakePipeline:
    def __init__(self, response, delay_s=0.0):
        self.response, self.delay_s = response, delay_s

    def query(self, question):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.response


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Runner que escribe en tmp_path y no siembra RNG globales ni toca torch."""
    monkeypatch.setattr("src.utils.reproducibility.set_all_seeds", lambda seed=42: None)
    r = BenchmarkRunner(results_dir=str(tmp_path / "results"))
    r._hallucination_detector = HallucinationDetector(use_nli=False)  # sin modelo NLI (D1)
    return r


def _query(qid="q001", relevant=("c1", "c9")):
    return SimpleNamespace(query_id=qid, question="Where does S3 store objects?",
                           relevant_chunk_ids=list(relevant), answer="")


def _row(qid, method, faithfulness, decline=False, error=None, **claims):
    hall = {"faithfulness": faithfulness, "hallucination_rate": 1 - faithfulness,
            "method": method, "is_honest_decline": decline, **claims}
    return QueryResult(query_id=qid, config_name="cfg", question="q", answer="a",
                       hallucination_metrics=hall, error=error,
                       retrieval_metrics={"ndcg@5": 0.5}, latency={"total_ms": 100.0})


# --------------------------------------------------------------- caracterización
def test_single_query_scores_retrieval_and_flags_the_decline(runner):
    response = RAGResponse(answer="Amazon S3 stores objects inside buckets in a region.",
                           retrieved_chunks=[CHUNK], confidence="HONEST_DECLINE",
                           latency=LatencyBreakdown(total_ms=12.0), config_name="cfg")

    result = runner._run_single_query(_query(), FakePipeline(response), "cfg")

    assert result.error is None
    assert result.retrieved_ids == ["c1"]
    assert result.retrieval_metrics["recall@5"] == 0.5      # 1 de 2 relevantes
    assert result.retrieval_metrics["precision@5"] == 1.0   # denominador histórico min(k, n)
    assert result.retrieval_metrics["mrr"] == 1.0
    assert result.hallucination_metrics["method"] == "keyword_fallback"
    assert result.hallucination_metrics["is_honest_decline"] is True
    assert result.latency["total_ms"] == 12.0 and "benchmark_total_ms" in result.latency


def test_an_exception_inside_the_pipeline_becomes_an_error_row(runner):
    class Exploding:
        def query(self, question):
            raise RuntimeError("índice no cargado")

    result = runner._run_single_query(_query(), Exploding(), "cfg")

    assert result.error == "índice no cargado"
    assert result.answer == "" and result.hallucination_metrics == {}


def test_aggregation_excludes_none_and_error_methods_from_the_faithfulness_mean(runner, tmp_path):
    """Flags 137/140: `none` y `error` llevan una fidelidad sintética y no se promedian."""
    rows = [_row("q1", "nli", 0.8, total_claims=5, supported_claims=4),
            _row("q2", "nli", 0.4, decline=True, total_claims=5, supported_claims=2),
            _row("q3", "none", 1.0, total_claims=0, supported_claims=0),
            _row("q4", "error", 0.0, total_claims=0, supported_claims=0),
            _row("q5", "no_evidence", 0.0, total_claims=3, supported_claims=0),
            _row("q6", "nli", 0.9, error="boom")]            # fila con error: fuera de todo

    (tmp_path / "results" / "expX").mkdir(parents=True)     # lo crea _save_experiment_results
    runner._save_aggregated_metrics("expX", {"cfg": rows})
    agg = json.loads((tmp_path / "results" / "expX" / "aggregated_metrics.json")
                     .read_text(encoding="utf-8"))["cfg"]

    assert (agg["total_queries"], agg["errors"]) == (6, 1)
    assert (agg["hall_n_total"], agg["hall_n_effective"], agg["hall_n_excluded_none_error"]) == (5, 3, 2)
    assert agg["hall_method_counts"] == {"nli": 2, "none": 1, "error": 1, "no_evidence": 1}
    assert agg["hall_faithfulness_mean"] == pytest.approx((0.8 + 0.4 + 0.0) / 3)
    assert agg["hall_faithfulness_n"] == 3
    assert agg["hall_total_claims_sum"] == 13                 # los conteos no se filtran
    assert agg["hall_honest_decline_rate_mean"] == pytest.approx(1 / 5)
    assert agg["ret_ndcg@5_mean"] == 0.5


# ------------------------------------------------------------------ H-10, clase A
def test_query_timeout_is_accepted_but_never_enforced(tmp_path, monkeypatch):
    """`query_timeout` se guarda y nada lo aplica: una consulta lenta termina con normalidad."""
    monkeypatch.setattr("src.utils.reproducibility.set_all_seeds", lambda seed=42: None)
    r = BenchmarkRunner(results_dir=str(tmp_path / "results"), query_timeout=0)
    response = RAGResponse(answer="", retrieved_chunks=[], config_name="cfg")

    result = r._run_single_query(_query(), FakePipeline(response, delay_s=0.05), "cfg")

    assert r.query_timeout == 0 and result.error is None
    assert result.latency["benchmark_total_ms"] >= 40      # margen: sleep(0.05) puede volver antes


def test_the_module_does_not_advertise_a_timeout_it_does_not_apply():
    doc = br_module.__doc__ or ""
    assert "120-second timeout per query" not in doc
    assert "not enforced" in doc
    assert not hasattr(br_module, "TimeoutError"), \
        "una clase TimeoutError sin uso tapa el builtin dentro del módulo"


# ------------------------------------------------------- propuestas de clase B
def test_rerunning_a_completed_protected_experiment_does_not_rewrite_its_artifacts(tmp_path, monkeypatch):
    """H-01 (B1 aplicada el 2026-11-15): relanzar un experimento protegido no toca sus artefactos."""
    from src.utils import signed_evidence

    monkeypatch.setattr("src.utils.reproducibility.set_all_seeds", lambda seed=42: None)
    results_root = tmp_path / "experiments" / "results"
    monkeypatch.setattr(signed_evidence, "RESULTS", results_root)
    exp_dir = results_root / "exp8"
    exp_dir.mkdir(parents=True)
    signed = '{"signed": true}\n'
    (exp_dir / "results.json").write_text(signed, encoding="utf-8")
    (exp_dir / "aggregated_metrics.json").write_text(signed, encoding="utf-8")
    done = QueryResult(query_id="q001", config_name="cfg", question="q", answer="a").to_dict()
    (exp_dir / "checkpoint_cfg.json").write_text(json.dumps(
        {"experiment_id": "exp8", "config_name": "cfg", "completed_query_ids": ["q001"],
         "results": [done]}), encoding="utf-8")

    config = SimpleNamespace(experiment_id="exp8", name="n", description="d", hypothesis="h",
                             max_queries=10, metrics=["retrieval"], seed=42,
                             pipeline_configs=[SimpleNamespace(name="cfg")])
    r = BenchmarkRunner(results_dir=str(results_root))

    try:
        with pytest.raises(SystemExit):
            r.run_experiment(config, [_query("q001")])
    finally:
        untouched = ((exp_dir / "results.json").read_text(encoding="utf-8") == signed
                     and (exp_dir / "aggregated_metrics.json").read_text(encoding="utf-8") == signed)
    assert untouched


@pytest.mark.xfail(strict=True, reason="FABLE-B3: pendiente de decisión de Enzo")
def test_a_pipeline_error_is_recorded_as_an_error_not_scored_as_an_answer(runner):
    """H-03: el texto «Error generating response…» se puntúa hoy como si fuera la respuesta."""
    response = RAGResponse(answer="Error generating response: All 3 retries failed: boom",
                           retrieved_chunks=[CHUNK], error="All 3 retries failed: boom",
                           config_name="cfg")

    result = runner._run_single_query(_query(), FakePipeline(response), "cfg")

    assert result.error == "All 3 retries failed: boom"
    assert "faithfulness" not in result.hallucination_metrics


@pytest.mark.xfail(strict=True, reason="FABLE-B7: pendiente de decisión de Enzo")
def test_index_cache_distinguishes_embedding_models(runner, monkeypatch):
    """H-07: dos modelos de embeddings con el mismo chunking comparten hoy un solo índice."""
    import src.pipeline.rag_pipeline as rag_pipeline

    loaded = []

    def fake_loader(embedding_model, chunking_strategy, chunk_size):
        loaded.append(embedding_model)
        return object()

    monkeypatch.setattr(rag_pipeline, "load_hybrid_index", fake_loader)

    large = runner._get_hybrid_index("adaptive", 500, "BAAI/bge-large-en-v1.5")
    mini = runner._get_hybrid_index("adaptive", 500, "sentence-transformers/all-MiniLM-L6-v2")

    assert len(loaded) == 2 and large is not mini
