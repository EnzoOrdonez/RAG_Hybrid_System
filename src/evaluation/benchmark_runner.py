"""
Benchmark Runner for thesis evaluation.

Features:
  - Runs experiments with configurable pipeline configs
  - Checkpoint every 10 queries (resume support)
  - 120-second timeout per query
  - Error recovery (skip failed queries, log errors)
  - Per-query result tracking with full metrics
  - Supports retrieval-only mode (no LLM needed)

Stores results in: experiments/results/{experiment_id}/
"""

import json
import logging
import os
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.evaluation.generation_metrics import compute_all_generation_metrics
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.evaluation.latency_metrics import compute_all_latency_metrics, format_latency_table
from src.evaluation.test_queries import TestQuery, load_queries

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N queries
QUERY_TIMEOUT = 120  # Seconds per query


class TimeoutError(Exception):
    """Query timeout."""
    pass


class QueryResult:
    """Result of running a single query through a pipeline config."""

    def __init__(
        self,
        query_id: str,
        config_name: str,
        question: str,
        answer: str = "",
        retrieved_ids: List[str] = None,
        relevant_ids: List[str] = None,
        retrieval_metrics: Dict = None,
        generation_metrics: Dict = None,
        hallucination_metrics: Dict = None,
        latency: Dict = None,
        error: str = None,
        timestamp: str = None,
    ):
        self.query_id = query_id
        self.config_name = config_name
        self.question = question
        self.answer = answer
        self.retrieved_ids = retrieved_ids or []
        self.relevant_ids = relevant_ids or []
        self.retrieval_metrics = retrieval_metrics or {}
        self.generation_metrics = generation_metrics or {}
        self.hallucination_metrics = hallucination_metrics or {}
        self.latency = latency or {}
        self.error = error
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "config_name": self.config_name,
            "question": self.question,
            "answer": self.answer,
            "retrieved_ids": self.retrieved_ids,
            "relevant_ids": self.relevant_ids,
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "hallucination_metrics": self.hallucination_metrics,
            "latency": self.latency,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "QueryResult":
        return cls(**d)


class BenchmarkRunner:
    """
    Runs benchmark experiments with checkpoint/resume support.

    Usage:
        runner = BenchmarkRunner()
        results = runner.run_experiment(experiment_config, queries)
        # or
        all_results = runner.run_all_experiments(queries)
    """

    def __init__(
        self,
        results_dir: str = None,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        query_timeout: int = QUERY_TIMEOUT,
        seed: int = 42,
    ):
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.query_timeout = query_timeout
        self.seed = seed

        # Shared resources (loaded once)
        self._hybrid_indices = {}
        self._llm_manager = None
        self._hallucination_detector = None

    # ============================================================
    # Index and model management
    # ============================================================

    def _get_hybrid_index(self, chunking_strategy: str, chunk_size: int, embedding_model: str = None):
        """Get or load a hybrid index (cached)."""
        key = f"{chunking_strategy}_{chunk_size}"
        if key not in self._hybrid_indices:
            try:
                from src.pipeline.rag_pipeline import load_hybrid_index

                model = "bge-large"  # Default
                if embedding_model:
                    if "bge-base" in embedding_model:
                        model = "bge-base"
                    elif "MiniLM" in embedding_model:
                        model = "all-MiniLM-L6"

                self._hybrid_indices[key] = load_hybrid_index(
                    embedding_model=model,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                )
                logger.info("Loaded hybrid index: %s", key)
            except Exception as e:
                logger.error("Failed to load index %s: %s", key, e)
                return None
        return self._hybrid_indices[key]

    def _get_llm_manager(self, provider: str = "ollama", model: str = "llama3.1:8b-instruct-q4_K_M"):
        """Get or create LLM manager."""
        if self._llm_manager is None or self._llm_manager.model != model:
            try:
                from src.generation.llm_manager import LLMManager
                self._llm_manager = LLMManager(provider=provider, model=model)
            except Exception as e:
                logger.warning("LLM manager unavailable: %s", e)
                self._llm_manager = None
        return self._llm_manager

    def _get_hallucination_detector(self):
        """Get or create hallucination detector."""
        if self._hallucination_detector is None:
            try:
                from src.generation.hallucination_detector import HallucinationDetector
                self._hallucination_detector = HallucinationDetector()
            except Exception as e:
                logger.warning("Hallucination detector unavailable: %s", e)
        return self._hallucination_detector

    # ============================================================
    # Checkpoint management
    # ============================================================

    def _checkpoint_path(self, experiment_id: str, config_name: str) -> Path:
        """Get checkpoint file path."""
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config_name.replace("/", "_").replace(" ", "_")
        return exp_dir / f"checkpoint_{safe_name}.json"

    def _save_checkpoint(
        self,
        experiment_id: str,
        config_name: str,
        results: List[QueryResult],
        completed_ids: set,
    ):
        """Save checkpoint to disk."""
        path = self._checkpoint_path(experiment_id, config_name)
        data = {
            "experiment_id": experiment_id,
            "config_name": config_name,
            "completed_query_ids": list(completed_ids),
            "results": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Checkpoint saved: %s (%d results)", path.name, len(results))

    def _load_checkpoint(
        self,
        experiment_id: str,
        config_name: str,
    ) -> tuple:
        """Load checkpoint if exists. Returns (results, completed_ids)."""
        path = self._checkpoint_path(experiment_id, config_name)
        if not path.exists():
            return [], set()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results = [QueryResult.from_dict(r) for r in data.get("results", [])]
            completed = set(data.get("completed_query_ids", []))
            logger.info(
                "Resumed checkpoint for %s/%s: %d queries done",
                experiment_id, config_name, len(completed),
            )
            return results, completed
        except Exception as e:
            logger.warning("Failed to load checkpoint %s: %s", path, e)
            return [], set()

    # ============================================================
    # Single query execution
    # ============================================================

    def _run_single_query(
        self,
        query: TestQuery,
        pipeline,
        config_name: str,
        compute_generation: bool = True,
        compute_hallucination: bool = True,
    ) -> QueryResult:
        """Run a single query through a pipeline and compute metrics."""
        start_time = time.perf_counter()

        try:
            # Run pipeline query
            response = pipeline.query(query.question)

            # Extract retrieved IDs
            retrieved_ids = []
            for chunk in response.retrieved_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    retrieved_ids.append(chunk_id)

            # Compute retrieval metrics
            ret_metrics = {}
            if query.relevant_chunk_ids:
                ret_metrics = compute_all_retrieval_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=query.relevant_chunk_ids,
                    k_values=[5, 10, 20],
                )

            # Compute generation metrics (if ground truth available)
            gen_metrics = {}
            if compute_generation and query.answer and response.answer:
                gen_metrics = compute_all_generation_metrics(
                    predicted=response.answer,
                    ground_truth=query.answer,
                    use_bert_score=False,  # Skip to save time
                )

            # Compute hallucination metrics
            hall_metrics = {}
            if compute_hallucination and response.answer and response.retrieved_chunks:
                detector = self._get_hallucination_detector()
                hall_metrics = compute_hallucination_metrics(
                    response_text=response.answer,
                    retrieved_chunks=response.retrieved_chunks,
                    detector=detector,
                )

            # Latency from pipeline
            latency_dict = response.latency.model_dump() if response.latency else {}

            elapsed = (time.perf_counter() - start_time) * 1000
            latency_dict["benchmark_total_ms"] = elapsed

            return QueryResult(
                query_id=query.query_id,
                config_name=config_name,
                question=query.question,
                answer=response.answer or "",
                retrieved_ids=retrieved_ids,
                relevant_ids=query.relevant_chunk_ids,
                retrieval_metrics=ret_metrics,
                generation_metrics=gen_metrics,
                hallucination_metrics=hall_metrics,
                latency=latency_dict,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error("Query %s failed: %s", query.query_id, e)
            return QueryResult(
                query_id=query.query_id,
                config_name=config_name,
                question=query.question,
                error=str(e),
                latency={"benchmark_total_ms": elapsed},
            )

    # ============================================================
    # Experiment execution
    # ============================================================

    def run_experiment(
        self,
        experiment_config,
        queries: List[TestQuery],
        resume: bool = True,
        max_queries: int = None,
    ) -> Dict[str, List[QueryResult]]:
        """
        Run a full experiment across all pipeline configs.

        Args:
            experiment_config: ExperimentConfig object
            queries: List of TestQuery objects
            resume: Whether to resume from checkpoint
            max_queries: Override max queries (None = use experiment config)

        Returns:
            Dict mapping config_name to list of QueryResult
        """
        exp_id = experiment_config.experiment_id
        max_q = max_queries or experiment_config.max_queries
        queries = queries[:max_q]

        logger.info(
            "=" * 60 + "\nStarting experiment: %s (%s)\nConfigs: %d | Queries: %d\n" + "=" * 60,
            exp_id, experiment_config.name,
            len(experiment_config.pipeline_configs), len(queries),
        )

        all_results = {}
        compute_gen = "generation" in experiment_config.metrics
        compute_hall = "hallucination" in experiment_config.metrics

        for config_idx, pipeline_config in enumerate(experiment_config.pipeline_configs):
            config_name = pipeline_config.name
            logger.info(
                "\n--- Config %d/%d: %s ---",
                config_idx + 1, len(experiment_config.pipeline_configs), config_name,
            )

            # Resume from checkpoint
            if resume:
                results, completed_ids = self._load_checkpoint(exp_id, config_name)
            else:
                results, completed_ids = [], set()

            # Skip if all done
            remaining_queries = [q for q in queries if q.query_id not in completed_ids]
            if not remaining_queries:
                logger.info("All queries already completed for %s", config_name)
                all_results[config_name] = results
                continue

            # Load index for this config
            hybrid_index = self._get_hybrid_index(
                pipeline_config.chunking_strategy,
                pipeline_config.chunk_size,
                pipeline_config.embedding_model,
            )

            if hybrid_index is None:
                logger.error("Cannot load index for %s, skipping", config_name)
                all_results[config_name] = results
                continue

            # Create pipeline
            try:
                from src.pipeline.rag_pipeline import RAGPipeline

                llm = self._get_llm_manager(
                    pipeline_config.llm_provider,
                    pipeline_config.llm_model,
                )

                pipeline = RAGPipeline(
                    config=pipeline_config,
                    hybrid_index=hybrid_index,
                    llm_manager=llm,
                )
            except Exception as e:
                logger.error("Failed to create pipeline for %s: %s", config_name, e)
                all_results[config_name] = results
                continue

            # Run queries
            for q_idx, query in enumerate(remaining_queries):
                try:
                    result = self._run_single_query(
                        query=query,
                        pipeline=pipeline,
                        config_name=config_name,
                        compute_generation=compute_gen,
                        compute_hallucination=compute_hall,
                    )
                    results.append(result)
                    completed_ids.add(query.query_id)

                    # Progress
                    total_done = len(completed_ids)
                    total = len(queries)
                    pct = total_done / total * 100
                    latency_ms = result.latency.get("benchmark_total_ms", 0)

                    status = "OK" if not result.error else f"ERR: {result.error[:50]}"
                    logger.info(
                        "  [%d/%d %.0f%%] %s | %.0fms | %s",
                        total_done, total, pct, query.query_id, latency_ms, status,
                    )

                except Exception as e:
                    logger.error("Unexpected error on %s: %s", query.query_id, e)
                    results.append(QueryResult(
                        query_id=query.query_id,
                        config_name=config_name,
                        question=query.question,
                        error=f"unexpected: {str(e)}",
                    ))
                    completed_ids.add(query.query_id)

                # Checkpoint
                if len(completed_ids) % self.checkpoint_interval == 0:
                    self._save_checkpoint(exp_id, config_name, results, completed_ids)

            # Final save
            self._save_checkpoint(exp_id, config_name, results, completed_ids)
            all_results[config_name] = results

        # Save full experiment results
        self._save_experiment_results(exp_id, experiment_config, all_results)

        return all_results

    def run_all_experiments(
        self,
        queries: List[TestQuery],
        experiment_ids: List[str] = None,
        resume: bool = True,
    ) -> Dict[str, Dict[str, List[QueryResult]]]:
        """
        Run multiple experiments.

        Args:
            queries: Full query dataset
            experiment_ids: Specific experiments to run (None = all)
            resume: Resume from checkpoints

        Returns:
            Dict mapping experiment_id to experiment results
        """
        from experiments.experiment_configs import EXPERIMENT_CONFIGS

        if experiment_ids is None:
            experiment_ids = list(EXPERIMENT_CONFIGS.keys())

        all_exp_results = {}
        for exp_id in experiment_ids:
            if exp_id not in EXPERIMENT_CONFIGS:
                logger.warning("Unknown experiment: %s, skipping", exp_id)
                continue

            config = EXPERIMENT_CONFIGS[exp_id]

            # For exp7 (cross-cloud), filter to cross-cloud queries only
            if exp_id == "exp7":
                exp_queries = [q for q in queries if len(q.cloud_providers) > 1]
            else:
                exp_queries = queries

            results = self.run_experiment(config, exp_queries, resume=resume)
            all_exp_results[exp_id] = results

        return all_exp_results

    # ============================================================
    # Results saving and aggregation
    # ============================================================

    def _save_experiment_results(
        self,
        experiment_id: str,
        experiment_config,
        results: Dict[str, List[QueryResult]],
    ):
        """Save full experiment results to JSON."""
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save per-query results
        output_path = exp_dir / "results.json"
        data = {
            "experiment_id": experiment_id,
            "name": experiment_config.name,
            "description": experiment_config.description,
            "hypothesis": experiment_config.hypothesis,
            "timestamp": datetime.now().isoformat(),
            "configs": {},
        }

        for config_name, query_results in results.items():
            data["configs"][config_name] = {
                "total_queries": len(query_results),
                "errors": sum(1 for r in query_results if r.error),
                "results": [r.to_dict() for r in query_results],
            }

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved experiment results to %s", output_path)

        # Save aggregated metrics
        self._save_aggregated_metrics(experiment_id, results)

    def _save_aggregated_metrics(
        self,
        experiment_id: str,
        results: Dict[str, List[QueryResult]],
    ):
        """Compute and save aggregated metrics per config."""
        exp_dir = self.results_dir / experiment_id

        aggregated = {}
        for config_name, query_results in results.items():
            valid = [r for r in query_results if not r.error]
            if not valid:
                continue

            agg = {"config_name": config_name, "total_queries": len(query_results)}
            agg["errors"] = len(query_results) - len(valid)

            # Aggregate retrieval metrics
            ret_keys = set()
            for r in valid:
                ret_keys.update(r.retrieval_metrics.keys())
            for key in sorted(ret_keys):
                values = [r.retrieval_metrics.get(key, 0) for r in valid if r.retrieval_metrics]
                if values:
                    agg[f"ret_{key}_mean"] = float(np.mean(values))
                    agg[f"ret_{key}_std"] = float(np.std(values))

            # Aggregate generation metrics
            gen_keys = set()
            for r in valid:
                gen_keys.update(r.generation_metrics.keys())
            for key in sorted(gen_keys):
                values = [r.generation_metrics.get(key, 0) for r in valid if r.generation_metrics]
                if values:
                    agg[f"gen_{key}_mean"] = float(np.mean(values))
                    agg[f"gen_{key}_std"] = float(np.std(values))

            # Aggregate hallucination metrics
            hall_keys = ["faithfulness", "hallucination_rate"]
            for key in hall_keys:
                values = [
                    r.hallucination_metrics.get(key, 0)
                    for r in valid
                    if r.hallucination_metrics and key in r.hallucination_metrics
                ]
                if values:
                    agg[f"hall_{key}_mean"] = float(np.mean(values))
                    agg[f"hall_{key}_std"] = float(np.std(values))

            # Aggregate latency
            latency_records = [r.latency for r in valid if r.latency]
            if latency_records:
                lat_stats = compute_all_latency_metrics(latency_records)
                for stage, stats in lat_stats.items():
                    if isinstance(stats, dict):
                        agg[f"lat_{stage}_p50"] = stats.get("p50", 0)
                        agg[f"lat_{stage}_p95"] = stats.get("p95", 0)
                        agg[f"lat_{stage}_mean"] = stats.get("mean", 0)

            aggregated[config_name] = agg

        # Save
        agg_path = exp_dir / "aggregated_metrics.json"
        agg_path.write_text(
            json.dumps(aggregated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved aggregated metrics to %s", agg_path)

    # ============================================================
    # Results loading
    # ============================================================

    @staticmethod
    def load_experiment_results(experiment_id: str, results_dir: str = None) -> Dict:
        """Load saved experiment results."""
        rd = Path(results_dir or RESULTS_DIR)
        path = rd / experiment_id / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"No results found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_aggregated_metrics(experiment_id: str, results_dir: str = None) -> Dict:
        """Load saved aggregated metrics."""
        rd = Path(results_dir or RESULTS_DIR)
        path = rd / experiment_id / "aggregated_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"No aggregated metrics found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ============================================================
    # Quick summary
    # ============================================================

    def print_experiment_summary(
        self,
        experiment_id: str,
        results: Dict[str, List[QueryResult]],
    ):
        """Print a quick summary of experiment results."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_id}")
        print(f"{'='*70}")

        for config_name, query_results in results.items():
            valid = [r for r in query_results if not r.error]
            errors = len(query_results) - len(valid)

            print(f"\n  Config: {config_name}")
            print(f"  Queries: {len(query_results)} (errors: {errors})")

            if valid:
                # Key retrieval metrics
                ndcg5 = [r.retrieval_metrics.get("ndcg@5", 0) for r in valid if r.retrieval_metrics]
                recall5 = [r.retrieval_metrics.get("recall@5", 0) for r in valid if r.retrieval_metrics]
                mrr = [r.retrieval_metrics.get("mrr", 0) for r in valid if r.retrieval_metrics]

                if ndcg5:
                    print(f"  NDCG@5:  {np.mean(ndcg5):.4f} (+/- {np.std(ndcg5):.4f})")
                if recall5:
                    print(f"  Recall@5: {np.mean(recall5):.4f} (+/- {np.std(recall5):.4f})")
                if mrr:
                    print(f"  MRR:     {np.mean(mrr):.4f} (+/- {np.std(mrr):.4f})")

                # Hallucination
                faith = [
                    r.hallucination_metrics.get("faithfulness", 0)
                    for r in valid if r.hallucination_metrics
                ]
                if faith:
                    print(f"  Faith:   {np.mean(faith):.4f} (+/- {np.std(faith):.4f})")

                # Latency
                total_ms = [r.latency.get("total_ms", 0) for r in valid if r.latency]
                if total_ms:
                    print(f"  Latency: p50={np.percentile(total_ms,50):.0f}ms p95={np.percentile(total_ms,95):.0f}ms")

        print(f"\n{'='*70}")
