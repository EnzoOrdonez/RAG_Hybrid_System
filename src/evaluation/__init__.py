"""
Evaluation module - Phase 4: Evaluation, Benchmarks & Experiments.

Submodules:
  - retrieval_metrics: Recall@K, Precision@K, MRR, NDCG@K, MAP
  - generation_metrics: Exact Match, F1 Token, ROUGE-L, BERTScore
  - hallucination_metrics: NLI-based faithfulness, RAGAS (optional)
  - latency_metrics: Per-stage latency stats (p50, p95, p99)
  - test_queries: 200 test queries with ground truth
  - statistical_analysis: Shapiro-Wilk, t-test/Wilcoxon, Cohen's d, Bootstrap CI
  - benchmark_runner: Experiment execution with checkpoints/resume
  - results_exporter: LaTeX tables, matplotlib figures, CSV export
"""
