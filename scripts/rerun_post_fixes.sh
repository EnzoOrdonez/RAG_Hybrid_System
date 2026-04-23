#!/usr/bin/env bash
# ---------------------------------------------------------------
# rerun_post_fixes.sh — Phase 2 re-run entry point.
#
# Re-runs every experiment whose hallucination numbers were
# invalidated by the Phase 2 fixes (Flags 135, 138, 137, 140, 152,
# 153, 155). These five experiments feed every faithfulness number
# in the paper's abstract, §V, §VI:
#   exp5  LLM comparison
#   exp6  Ablation study
#   exp7  Cross-provider query expansion (the +16.8% claim)
#   exp8  End-to-end system comparison (Llama 3.1)
#   exp8b End-to-end system comparison (Mistral)
#
# exp3 and exp4 are NOT re-run here. exp3 tests retrieval fusion
# and exp4 tests reranker choice; neither depends on
# hallucination_detector. If exp3/exp4 were also missing
# retrieval_metrics in their results.json, that is a separate
# instrumentation re-run to be decided after Phase 3 (audit §10
# Flag 69, §14.8 Flag 105).
#
# Estimated wall time on an RTX 3060 Laptop 6GB: 24-48h. Run it
# overnight. DO NOT invoke from the Claude Code agent — launch
# manually from a shell you own.
#
# Once complete, notify Claude Code that the re-run is done and
# the JSON under experiments/results/exp{5,6,7,8,8b}/ is fresh.
# Phase 3 begins with the figure/table regeneration from those
# new JSONs.
# ---------------------------------------------------------------
set -euo pipefail

# Seed interpreter-level string-hash randomization so test_queries.py
# (audit §20.2 Flag 153) produces a bit-identical query set on every
# launch. Must be set BEFORE Python starts.
export PYTHONHASHSEED=42

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENTS=(exp5 exp6 exp7 exp8 exp8b)

echo "=============================================================="
echo "Phase 2 re-run: ${EXPERIMENTS[*]}"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
echo "Repo: $REPO_ROOT"
echo "=============================================================="

# ---------------------------------------------------------------
# Pre-flight sanity checks
# ---------------------------------------------------------------
echo ""
echo "Pre-flight: verifying Phase 2 fixes are present in HEAD..."

verify_marker() {
    local file="$1"
    local marker="$2"
    local flag="$3"
    if ! grep -q -- "$marker" "$file"; then
        echo "FAIL: $flag marker '$marker' not found in $file"
        echo "This script should only run on a checkout that contains Phase 2 fixes."
        exit 1
    fi
}

verify_marker src/generation/hallucination_detector.py "apply_softmax=True" "Flag 135"
verify_marker src/generation/hallucination_detector.py "Honovich" "Flag 138"
verify_marker src/evaluation/benchmark_runner.py "hall_n_effective" "Flag 137/140"
verify_marker src/utils/reproducibility.py "set_all_seeds" "Flag 152"
verify_marker src/generation/llm_manager.py '"seed": self.seed' "Flag 155"

echo "All Phase 2 fix markers present in HEAD."

# ---------------------------------------------------------------
# Cache + checkpoint cleanup
# ---------------------------------------------------------------
echo ""
echo "Clearing LLM response cache (data/llm_cache/*.json)..."
if [[ -d data/llm_cache ]]; then
    find data/llm_cache -maxdepth 1 -name '*.json' -print -delete
else
    mkdir -p data/llm_cache
fi

for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="experiments/results/$exp"
    if [[ -d "$exp_dir" ]]; then
        echo "Clearing checkpoints under $exp_dir/checkpoint_*.json..."
        find "$exp_dir" -maxdepth 1 -name 'checkpoint_*.json' -print -delete
    else
        echo "Note: $exp_dir does not exist yet — will be created by the benchmark."
    fi
done

# ---------------------------------------------------------------
# Re-run each affected experiment in order.
# --resume is OFF so checkpoints from the broken (pre-Phase 2) runs
# cannot poison the new numbers.
# ---------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------"
    echo "Running $exp ..."
    echo "--------------------------------------------------------------"
    python scripts/run_benchmark.py --experiment "$exp"
done

# ---------------------------------------------------------------
# Re-compute retrieval metrics for the two end-to-end experiments
# that actually have ground-truth retrieval (exp8 and exp8b) so the
# statistical_tests field in retrieval_metrics.json is regenerated
# with Phase 1's BH-FDR + paired Cohen's d.
# ---------------------------------------------------------------
for exp in exp8 exp8b; do
    echo ""
    echo "Recomputing retrieval metrics for $exp ..."
    python scripts/compute_retrieval_metrics.py --experiment "$exp"
done

echo ""
echo "=============================================================="
echo "Re-run complete. Next step: Phase 3 (regenerate figures and"
echo "tables from the new experiments/results/*.json)."
echo "=============================================================="
