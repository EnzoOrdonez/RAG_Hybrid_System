"""
Smoke test for Flags 25, 26, and 135 from audit_findings.md.

Resolves three questions about the NLI pipeline in
hallucination_detector.py:

  FLAG 25: Does sentence-transformers CrossEncoder.predict() return
           softmaxed probabilities or raw logits for nli-deberta-v3-small?
           If logits, the 0.7 threshold is not a probability threshold.

  FLAG 26: Is the label order [contradiction, entailment, neutral] as
           assumed at line 282 of hallucination_detector.py? If the real
           order differs (e.g. [entailment, neutral, contradiction] in
           BART-MNLI style), the faithfulness metric is INVERTED.

  FLAG 135: Post-fix validation — with apply_softmax=True, the obvious
           entailment pair must give entailment > 0.7 (probability) and
           the obvious contradiction pair must give contradiction > 0.7.

Run with:
    python scripts/audit/smoke_test_nli.py

Requires network access to huggingface.co for the first run (downloads
~200MB nli-deberta-v3-small). Proxied sandboxes may fail to fetch.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Pairs: (premise, hypothesis). Expected labels by human judgment:
#   1 => ENTAILMENT (premise clearly implies hypothesis)
#   2 => CONTRADICTION (premise contradicts hypothesis)
#   3 => NEUTRAL (premise neither implies nor contradicts hypothesis)
PAIRS = [
    ("Paris is the capital of France.",
     "The capital of France is Paris."),          # ENTAILMENT
    ("Paris is the capital of France.",
     "Paris is the capital of Germany."),         # CONTRADICTION
    ("Paris is the capital of France.",
     "Berlin has a large population."),           # NEUTRAL
]

EXPECTED = ["entailment", "contradiction", "neutral"]


def main() -> None:
    print(f"Loading {MODEL_NAME} ...")
    model = CrossEncoder(MODEL_NAME, max_length=512)

    # Inspect the label order from the underlying HF config.
    hf_config = getattr(model, "config", None) or getattr(
        model.model, "config", None
    )
    if hf_config is not None and hasattr(hf_config, "id2label"):
        id2label = hf_config.id2label
        print("\n=== MODEL CONFIG id2label ===")
        for i in sorted(id2label.keys()):
            print(f"  index {i}: {id2label[i]}")
        label_order = [id2label[i].lower() for i in sorted(id2label.keys())]
    else:
        print("\n[WARN] Could not inspect id2label from config.")
        label_order = None

    print("\n=== RAW MODEL OUTPUT (no softmax applied) ===")
    scores = model.predict(PAIRS)
    for (premise, hypothesis), score, expected in zip(PAIRS, scores, EXPECTED):
        print(f"\npair expected={expected}")
        print(f"  premise   : {premise}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  scores    : {list(score)}")
        total = sum(float(s) for s in score)
        print(f"  sum       : {total:.4f}  "
              f"(≈1.0 => softmax; arbitrary => logits)")
        if label_order:
            best_idx = int(max(range(len(score)), key=lambda i: score[i]))
            print(f"  argmax    : index {best_idx} "
                  f"=> label '{label_order[best_idx]}'")

    print("\n=== HOW TO READ THIS OUTPUT ===")
    print("Flag 25: If 'sum' is consistently ≈ 1.0 across pairs, model "
          "returns probabilities (softmax applied). "
          "Threshold 0.7 is then a probability. "
          "If sums are arbitrary or > 1, output is logits and the 0.7 "
          "threshold in hallucination_detector.py is mis-named.")
    print("\nFlag 26: For each pair, check that argmax matches the expected "
          "label. If not, the label-order assumption "
          "[contradiction, entailment, neutral] at line 282 of "
          "hallucination_detector.py is WRONG, and faithfulness is "
          "inverted.")
    print("\nThe code at line 284-285 reads:")
    print("    contradiction_score = float(score_set[0])")
    print("    entailment_score    = float(score_set[1])")
    print("Verify those indices match the 'MODEL CONFIG id2label' block "
          "above. If not, patch the indices before trusting any "
          "hallucination number in the paper.")

    # -----------------------------------------------------------------
    # FLAG 135 post-fix assertions: with apply_softmax=True, the threshold
    # 0.7 is a probability threshold and obvious cases must cross it.
    # -----------------------------------------------------------------
    print("\n=== FLAG 135 POST-FIX ASSERTIONS (apply_softmax=True) ===")
    POST_FIX_PAIRS = [
        ("The capital of France is Paris.",
         "Paris is the capital of France."),          # expect entailment > 0.7
        ("The capital of France is Paris.",
         "Paris is in Japan."),                        # expect contradiction > 0.7
    ]
    post_scores = model.predict(POST_FIX_PAIRS, apply_softmax=True)

    # Resolve indices from the config (do not hard-code).
    if not label_order:
        print("  SKIP: cannot resolve id2label; run the sections above first.")
        return
    try:
        contradiction_idx = label_order.index("contradiction")
        entailment_idx = label_order.index("entailment")
    except ValueError:
        print(f"  SKIP: unexpected label_order {label_order}")
        return

    failures = []

    pair_a, pair_b = POST_FIX_PAIRS
    scores_a, scores_b = post_scores
    ent_a = float(scores_a[entailment_idx])
    contr_b = float(scores_b[contradiction_idx])
    sum_a = float(sum(scores_a))
    sum_b = float(sum(scores_b))

    print(f"  pair A (entailment): entailment_prob={ent_a:.4f} "
          f"(sum={sum_a:.4f})")
    if ent_a <= 0.7:
        failures.append(
            f"FAIL: pair A entailment_prob={ent_a:.4f} <= 0.7. "
            f"Expected > 0.7 after softmax."
        )
    else:
        print(f"  PASS: pair A entailment_prob > 0.7 (post-softmax)")

    print(f"  pair B (contradiction): contradiction_prob={contr_b:.4f} "
          f"(sum={sum_b:.4f})")
    if contr_b <= 0.7:
        failures.append(
            f"FAIL: pair B contradiction_prob={contr_b:.4f} <= 0.7. "
            f"Expected > 0.7 after softmax."
        )
    else:
        print(f"  PASS: pair B contradiction_prob > 0.7 (post-softmax)")

    # Also verify the softmax sums to ~1.
    for name, s in (("A", sum_a), ("B", sum_b)):
        if abs(s - 1.0) > 1e-4:
            failures.append(
                f"FAIL: pair {name} softmax sum = {s:.6f} (expected ≈ 1.0)"
            )
        else:
            print(f"  PASS: pair {name} softmax sum ≈ 1.0 ({s:.6f})")

    if failures:
        print("\n".join(failures))
        raise AssertionError(
            f"Flag 135 post-fix smoke failed: {len(failures)} assertion(s). "
            "Do NOT trust faithfulness numbers in the next re-run."
        )
    print("\nFlag 135 post-fix smoke: ALL PASS.")


if __name__ == "__main__":
    main()
