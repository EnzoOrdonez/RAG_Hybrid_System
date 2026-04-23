"""
Smoke test for Flags 25 and 26 from audit_findings.md.

Resolves two questions about the NLI pipeline in hallucination_detector.py:

  FLAG 25: Does sentence-transformers CrossEncoder.predict() return
           softmaxed probabilities or raw logits for nli-deberta-v3-small?
           If logits, the 0.7 threshold is not a probability threshold.

  FLAG 26: Is the label order [contradiction, entailment, neutral] as
           assumed at line 282 of hallucination_detector.py? If the real
           order differs (e.g. [entailment, neutral, contradiction] in
           BART-MNLI style), the faithfulness metric is INVERTED.

Run with:
    python scripts/audit/smoke_test_nli.py

Expected output: prints the label order from the model config and the
raw outputs for three obviously entailed/neutral/contradicted pairs.
Reads the 3 scores per pair; compare with your expectations to confirm
the assumption at line 282 of hallucination_detector.py.
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


if __name__ == "__main__":
    main()
