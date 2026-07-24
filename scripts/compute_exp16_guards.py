"""exp16 anti-gaming guards — per arm, so a faithfulness gain can't hide copying/abstaining.

A prompt that raises NLI/HHEM faithfulness by (a) copying chunk text verbatim or (b) declining
more often is gaming the instrument, not improving grounded answering. Alongside faithfulness
(computed elsewhere), this reports per arm:
  - decline_rate: answers that are the canonical insufficient-info abstention
  - mean_answer_chars / mean_answer_words: completeness proxy (short = less said)
  - mean_genuine_claims: from faithfulness_rows (how much is actually asserted+scored)
  - verbatim_overlap: mean over answers of the fraction of answer word-5-grams that also
    appear in the concatenated retrieved chunk text (high = copy-paste)
A "better" arm should raise faithfulness WITHOUT collapsing length/claims or spiking overlap.

Usage: python scripts/compute_exp16_guards.py --exp-dir experiments/results/exp16_anchored_decoding
Writes: <exp-dir>/guards.{json,md}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
DECLINE = "cannot find sufficient information to fully answer this question"


def word_ngrams(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if len(w) >= n else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--rows-verifier", default="small",
                    help="which faithfulness_rows to read genuine-claim counts from")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)

    res = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    rows_path = exp_dir / f"faithfulness_rows__{args.rows_verifier}__vb_agree.json"
    rows = (json.loads(rows_path.read_text(encoding="utf-8"))["configs"]
            if rows_path.exists() else {})

    arms = []
    for cname, c in res.items():
        answers, chars, words, overlaps, declines = 0, [], [], [], 0
        for r in c["results"]:
            a = r.get("answer") or ""
            answers += 1
            chars.append(len(a))
            words.append(len(re.findall(r"\w+", a)))
            if DECLINE in a:
                declines += 1
            # verbatim overlap: answer 5-grams vs concatenated chunk 5-grams
            ctext = " ".join(chunk_map[cid]["text"] for cid in r["retrieved_ids"]
                             if cid in chunk_map)
            ans_ng = word_ngrams(a)
            if ans_ng:
                chunk_ng = word_ngrams(ctext)
                overlaps.append(len(ans_ng & chunk_ng) / len(ans_ng))
        gvals = [v.get("genuine") for v in rows.get(cname, {}).values()
                 if v.get("genuine") is not None]
        arms.append({
            "config": cname,
            "scenario": c.get("scenario", cname.split(" | ")[0]),
            "n": answers,
            "decline_rate": round(declines / answers, 4) if answers else None,
            "n_decline": declines,
            "mean_answer_chars": round(float(np.mean(chars)), 1) if chars else None,
            "mean_answer_words": round(float(np.mean(words)), 1) if words else None,
            "mean_genuine_claims": round(float(np.mean(gvals)), 3) if gvals else None,
            "verbatim_overlap_5gram": round(float(np.mean(overlaps)), 4) if overlaps else None,
        })

    out = {"experiment_id": exp_dir.name, "rows_verifier": args.rows_verifier,
           "decline_phrase": DECLINE, "overlap": "mean frac of answer word-5-grams in chunks",
           "arms": arms, "generated_by": "scripts/compute_exp16_guards.py"}
    (exp_dir / "guards.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = [f"# {exp_dir.name} — anti-gaming guards (rows: {args.rows_verifier})", "",
         "| Arm | n | decline | words | genuine claims | verbatim 5gram overlap |",
         "|---|---|---|---|---|---|"]
    for a in arms:
        L.append(f"| {a['scenario']} | {a['n']} | {a['decline_rate']} ({a['n_decline']}) | "
                 f"{a['mean_answer_words']} | {a['mean_genuine_claims']} | "
                 f"{a['verbatim_overlap_5gram']} |")
    L += ["", "Read WITH arm_stats faithfulness: a real gain raises faithfulness while decline/",
          "overlap stay near baseline and words/claims don't collapse. High overlap or high",
          "decline alongside a faithfulness gain = instrument gaming, not improvement."]
    (exp_dir / "guards.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
