"""exp16 anti-gaming guards — per arm, so a faithfulness gain can't hide copying/abstaining.

A prompt that raises NLI/HHEM faithfulness by (a) copying chunk text verbatim or (b) declining
more often is gaming the instrument, not improving grounded answering. Alongside faithfulness
(computed elsewhere), this reports per arm:
  - declination, split into pure_decline / hedged_partial / answered (see below)
  - mean_answer_chars / mean_answer_words: completeness proxy (short = less said)
  - mean_genuine_claims: from faithfulness_rows (how much is actually asserted+scored)
  - verbatim_overlap: mean over answers of the fraction of answer word-5-grams that also
    appear in the concatenated retrieved chunk text (high = copy-paste)
A "better" arm should raise faithfulness WITHOUT collapsing length/claims or spiking overlap.

DECLINE RULE, unified 2026-07-30
--------------------------------
This file used to test ONE exact case-sensitive substring while the faithfulness metric used
`classify_response` over 28 case-insensitive patterns (the 14 canonical
response_formatter.DECLINE_PATTERNS plus EXTENDED_REFUSAL_PATTERNS). The two disagreed by
4-24 points on EVERY arm, so the declination quoted in the ledger was not the declination the
metric saw. Both verdicts survive the correction and get stronger (exp16 raises declination
about twice as much as reported; exp17 lowers it more), but two live definitions of
"declination" in one repo is exactly how a number ends up meaning nothing.

Now it imports `classify_response` itself — same function, not a copy — and reports its THREE
classes, because the distinction carries the whole interpretation:
  pure_decline    refusal marker in the first 300 chars: the model led with a refusal
  hedged_partial  refusal marker later on: it hedged AND still answered
  answered        no refusal marker anywhere
That split matters: 37/60 of the Tier A baseline answers contain a refusal phrase yet still
assert claims and are scored normally (q002 declines, then answers 128 words, and scores
faithfulness 1.0 on 1 genuine claim). Calling those "declines" overstates abstention, and
`decline_rate` alone cannot tell an arm that truly abstains from one that merely hedges.

Usage: python scripts/compute_exp16_guards.py --exp-dir experiments/results/exp16_anchored_decoding
Writes: <exp-dir>/guards.{json,md}
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"

# The canonical classifier, imported from the module the faithfulness metric uses, so the
# guards and the metric can never drift apart again. Guarded by tests/test_decline_rule.py.
_spec = importlib.util.spec_from_file_location(
    "cfm_guards", PROJECT_ROOT / "scripts" / "compute_faithfulness_metrics.py")
_cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfm)
classify_response = _cfm.classify_response


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
        answers, chars, words, overlaps = 0, [], [], []
        klass = {"pure_decline": 0, "hedged_partial": 0, "answered": 0, "empty": 0}
        for r in c["results"]:
            a = r.get("answer") or ""
            answers += 1
            chars.append(len(a))
            words.append(len(re.findall(r"\w+", a)))
            klass[classify_response(a) or "empty"] += 1
            # verbatim overlap: answer 5-grams vs concatenated chunk 5-grams
            ctext = " ".join(chunk_map[cid]["text"] for cid in r["retrieved_ids"]
                             if cid in chunk_map)
            ans_ng = word_ngrams(a)
            if ans_ng:
                chunk_ng = word_ngrams(ctext)
                overlaps.append(len(ans_ng & chunk_ng) / len(ans_ng))
        gvals = [v.get("genuine") for v in rows.get(cname, {}).values()
                 if v.get("genuine") is not None]
        # "any refusal marker" = pure + hedged; kept because it is the quantity the ledger
        # historically quoted, now computed with the canonical rule.
        any_refusal = klass["pure_decline"] + klass["hedged_partial"]
        # DEFECT #7 (ledger entry 22): the class rates above are PREFIX rates. Most rows
        # labelled pure_decline still assert claims, so quoting `pure_decline_rate` as "how
        # often the system refused" overstates abstention by an order of magnitude. The
        # content-based rate is measured here from the same rows the metric scores, so any
        # downstream argument about usability reads the right number.
        arm_rows = rows.get(cname, {})
        silent = [r["query_id"] for r in c["results"]
                  if not _cfm.asserts_content((arm_rows.get(r["query_id"]) or {}).get("genuine"))]
        decline_prefix_but_answers = sum(
            1 for r in c["results"]
            if classify_response(r.get("answer") or "") == "pure_decline"
            and _cfm.asserts_content((arm_rows.get(r["query_id"]) or {}).get("genuine")))
        arms.append({
            "config": cname,
            "scenario": c.get("scenario", cname.split(" | ")[0]),
            "n": answers,
            "pure_decline_rate": round(klass["pure_decline"] / answers, 4) if answers else None,
            "hedged_partial_rate": round(klass["hedged_partial"] / answers, 4) if answers else None,
            "answered_rate": round(klass["answered"] / answers, 4) if answers else None,
            "any_refusal_rate": round(any_refusal / answers, 4) if answers else None,
            "n_by_class": klass,
            # content-based, not prefix-based (defect #7)
            "asserts_nothing_rate": (round(len(silent) / answers, 4)
                                     if (answers and arm_rows) else None),
            "n_asserts_nothing": len(silent) if arm_rows else None,
            "n_decline_prefix_that_still_answers": decline_prefix_but_answers if arm_rows else None,
            "mean_answer_chars": round(float(np.mean(chars)), 1) if chars else None,
            "mean_answer_words": round(float(np.mean(words)), 1) if words else None,
            "mean_genuine_claims": round(float(np.mean(gvals)), 3) if gvals else None,
            "verbatim_overlap_5gram": round(float(np.mean(overlaps)), 4) if overlaps else None,
        })

    out = {"experiment_id": exp_dir.name, "rows_verifier": args.rows_verifier,
           "decline_rule": ("scripts/compute_faithfulness_metrics.classify_response — the SAME "
                            "function the faithfulness metric uses (14 canonical "
                            "DECLINE_PATTERNS + 14 EXTENDED_REFUSAL_PATTERNS, case-insensitive; "
                            "pure_decline = marker within the first 300 chars)"),
           "decline_prefix_caveat": (
               "READ `pure_decline` AS `decline_prefix`. It is a PREFIX test, not a refusal: "
               "most rows so labelled go on to assert claims from parametric knowledge "
               "(\"...However, I can outline general steps...\"). Use `asserts_nothing_rate` "
               "for any usability or abstention argument. Defect #7, ledger entry 22."),
           "overlap": "mean frac of answer word-5-grams in chunks",
           "arms": arms, "generated_by": "scripts/compute_exp16_guards.py"}
    (exp_dir / "guards.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = [f"# {exp_dir.name} — anti-gaming guards (rows: {args.rows_verifier})", "",
         "Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, "
         "la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).", "",
         "**`decline_prefix` (antes `pure_decline`) mide un PREFIJO, no un rechazo.** La mayoría "
         "de esas filas sí afirman claims (memoria paramétrica tras el prefijo). Para cualquier "
         "argumento de usabilidad usar **`no afirma nada`**. Defecto #7, entrada 22.", "",
         "| Arm | n | decline_prefix | hedged | answered | any refusal | **no afirma nada** | "
         "prefijo pero contesta | words | genuine claims | overlap 5gram |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        k = a["n_by_class"]
        L.append(f"| {a['scenario']} | {a['n']} | {a['pure_decline_rate']} ({k['pure_decline']}) | "
                 f"{a['hedged_partial_rate']} ({k['hedged_partial']}) | "
                 f"{a['answered_rate']} ({k['answered']}) | {a['any_refusal_rate']} | "
                 f"**{a['asserts_nothing_rate']}** ({a['n_asserts_nothing']}) | "
                 f"{a['n_decline_prefix_that_still_answers']} | "
                 f"{a['mean_answer_words']} | {a['mean_genuine_claims']} | "
                 f"{a['verbatim_overlap_5gram']} |")
    L += ["", "Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin "
              "disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta "
              "junto a una ganancia de fidelidad = gaming del instrumento, no mejora.", "",
          "**Ni `hedged_partial` ni `decline_prefix` son abstención.** Ambas clases llevan una "
          "frase de rechazo y aun así afirman claims, y se puntúan normal. La afirmación previa "
          "de este mismo archivo —«un brazo que sube `pure_decline` sí está callándose»— era "
          "**falsa** y queda retirada (defecto #7): la mayoría de esas filas contestan de memoria "
          "paramétrica tras el prefijo. El brazo que de verdad se calla es el que sube "
          "**`no afirma nada`**, que es la columna medida por contenido."]
    (exp_dir / "guards.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
