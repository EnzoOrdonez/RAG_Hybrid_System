"""Free determinism check across a re-score: an arm scored twice must come out identical.

Scoring is deterministic re-aggregation of a fixed model over fixed text -- unlike
generation, it carries no H5 cold/warm exposure. So when a re-run re-scores an arm that was
already scored, the second result MUST equal the first, bit for bit. If it does not,
something else changed (model weights, claim extractor, thresholds, tokenizer) and NOTHING
downstream should be trusted until that is explained.

exp18 hands us this check for free: the botched pass scored only `baseline_repro`, and the
fixed re-run scores it again on the way to the other three arms.

  snapshot   before re-running, copy the current artifacts aside
  compare    after, diff the overlapping arms cell by cell

Compares `faithfulness_rows__{verifier}__vb_agree.json`, `nli_probs__{verifier}.json.gz`
(raw probabilities -- the strictest of the three) and `claims_extraction.json` (the
extractor is verifier-independent, so a change there would mean the claim set itself moved).

Usage:
  python scripts/check_rescore_determinism.py snapshot --exp-dir <dir> [--verifier small]
  python scripts/check_rescore_determinism.py compare  --exp-dir <dir> [--verifier small]
Exit 0 = identical on every overlapping arm; 1 = drift (STOP and diagnose).
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAP_DIRNAME = "_determinism_snapshot"


def artifacts(exp_dir: Path, verifier: str):
    return [exp_dir / f"faithfulness_rows__{verifier}__vb_agree.json",
            exp_dir / f"nli_probs__{verifier}.json.gz",
            exp_dir / "claims_extraction.json"]


def load(p: Path):
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["snapshot", "compare"])
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--verifier", default="small", choices=["small", "base"])
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)
    snap = exp_dir / SNAP_DIRNAME / args.verifier

    if args.action == "snapshot":
        snap.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in artifacts(exp_dir, args.verifier):
            if p.exists():
                shutil.copy2(p, snap / p.name)
                n += 1
                print(f"  saved {p.name}")
        if not n:
            sys.exit(f"nothing to snapshot in {exp_dir} for verifier {args.verifier}")
        print(f"snapshot -> {snap} ({n} artifacts)")
        return 0

    if not snap.exists():
        sys.exit(f"no snapshot at {snap}; run `snapshot` BEFORE re-scoring")

    drift, checked = [], 0
    for p in artifacts(exp_dir, args.verifier):
        old_p = snap / p.name
        if not old_p.exists() or not p.exists():
            continue
        old, new = load(old_p), load(p)
        old_cfg, new_cfg = old.get("configs", {}), new.get("configs", {})
        shared = sorted(set(old_cfg) & set(new_cfg))
        if not shared:
            print(f"  {p.name}: no overlapping arm to compare")
            continue
        for cname in shared:
            checked += 1
            if old_cfg[cname] != new_cfg[cname]:
                # locate the first differing query so the report is actionable
                o, n_ = old_cfg[cname], new_cfg[cname]
                keys = sorted(set(o) | set(n_))
                first = next((k for k in keys if o.get(k) != n_.get(k)), None)
                drift.append((p.name, cname, first,
                              len([k for k in keys if o.get(k) != n_.get(k)])))
                print(f"  DRIFT {p.name} :: {cname} — {drift[-1][3]} query(ies) differ, "
                      f"first={first}")
            else:
                print(f"  OK    {p.name} :: {cname} (identical)")

    if drift:
        print(f"\n{len(drift)} artifact/arm pair(s) DRIFTED across the re-score.")
        print("Scoring is deterministic over fixed text, so this cannot be noise: something "
              "changed (model weights, claim extractor, thresholds, tokenizer). STOP and "
              "diagnose before trusting anything downstream.")
        return 1
    print(f"\nAll {checked} overlapping arm(s) identical across the re-score. "
          f"Scoring reproduced exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
