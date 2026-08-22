"""Measure whether the exp19b Ollama session fingerprint drifts while the server is idle.

This probe reports observations only. It does not recommend a pipeline design decision.

Usage: python scripts/probe_fingerprint_drift.py [--n 6] [--interval-min 10]
Smoke: python scripts/probe_fingerprint_drift.py --n 2 --interval-min 0
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp19b_pipeline import _live_warmup, capture_fingerprint  # noqa: E402
from src.utils.signed_evidence import guard_write  # noqa: E402

PROBE_DIR = PROJECT_ROOT / "experiments/probes/runtime_noise"
STABLE_VERDICT = "huella estable en idle"
DRIFT_VERDICT = "huella deriva en idle"


def live_capture():
    """Run the pipeline's exact warmup and retain its response token count."""
    from src.generation.llm_manager import LLMManager

    responses = []
    original_generate = LLMManager.generate

    def recording_generate(manager, *args, **kwargs):
        response = original_generate(manager, *args, **kwargs)
        responses.append(response)
        return response

    with patch.object(LLMManager, "generate", recording_generate):
        fingerprint = capture_fingerprint(_live_warmup)

    if len(responses) != 1:
        raise RuntimeError(f"expected one warmup generation, observed {len(responses)}")
    response = responses[0]
    if response.error:
        raise RuntimeError(f"warmup generation failed: {response.error}")
    return fingerprint, int(response.tokens_output)


def collect_probe(n, interval_min, capture_fn=live_capture, now_fn=None, sleep_fn=None):
    """Collect captures with injectable time and generator dependencies for offline tests."""
    if n < 2:
        raise ValueError("n must be at least 2 to observe a transition")
    if interval_min < 0:
        raise ValueError("interval-min must be non-negative")

    now_fn = now_fn or (lambda: datetime.now().astimezone())
    sleep_fn = sleep_fn or time.sleep
    captures = []
    previous = None

    for index in range(n):
        fingerprint, tokens_out = capture_fn()
        timestamp = now_fn()
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        captures.append({
            "timestamp": str(timestamp),
            "fingerprint": fingerprint,
            # The first capture has no predecessor; False keeps the field boolean and it is
            # excluded from transition counts below.
            "identical_to_previous": False if previous is None else fingerprint == previous,
            "tokens_out": int(tokens_out),
        })
        previous = fingerprint
        if index < n - 1 and interval_min:
            sleep_fn(interval_min * 60)

    changed = sum(not row["identical_to_previous"] for row in captures[1:])
    return {
        "probe": "exp19b Ollama session fingerprint drift while idle",
        "generated_by": "scripts/probe_fingerprint_drift.py",
        "n": n,
        "interval_min": interval_min,
        "transitions_total": n - 1,
        "transitions_changed": changed,
        "verdict": DRIFT_VERDICT if changed else STABLE_VERDICT,
        "captures": captures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--interval-min", type=float, default=10)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least 2")
    if args.interval_min < 0:
        parser.error("--interval-min must be non-negative")

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    output = PROBE_DIR / f"fingerprint_drift_{date.today().isoformat()}.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing probe: {output}")

    report = collect_probe(args.n, args.interval_min)
    guard_write(output).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"transiciones que cambiaron: {report['transitions_changed']}/"
          f"{report['transitions_total']}")
    print(report["verdict"])


if __name__ == "__main__":
    main()
