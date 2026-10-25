"""Isolated synthetic timeout probes; never change participant app configuration."""
import argparse
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import measure_interview_gate as gate


class TracedClient:
    """Transparent Ollama list/chat adapter with exclusive per-call evidence."""

    def __init__(self, client, output):
        self.client = client
        self.output = Path(output)

    def _call(self, method, **kwargs):
        record = dict(method=method, request=kwargs, started_at=gate.now())
        started = time.perf_counter()
        try:
            response = getattr(self.client, method)(**kwargs)
            record.update(status="success", server={
                key: response.get(key) for key in (
                    "total_duration", "load_duration", "prompt_eval_count",
                    "prompt_eval_duration", "eval_count", "eval_duration", "done_reason")})
            return response
        except Exception as exc:
            record.update(status="error", exception_type=f"{type(exc).__module__}.{type(exc).__name__}",
                          error=str(exc))
            raise
        finally:
            record.update(elapsed_s=time.perf_counter() - started, finished_at=gate.now())
            gate.write_new(self.output / f"{uuid.uuid4().hex}.json", record)

    def list(self):
        return self._call("list")

    def chat(self, **kwargs):
        return self._call("chat", **kwargs)


def run(output, cohort, phase, timeout):
    output = Path(output).resolve()
    common_git = Path(gate.git("rev-parse", "--git-common-dir")).resolve()
    if output.is_relative_to(common_git.parent) or output.is_relative_to(Path(cohort).resolve()):
        raise ValueError("Diagnostic output must be outside checkout and cohort")
    baseline = gate.read_json(Path(cohort) / "source-manifest.json")["protocol"]
    current = gate.environment_identity()
    # Diagnostic scripts may have a new commit; the measured app and runtime may not change.
    for key, value in baseline["environment"].items():
        if key not in ("commit", "runner_sha256") and current[key] != value:
            raise ValueError(f"Diagnostic runtime differs from cohort: {key}")
    gate.validate_app_identity(baseline["build_id"])
    protocol = dict(baseline, build_id=gate.git("rev-parse", "HEAD"), environment=current,
                    diagnostic_timeout_s=timeout, phase=phase, diagnostic_only=True,
                    cohort_manifest_sha256=gate.digest(Path(cohort) / "source-manifest.json"),
                    probe_sha256=gate.digest(__file__))
    gate.preflight(protocol)
    gate.write_new(output / "protocol.json", protocol)
    model = gate.check_model()
    if any(m["name"] != model for m in gate.api("/api/ps")["models"]):
        raise RuntimeError("Concurrent model detected")
    unloaded = gate.api("/api/generate", {"model": model, "keep_alive": 0})
    after = gate.api("/api/ps")
    if after["models"]:
        raise RuntimeError("Model did not unload")
    gate.write_new(output / "unload.json", dict(at=gate.now(), unload=unloaded, ps_after=after))
    pipeline = None
    target = next(q for q in baseline["queries"] if q["query_id"] == "q027")
    sequence = [baseline["queries"][0], target] if phase == "warm" else [target]
    for index, query in enumerate(sequence):
        warmup = phase == "warm" and index == 0
        call_root = output / ("warmup-http" if warmup else "measured-http")

        def work():
            nonlocal pipeline
            if pipeline is None:
                import httpx
                import ollama
                from src.ui.components.index_loader import load_hybrid_index, load_pipeline
                pipeline = load_pipeline("hybrid", _hybrid_index=load_hybrid_index())
                pipeline.llm.timeout = timeout  # isolated probe; app source/default stays at 60
                pipeline.llm._ollama_client = TracedClient(ollama.Client(
                    host=os.environ["OLLAMA_HOST"], timeout=httpx.Timeout(timeout, connect=5)), call_root)
            pipeline.llm._ollama_client.output = call_root
            if pipeline.llm.cache_enabled or pipeline.llm.seed != 42 or pipeline.llm.max_retries != 1:
                raise ValueError("Unexpected diagnostic recipe")
            response = pipeline.query(query["question"]).model_dump(mode="json")
            error = response.get("error")
            report = response.get("hallucination_report")
            if not error and (not response["answer"].strip() or response["confidence"] == "ERROR"
                              or report and report["method"] in ("keyword_fallback", "mixed")):
                error = "incomplete_response_or_verification"
            return dict(status="error" if error else "success", error=error, response=response,
                        configuration=pipeline.config.model_dump(mode="json"))

        result = gate.measure_attempt(output, dict(system="hybrid", phase=phase, index=index,
            warmup=warmup, query=query, timeout_s=timeout, diagnostic_only=True,
            build_id=protocol["build_id"], protocol_sha256=gate.digest(output / "protocol.json")),
            work, validate_after=lambda: gate.check_environment(protocol))
        print({k: result.get(k) for k in ("warmup", "status", "elapsed_s", "error")}, flush=True)
        if result.get("environment_invalid") or warmup and result["status"] != "success":
            raise RuntimeError("Diagnostic environment/warmup invalid; stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cold", "warm"))
    parser.add_argument("--timeout", required=True, type=int, choices=(60, 180))
    args = parser.parse_args()
    run(args.output, args.cohort, args.phase, args.timeout)
