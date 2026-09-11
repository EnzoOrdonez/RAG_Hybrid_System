"""Opt-in, in-memory probes for a prepared pipeline; no production configuration edits.

The caller publishes export() outside the response clock using the durable gate.
Only sequential, isolated diagnostic workers may install these temporary probes.
"""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import time
from unittest.mock import patch


def paired_schedule():
    return [(system, index) for index in range(20)
            for system in (('hybrid', 'lexical') if index % 2 == 0 else ('lexical', 'hybrid'))]


def text_metrics(text):
    encoded = text.encode('utf-8')
    return dict(chars=len(text), utf8_bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())


def measure_traced_attempt(root, metadata, pipeline, observer=None, validate_after=None):
    """Use the existing durable response clock and publish probes with its terminal record.

    Preparation/admission belong to the caller. This refuses cold models, but cannot
    prove residency or clean conditions by itself. Aborted attempts keep the gate's
    journal/lower bound; partial in-memory probes are never invented on recovery.
    """
    from scripts import measure_interview_gate as gate

    # Install before the clock, without loading any model. Publication is handled
    # by measure_attempt after its response clock, as for the existing gate payload.
    with PipelineTrace(pipeline) as trace:
        def work():
            try:
                response = pipeline.query(metadata['query']['question']).model_dump(mode='json')
                report = response.get('hallucination_report') or {}
                error = response.get('error')
                if not error and (not response.get('answer', '').strip() or response.get('confidence') == 'ERROR'
                                  or report.get('method') in ('mixed', 'keyword_fallback')):
                    error = 'incomplete_response_or_verification'
                return dict(status='error' if error else 'success', error=error, response=response,
                            diagnostic_trace=trace.export())
            except Exception as exc:
                return dict(status='error', error=f'{type(exc).__name__}: {exc}',
                            diagnostic_trace=trace.export())

        return gate.measure_attempt(root, metadata, work, observer=observer, validate_after=validate_after)


class PipelineTrace:
    """Observe actual calls, preserving their arguments, results, and exceptions."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.records = dict(stages=[], generation=[], extractions=[], nli=[])
        self.stack = ExitStack()
        self.entered = False

    def _install(self, target, method, category, describe, describe_result=None):
        original = getattr(target, method)
        signature = inspect.signature(original)

        @wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            record = dict(describe(bound.arguments), started_at=datetime.now(timezone.utc).isoformat())
            self.records[category].append(record)
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                record['status'] = 'success'
                if describe_result is not None:
                    record.update(describe_result(result))
                return result
            except Exception as exc:
                record.update(status='error', error_type=type(exc).__name__)
                raise
            finally:
                record.update(elapsed_s=time.perf_counter() - started,
                              finished_at=datetime.now(timezone.utc).isoformat())

        self.stack.enter_context(patch.object(target, method, wrapped))

    def __enter__(self):
        detector = self.pipeline.hallucination_detector
        # Access storage, not the lazy property: instrumentation must not load weights.
        if detector._nli_model is None:
            raise ValueError('Trace requires a prepared NLI model')
        if self.entered:
            raise ValueError('Trace cannot be reused or nested')
        self.entered = True

        def generation(arguments):
            # signature.bind keeps **kwargs nested; support concrete and fake clients.
            merged = {**arguments.get('kwargs', {}), **arguments}
            prompt, system = merged['prompt'], merged.get('system_prompt') or ''
            return dict(prompt=prompt, system_prompt=system, prompt_chars=len(prompt),
                        prompt_metrics=text_metrics(prompt), system_metrics=text_metrics(system))

        def prediction(arguments):
            merged = {**arguments.get('kwargs', {}), **arguments}
            pairs = merged.get('sentences')
            if pairs is None:
                # Real CrossEncoder uses sentences; wrappers may expose *args.
                pairs = merged.get('args', ())[0]
            return dict(pairs=len(pairs), batch_size=merged.get('batch_size'),
                        apply_softmax=merged.get('apply_softmax'))

        try:
            from src.pipeline.rag_pipeline import LatencyTracker
            original_measure = LatencyTracker.measure

            @contextmanager
            def measured_stage(tracker, name):
                record = dict(name=name, started_at=datetime.now(timezone.utc).isoformat())
                self.records['stages'].append(record)
                started = time.perf_counter()
                try:
                    with original_measure(tracker, name):
                        yield
                    record['status'] = 'success'
                except Exception as exc:
                    record.update(status='error', error_type=type(exc).__name__)
                    raise
                finally:
                    record.update(elapsed_s=time.perf_counter() - started,
                                  finished_at=datetime.now(timezone.utc).isoformat())

            # Isolated sequential worker only: the class-level probe is restored
            # on exit and still delegates timing to the unmodified tracker.
            self.stack.enter_context(patch.object(LatencyTracker, 'measure', measured_stage))
            self._install(self.pipeline.llm, 'generate', 'generation', generation)
            self._install(detector, '_extract_claims', 'extractions', lambda args: {},
                          lambda result: dict(claims=list(result), count=len(result)))
            self._install(detector._nli_model, 'predict', 'nli', prediction)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def export(self):
        return dict(schema_version=1, **self.records,
                    nli_predict_calls=len(self.records['nli']),
                    nli_pairs_attempted=sum(r['pairs'] for r in self.records['nli']),
                    nli_pairs_successful=sum(r['pairs'] for r in self.records['nli']
                                             if r['status'] == 'success'))
