"""Study queries and a strictly ephemeral practice path. No model execution on import."""

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import time
from urllib.parse import urlsplit

from filelock import FileLock

from src.ui.components.session_storage import SessionStorageError


class QueryTimer:
    """App/gate boundary: construct before durable request; sample before result flush."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()

    def elapsed_ms(self):
        return (self.clock() - self.started) * 1000


def presented_sources(sources):
    """Project citation metadata only; never edit the original answer."""
    result = []
    for source in sources or []:
        parts = []
        for key in ("provider", "service", "section"):
            value = source.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and value.strip().casefold() not in ("none", "n/a", "null")
            ):
                parts.append(value.strip())
        url = source.get("url")
        if (
            not isinstance(url, str)
            or urlsplit(url).scheme not in ("http", "https")
            or not urlsplit(url).netloc
        ):
            url = ""
        if parts or url:
            item = dict(label=" / ".join(parts) or urlsplit(url).netloc)
            if url:
                item["url"] = url
            if item not in result:
                result.append(item)
    return result


def presentation(response, condition):
    report = getattr(response, "hallucination_report", None)
    method = (
        report.get("method")
        if isinstance(report, dict)
        else getattr(report, "method", None)
    )
    if (
        response.error
        or not response.answer
        or not response.answer.strip()
        or response.confidence == "ERROR"
        or method in ("mixed", "keyword_fallback")
    ):
        raise ValueError("Query did not complete successfully")
    return dict(
        answer=response.answer,
        sources=presented_sources(response.sources) if condition == "hybrid" else [],
    )


def execute_query(
    condition, question, pipeline_factory, *, clock=time.perf_counter, capture=None
):
    """Shared app/gate path, from pipeline construction to ready-to-display payload."""
    started = clock()
    response = pipeline_factory(condition).query(question)
    if capture is not None:
        capture(response)
    payload = presentation(response, condition)
    return payload, (clock() - started) * 1000


def practice(session, pipeline_factory, on_started=None):
    """No attempt/checkpoint/timing/error record, even on retry. Return transient UI payload."""
    session.require("familiarization")
    with FileLock(str(session.store.root / "_inference.lock"), timeout=0):
        session.store.assert_active(session.session_id)
        # Single-inference deployment: suppress library diagnostics during practice;
        # do not let exception messages, prompts or durations reach persistent logs.
        previous = logging.root.manager.disable
        try:
            if on_started:
                on_started(time.time())
            logging.disable(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                condition = session.block["condition"]
                pipeline = pipeline_factory(condition)
                response = pipeline.query(
                    session.store.protocol["config"]["familiarization"]
                )
                return presentation(response, condition)
        except Exception:
            return None
        finally:
            logging.disable(previous)


def answer(
    session,
    pipeline_factory,
    free_question=None,
    on_started=None,
    clock=time.perf_counter,
    capture=None,
):
    """Clock: before durable request through ready-to-display payload, before final flush.

    Includes preparation residency check, retrieval/generation/NLI, and citation projection.
    Excludes lock waiting, preflight warmup, final result flush, browser/network and reading.
    """
    with FileLock(str(session.store.root / "_inference.lock"), timeout=0):
        session.store.assert_active(session.session_id)
        timer = QueryTimer(clock)
        session.begin(free_question)
        try:
            if on_started:
                on_started(session.pending["started_at"])
            condition = session.block["condition"]
            result, _ = execute_query(
                condition,
                session.pending["question"],
                pipeline_factory,
                clock=clock,
                capture=capture,
            )
        except SessionStorageError:
            raise
        except Exception:
            session.finish(error="query_failed", elapsed_ms=timer.elapsed_ms())
        else:
            # Storage failures must escape, but an OSError from Ollama is a query
            # failure. Keeping the final flush outside the query try distinguishes them.
            session.finish(**result, elapsed_ms=timer.elapsed_ms())


def recover(session):
    with FileLock(str(session.store.root / "_inference.lock"), timeout=0):
        session.store.assert_active(session.session_id)
        if session.pending and session.pending["status"] == "running":
            session.finish(error="interrupted", elapsed_ms=None)
