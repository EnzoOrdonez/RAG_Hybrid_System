"""Study queries and a strictly ephemeral practice path. No model execution on import."""
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import time
from urllib.parse import urlsplit

from filelock import FileLock

from src.ui.components.session_storage import SessionStorageError


def presented_sources(sources):
    """Project citation metadata only; never edit the original answer."""
    result = []
    for source in sources or []:
        parts = []
        for key in ('provider', 'service', 'section'):
            value = source.get(key)
            if isinstance(value, str) and value.strip() and value.strip().casefold() not in ('none', 'n/a', 'null'):
                parts.append(value.strip())
        url = source.get('url')
        if not isinstance(url, str) or urlsplit(url).scheme not in ('http', 'https') or not urlsplit(url).netloc:
            url = ''
        if parts or url:
            item = dict(label=' / '.join(parts) or urlsplit(url).netloc)
            if url:
                item['url'] = url
            if item not in result:
                result.append(item)
    return result


def presentation(response, condition):
    report = getattr(response, 'hallucination_report', None)
    method = report.get('method') if isinstance(report, dict) else getattr(report, 'method', None)
    if response.error or not response.answer or not response.answer.strip() or response.confidence == 'ERROR' or method in ('mixed', 'keyword_fallback'):
        raise ValueError('Query did not complete successfully')
    return dict(answer=response.answer, sources=presented_sources(response.sources) if condition == 'hybrid' else [])


def practice(session, pipeline_factory, on_started=None):
    """No attempt/checkpoint/timing/error record, even on retry. Return transient UI payload."""
    session.require('familiarization')
    with FileLock(str(session.store.root / '_inference.lock'), timeout=0):
        session.store.assert_active(session.session_id)
        # Single-inference deployment: suppress library diagnostics during practice;
        # do not let exception messages, prompts or durations reach persistent logs.
        previous = logging.root.manager.disable
        try:
            if on_started:
                on_started(time.time())
            logging.disable(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                condition = session.block['condition']
                pipeline = pipeline_factory(condition)
                response = pipeline.query(session.store.protocol['config']['familiarization'])
                return presentation(response, condition)
        except Exception:
            return None
        finally:
            logging.disable(previous)


def answer(session, pipeline_factory, free_question=None, on_started=None, clock=time.perf_counter):
    """Clock: before durable request through ready-to-display payload, before final flush.

    Includes preparation residency check, retrieval/generation/NLI, and citation projection.
    Excludes lock waiting, preflight warmup, final result flush, browser/network and reading.
    """
    with FileLock(str(session.store.root / '_inference.lock'), timeout=0):
        session.store.assert_active(session.session_id)
        started = clock()
        session.begin(free_question)
        try:
            if on_started:
                on_started(session.pending['started_at'])
            condition = session.block['condition']
            pipeline = pipeline_factory(condition)
            result = presentation(pipeline.query(session.pending['question']), condition)
            session.finish(**result, elapsed_ms=(clock() - started) * 1000)
        except (SessionStorageError, OSError):
            raise
        except Exception:
            session.finish(error='query_failed', elapsed_ms=(clock() - started) * 1000)


def recover(session):
    with FileLock(str(session.store.root / '_inference.lock'), timeout=0):
        session.store.assert_active(session.session_id)
        if session.pending and session.pending['status'] == 'running':
            session.finish(error='interrupted', elapsed_ms=None)
