"""Interview actions, independent of Streamlit and exercised with model doubles."""

import logging
import os
import time

from filelock import FileLock

from src.ui.components import session_manager as sm
from src.ui.components.session_storage import InvitationStore

logger = logging.getLogger(__name__)


def _dump(value):
    if value is None:
        return None
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def recover_interrupted(session):
    # Acquiring this lock proves no other request in this deployment is still running.
    with FileLock(str(sm.SESSIONS_DIR / "_inference.lock"), timeout=0):
        InvitationStore(sm.SESSIONS_DIR).assert_active(session.session_id)
        if session.pending_attempt and session.pending_attempt["status"] == "running":
            session.finish_attempt(session.pending_attempt["attempt_id"], error="interrupted")


def answer_query(session, pipeline_factory, query_shown_ts=None):
    """Persist request before work and the exact presentation payload before rating."""
    with FileLock(str(sm.SESSIONS_DIR / "_inference.lock"), timeout=0):
        InvitationStore(sm.SESSIONS_DIR).assert_active(session.session_id)
        aid = session.begin_attempt({"build_id": os.environ.get("CLOUDRAG_BUILD_ID", "unverified")},
                                    time.time() if query_shown_ts is None else query_shown_ts)
        started = time.perf_counter()
        try:
            pipeline = pipeline_factory(session.current_system)
            config = _dump(pipeline.config)
            config.update(seed=pipeline.llm.seed, max_tokens=1024,
                          num_ctx=getattr(pipeline.llm, "num_ctx", None),
                          cache_enabled=pipeline.llm.cache_enabled,
                          build_id=os.environ.get("CLOUDRAG_BUILD_ID", "unverified"),
                          artifact_manifest_sha256=getattr(getattr(pipeline, "hybrid_index", None), "deployment_manifest_sha256", None))
            session.pending_attempt["configuration"] = config
            session.save_checkpoint()
            response = pipeline.query(session.current_query["question"])
            session.pending_attempt["configuration"]["model_digest"] = getattr(pipeline.llm, "model_digest", None)
            report = _dump(response.hallucination_report)
            error = "pipeline_error" if response.error else None
            if not error and (not response.answer or not response.answer.strip() or response.confidence == "ERROR"):
                error = "empty_response"
            if not error and report and report.get("method") in ("keyword_fallback", "mixed"):
                error = "verification_unavailable"
            session.finish_attempt(
                aid, answer=None if error else response.answer,
                sources=response.sources, chunks=response.retrieved_chunks,
                verification=report, error=error,
                elapsed_ms=(time.perf_counter() - started) * 1000)
        except (sm.SessionStorageError, OSError):
            raise
        except Exception as exc:
            # Do not expose provider URLs, credentials or stack traces to participants.
            logger.error("Interview attempt %s failed (%s)", aid, type(exc).__name__)
            session.finish_attempt(aid, error="generation_failed",
                                   elapsed_ms=(time.perf_counter() - started) * 1000)


def submit_rating(session, utility, accuracy):
    attempt = session.pending_attempt
    if not attempt or attempt["status"] != "success" or not attempt.get("shown_at"):
        raise ValueError("La respuesta debe mostrarse antes de calificarla.")
    session.record_rating(utility, accuracy, attempt["query_shown_ts"], attempt["started_at"],
                          attempt["shown_at"], time.time())
    session.advance()
    if session.state == "break":
        session.break_started_at = time.time()
    session.save_checkpoint()
