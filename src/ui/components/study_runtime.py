"""Operator-selected configuration; no inference at import time."""
import os
import time

from filelock import FileLock
import streamlit as st

from src.ui.components.interview_preparation import Preparation
from src.ui.components.study_pipeline import build_study_pipeline
from src.ui.components.study_protocol import load_protocol
from src.ui.components.study_sessions import StudyStore
from src.ui.components.wait_feedback import wait_html


def store_from_env():
    protocol = load_protocol(os.environ['CLOUDRAG_STUDY_CONFIG'], os.environ['CLOUDRAG_STUDY_ASSIGNMENTS'])
    return StudyStore(os.environ['CLOUDRAG_STUDY_SESSION_DIR'], protocol,
                      os.environ.get('CLOUDRAG_STUDY_PURPOSE', 'study'))


@st.cache_resource
def preparation(root):
    return Preparation(root + '/_preparation', factory=build_study_pipeline,
                       systems=('hybrid', 'no_rag'), nli_systems=('hybrid',))


def prepare(session):
    with FileLock(str(session.store.root / '_inference.lock'), timeout=0):
        session.store.assert_active(session.session_id)
        return preparation(str(session.store.root)).prepare(session.session_id)


def render_wait(started):
    import streamlit.components.v1 as components
    html = wait_html(time.time() - started).replace('Buscando y verificando', 'Preparando')
    components.html(html, height=155)
