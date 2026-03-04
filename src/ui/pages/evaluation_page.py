"""
Evaluation Mode page for controlled within-subjects user study.

Flow: Login -> Training -> Evaluate System A -> Break -> Evaluate System B ->
      Break -> Evaluate System C -> SUS Questionnaire -> Open Questions -> Done
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import time

import streamlit as st

from src.ui.components.session_manager import (
    SYSTEM_DISPLAY_NAMES,
    SUS_QUESTIONS,
    TRAINING_QUERIES,
    EvaluationSession,
)


def _init_session_state():
    """Initialize evaluation session state."""
    if "eval_session" not in st.session_state:
        st.session_state.eval_session = None
    if "eval_query_shown_ts" not in st.session_state:
        st.session_state.eval_query_shown_ts = None
    if "eval_search_clicked_ts" not in st.session_state:
        st.session_state.eval_search_clicked_ts = None
    if "eval_response_shown_ts" not in st.session_state:
        st.session_state.eval_response_shown_ts = None
    if "eval_response_text" not in st.session_state:
        st.session_state.eval_response_text = None
    if "eval_training_idx" not in st.session_state:
        st.session_state.eval_training_idx = 0
    if "eval_break_start" not in st.session_state:
        st.session_state.eval_break_start = None


def _render_login():
    """Step 1: Participant login and consent."""
    st.subheader("Participant Login")

    with st.form("login_form"):
        participant_id = st.text_input(
            "Participant ID",
            placeholder="P01, P02, ..., P10",
        )
        experience = st.selectbox(
            "Cloud experience level",
            ["Principiante", "Intermedio", "Avanzado"],
        )
        consent = st.checkbox(
            "Acepto participar voluntariamente en esta evaluacion. "
            "Entiendo que mis respuestas seran anonimas y usadas "
            "solo para fines academicos."
        )
        submitted = st.form_submit_button("Comenzar Evaluacion", type="primary")

    if submitted:
        if not participant_id:
            st.error("Please enter a Participant ID.")
            return
        if not consent:
            st.error("You must accept the consent to participate.")
            return

        # Check for existing session
        existing = EvaluationSession.load_checkpoint(participant_id)
        if existing and existing.state != "complete":
            st.info(f"Found existing session for {participant_id}. Resuming...")
            st.session_state.eval_session = existing
        else:
            st.session_state.eval_session = EvaluationSession(participant_id, experience)

        st.session_state.eval_session.save_checkpoint()
        st.rerun()


def _render_training():
    """Step 2: Training with 3 practice queries."""
    session = st.session_state.eval_session
    idx = st.session_state.eval_training_idx

    st.subheader("Training Phase")
    st.info(
        "Vas a interactuar con un sistema de consulta de documentacion cloud. "
        "Para cada pregunta:\n"
        "1. Lee la pregunta\n"
        "2. Presiona 'Buscar'\n"
        "3. Lee la respuesta\n"
        "4. Califica la utilidad y precision\n\n"
        "No hay respuestas correctas o incorrectas. "
        "Estas 3 preguntas son de practica y NO se registraran."
    )

    if idx >= len(TRAINING_QUERIES):
        st.success("Training complete!")
        if st.button("Begin Evaluation", type="primary"):
            session.training_complete = True
            session.state = "evaluating"
            session.save_checkpoint()
            st.rerun()
        return

    query = TRAINING_QUERIES[idx]
    st.write(f"**Practice {idx + 1}/{len(TRAINING_QUERIES)}**")
    st.write(f"> {query}")

    # Search button
    if st.session_state.eval_response_text is None:
        if st.button("Buscar respuesta", key="training_search"):
            try:
                from src.ui.components.index_loader import load_hybrid_index, load_pipeline
                hybrid_index = load_hybrid_index()
                pipeline = load_pipeline("hybrid", _hybrid_index=hybrid_index)
                response = pipeline.query(query)
                st.session_state.eval_response_text = response.answer or "No response generated."
            except Exception as e:
                st.session_state.eval_response_text = f"[Retrieval-only mode] Error: {e}"
            st.rerun()
    else:
        st.markdown("**Response:**")
        st.write(st.session_state.eval_response_text)

        if st.button("Next Practice Question", key="training_next"):
            st.session_state.eval_training_idx += 1
            st.session_state.eval_response_text = None
            st.rerun()


def _render_evaluation():
    """Step 3/5/7: Evaluate queries for current system."""
    session = st.session_state.eval_session
    query = session.current_query

    if query is None:
        # All queries for this system done
        new_state = session.advance()
        session.save_checkpoint()
        st.rerun()
        return

    # Show progress
    system_label = session.current_system_label
    q_idx = session.current_query_index + 1
    total_q = len(session.current_queries)

    st.subheader(f"{system_label}")
    st.progress(q_idx / total_q, text=f"Question {q_idx}/{total_q}")
    st.caption(f"Overall progress: {session.progress_pct:.0f}%")

    question = query.get("question", "")
    st.write(f"**Question {q_idx}/{total_q}**")
    st.write(f"> {question}")

    # Record query shown timestamp
    if st.session_state.eval_query_shown_ts is None:
        st.session_state.eval_query_shown_ts = time.time()

    # Search button
    if st.session_state.eval_response_text is None:
        if st.button("Buscar respuesta", key="eval_search", type="primary"):
            st.session_state.eval_search_clicked_ts = time.time()
            try:
                from src.ui.components.index_loader import load_hybrid_index, load_pipeline
                system_key = session.current_system
                hybrid_index = load_hybrid_index()
                pipeline = load_pipeline(system_key, _hybrid_index=hybrid_index)
                response = pipeline.query(question)
                st.session_state.eval_response_text = response.answer or "No response generated."
            except Exception as e:
                st.session_state.eval_response_text = f"[Error: {e}]"
            st.session_state.eval_response_shown_ts = time.time()
            st.rerun()
    else:
        # Show response
        st.divider()
        st.markdown("**System Response:**")
        st.write(st.session_state.eval_response_text)
        st.divider()

        # Rating form
        with st.form(f"rating_form_{session.current_system}_{q_idx}"):
            st.write("**Que tan UTIL fue esta respuesta?**")
            utility = st.radio(
                "Utility",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {1: "1 - Nada util", 2: "2", 3: "3", 4: "4", 5: "5 - Muy util"}[x],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.write("**Que tan PRECISA parece esta respuesta?**")
            accuracy = st.radio(
                "Accuracy",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {1: "1 - Incorrecta", 2: "2", 3: "3", 4: "4", 5: "5 - Correcta"}[x],
                horizontal=True,
                label_visibility="collapsed",
            )

            submitted = st.form_submit_button("Siguiente pregunta →", type="primary")

        if submitted:
            rated_ts = time.time()
            session.record_rating(
                utility=utility,
                accuracy=accuracy,
                query_shown_ts=st.session_state.eval_query_shown_ts or rated_ts,
                search_clicked_ts=st.session_state.eval_search_clicked_ts or rated_ts,
                response_shown_ts=st.session_state.eval_response_shown_ts or rated_ts,
                rated_ts=rated_ts,
            )

            # Reset query state
            st.session_state.eval_query_shown_ts = None
            st.session_state.eval_search_clicked_ts = None
            st.session_state.eval_response_shown_ts = None
            st.session_state.eval_response_text = None

            new_state = session.advance()
            session.save_checkpoint()
            st.rerun()


def _render_break():
    """Step 4/6: Break between systems."""
    session = st.session_state.eval_session

    st.subheader("Descanso")

    if st.session_state.eval_break_start is None:
        st.session_state.eval_break_start = time.time()

    elapsed = time.time() - st.session_state.eval_break_start
    min_break = 120  # 2 minutes minimum
    max_break = 600  # 10 minutes maximum

    remaining = max(0, min_break - elapsed)

    if remaining > 0:
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        st.info(
            f"Toma un descanso. El siguiente sistema comenzara "
            f"cuando estes listo.\n\n"
            f"Tiempo minimo restante: **{mins}:{secs:02d}**"
        )
        # Auto-refresh every second
        time.sleep(1)
        st.rerun()
    else:
        st.success("Puedes continuar cuando estes listo.")

    if remaining <= 0 or st.button("Estoy listo (continuar)", type="primary"):
        st.session_state.eval_break_start = None
        session.state = "evaluating"
        session.save_checkpoint()
        st.rerun()


def _render_sus():
    """Step 8: SUS Questionnaire."""
    session = st.session_state.eval_session

    st.subheader("Cuestionario de Usabilidad (SUS)")
    st.info(
        "Por favor, indica tu nivel de acuerdo con cada afirmacion. "
        "Evalua la experiencia general con los sistemas que probaste."
    )

    with st.form("sus_form"):
        responses = []
        for i, question in enumerate(SUS_QUESTIONS):
            st.write(f"**{i + 1}.** {question}")
            resp = st.radio(
                f"Q{i+1}",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "1 - Totalmente en desacuerdo",
                    2: "2 - En desacuerdo",
                    3: "3 - Neutral",
                    4: "4 - De acuerdo",
                    5: "5 - Totalmente de acuerdo",
                }[x],
                horizontal=True,
                label_visibility="collapsed",
                key=f"sus_{i}",
            )
            responses.append(resp)
            if i < len(SUS_QUESTIONS) - 1:
                st.divider()

        submitted = st.form_submit_button("Enviar cuestionario", type="primary")

    if submitted:
        session.sus_responses = responses
        session.state = "open_questions"
        session.save_checkpoint()
        st.rerun()


def _render_open_questions():
    """Step 9: Open-ended questions."""
    session = st.session_state.eval_session

    st.subheader("Preguntas Abiertas")

    with st.form("open_form"):
        q1 = st.text_area(
            "1. Cual sistema prefirio y por que?",
            height=100,
        )
        q2 = st.text_area(
            "2. Noto diferencias entre los sistemas?",
            height=100,
        )
        q3 = st.text_area(
            "3. Que mejoraria del sistema?",
            height=100,
        )

        submitted = st.form_submit_button("Finalizar evaluacion", type="primary")

    if submitted:
        session.open_responses = {
            "preferred_system": q1,
            "noticed_differences": q2,
            "improvements": q3,
        }
        session.state = "complete"
        session.save_checkpoint()
        session.export_results()
        st.rerun()


def _render_complete():
    """Step 10: Thank you screen."""
    session = st.session_state.eval_session

    st.subheader("Gracias por tu participacion!")
    st.balloons()

    # Summary
    total_queries = session.total_queries_answered
    sus_score = session.calculate_sus_score()

    col1, col2, col3 = st.columns(3)
    col1.metric("Queries Answered", total_queries)
    col2.metric("SUS Score", f"{sus_score:.1f}/100")
    col3.metric("Systems Evaluated", len(session.system_order))

    st.info(f"Results exported to: `data/evaluation/user_sessions/{session.participant_id}/`")

    # Option to start new session
    if st.button("New Evaluation Session"):
        st.session_state.eval_session = None
        st.session_state.eval_training_idx = 0
        st.session_state.eval_response_text = None
        st.session_state.eval_query_shown_ts = None
        st.session_state.eval_break_start = None
        st.rerun()


def render():
    """Render the Evaluation Mode page."""
    st.header("Evaluation Mode")

    _init_session_state()

    session = st.session_state.eval_session

    # No session yet -> show login
    if session is None:
        _render_login()
        return

    # Route based on session state
    state = session.state

    # Show progress sidebar info
    with st.sidebar:
        st.divider()
        st.subheader("Evaluation Progress")
        st.write(f"**Participant:** {session.participant_id}")
        st.write(f"**State:** {state}")
        st.progress(session.progress_pct / 100)
        st.write(f"System order: {' → '.join(s.upper()[:3] for s in session.system_order)}")

        if session.current_system:
            st.write(f"**Current:** {session.current_system_label}")

    if state == "training":
        _render_training()
    elif state == "evaluating":
        _render_evaluation()
    elif state == "break":
        _render_break()
    elif state == "sus":
        _render_sus()
    elif state == "open_questions":
        _render_open_questions()
    elif state == "complete":
        _render_complete()
    else:
        st.error(f"Unknown state: {state}")
