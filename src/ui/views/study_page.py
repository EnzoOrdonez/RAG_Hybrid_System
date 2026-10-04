"""Participant-only presentation. Instrument text comes from the frozen configuration."""
import streamlit as st

from src.ui.components import study_runtime as runtime, study_service as service


def response(payload):
    st.markdown(payload['answer'])
    if payload['sources']:
        with st.expander('Fuentes'):
            for source in payload['sources']:
                st.write(source['label'])
                if source.get('url'):
                    st.link_button('Abrir fuente', source['url'])


def render():
    st.title('Sesión')
    try:
        store = runtime.store_from_env()
        store.check()
    except Exception:
        st.error('La sesión necesita preparación. Contacta al coordinador.')
        return
    if st.session_state.get('study_closed'):
        st.success('La sesión ha terminado. Gracias por participar.')
        return
    if 'study_token' not in st.session_state:
        with st.form('login'):
            token = st.text_input('Invitación', type='password')
            if st.form_submit_button('Entrar'):
                try:
                    store.admit(token)
                    st.session_state.study_token = token
                    st.rerun()
                except (ValueError, RuntimeError):
                    st.error('No se pudo abrir la sesión. Contacta al coordinador.')
        return
    try:
        session = store.admit(st.session_state.study_token)
        view = st.empty()
        with view.container():
            _flow(session, view)
    except Exception:
        st.error('No se pudo continuar. Contacta al coordinador antes de reintentar.')


def _flow(session, view):
    stage = session.data['stage']
    config = session.store.protocol['config']
    if stage in ('complete', 'abandoned'):
        if stage == 'complete':
            session.export()
            st.success('La sesión ha terminado. Gracias por participar.')
        else:
            st.info('La sesión está cerrada. Contacta al coordinador.')
        return
    if stage not in ('comparative', 'blinding'):
        st.subheader('Sistema ' + session.block['label'])
    if stage in ('familiarization', 'tasks', 'free_query'):
        prep = runtime.preparation(str(session.store.root))
        if not prep.ready(session.session_id):
            st.info('El coordinador debe preparar la sesión antes de continuar.')
            if st.button('Preparar sesión'):
                runtime.prepare(session)
                st.rerun()
            return
        def factory(condition):
            return prep.pipeline(condition, session.session_id)
        if stage == 'familiarization':
            key = 'practice_' + str(session.data['block_index'])
            st.write('Prueba de familiarización')
            st.write(config['familiarization'])
            if st.button('Probar consulta'):
                st.session_state[key] = service.practice(session, factory, runtime.render_wait)
            if st.session_state.get(key):
                response(st.session_state[key])
                if st.button('Comenzar tareas'):
                    del st.session_state[key]
                    session.familiarization_done()
                    st.rerun()
            else:
                st.caption('Prueba la consulta antes de continuar. Si no aparece una respuesta, avisa al coordinador.')
            return
        pending = session.pending
        if pending and pending['status'] == 'running':
            st.info('Hay una consulta pendiente. Contacta al coordinador si la conexión se interrumpió.')
            if st.button('Comprobar consulta pendiente'):
                service.recover(session)
                st.rerun()
            return
        if pending and pending['status'] == 'success':
            st.write(pending['question'])
            response(pending)
            session.shown()
            if st.button('Continuar'):
                session.acknowledge()
                st.rerun()
            return
        if pending and pending['status'] == 'error':
            st.error('No se pudo completar la consulta. Puedes reintentar o avisar al coordinador.')
        free = None
        if stage == 'free_query':
            st.write(config['free_instruction'])
            free = st.text_area('Tu consulta', max_chars=6000, key='free_' + str(session.data['block_index']))
        else:
            qid = config['tasks'][session.block['task_set']][session.data['task_index']]
            st.write(session.store.protocol['queries'][qid]['question'])
        if st.button('Consultar', disabled=stage == 'free_query' and not (free or '').strip()):
            service.answer(session, factory, free, runtime.render_wait)
            st.rerun()
    elif stage == 'instruments':
        st.caption('1 = Totalmente en desacuerdo · 5 = Totalmente de acuerdo')
        block = session.data['block_index']
        def submit():
            sus = [st.session_state[f'sus_{block}_{i}'] for i in range(10)]
            likert = {item['id']: st.session_state[f'likert_{block}_{item["id"]}'] for item in config['likert']}
            if None in sus or None in likert.values():
                st.error('Responde todos los ítems antes de continuar.')
            else:
                try:
                    session.submit_instruments(sus, likert)
                except Exception:
                    st.error('La sesión cambió o no se pudo guardar. Contacta al coordinador y vuelve a comprobar las respuestas.')
        with st.form('block_' + str(block)):
            for i, text in enumerate(config['sus']['items']):
                st.radio(text, range(1, 6), index=None, horizontal=True, key=f'sus_{block}_{i}')
            for item in config['likert']:
                st.radio(item['text'], range(1, 6), index=None, horizontal=True, key=f'likert_{block}_{item["id"]}')
            st.form_submit_button('Guardar respuestas del bloque', on_click=submit)
    elif stage == 'comparative':
        with st.form('comparative'):
            values = {item['id']: (st.radio(item['text'], item['choices'], index=None)
                      if item['choices'] else st.text_area(item['text'])) for item in config['comparative']}
            if st.form_submit_button('Continuar al cierre'):
                if None in values.values():
                    st.error('Responde todos los ítems antes de continuar.')
                else:
                    session.submit_comparative(values)
                    view.empty()
                    st.rerun()
    elif stage == 'blinding':
        with st.form('blinding'):
            choice = st.radio(config['blinding_question'], config['blinding_choices'], index=None)
            reason = st.text_area(config['blinding_reason'])
            if st.form_submit_button('Finalizar'):
                if choice is None:
                    st.error('Selecciona una opción antes de continuar.')
                else:
                    session.submit_blinding(choice, reason)
                    st.session_state.study_closed = True
                    del st.session_state.study_token
                    view.empty()
                    st.rerun()
