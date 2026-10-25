"""Test server: unchanged UI/storage; doubles only for models and remote backup."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

from scripts import cloud_storage
from src.ui.components import study_runtime
from src.ui.components.session_storage import atomic_json

root=Path(os.environ['CLOUDRAG_TEST_ROOT'])
os.environ.update(CLOUDRAG_STUDY_CONFIG=str(root/'config.json'),
    CLOUDRAG_STUDY_ASSIGNMENTS=str(root/'assignments.csv'),CLOUDRAG_STUDY_SESSION_DIR=str(root/'sessions'),
    CLOUDRAG_STUDY_PURPOSE='smoke',CLOUDRAG_BACKUP_BUCKET='synthetic-bucket',CLOUDRAG_BACKUP_PREFIX='synthetic')


class Backend:
    def __init__(self):
        self.prepared=False
    def ready(self,scope):
        return self.prepared
    def pipeline(self,condition,scope):
        def query(question):
            if os.environ.get('CLOUDRAG_TEST_CANARY_TRACE')=='1':
                trace={'ip':st.context.headers.get('X-Forwarded-For'),'ua':st.context.headers.get('User-Agent')}
                if not trace['ip'] and ' CloudRAG-IP/' in (trace['ua'] or ''):
                    trace['ip']=trace['ua'].split(' CloudRAG-IP/',1)[1]
                for domain in ('app','runner'):
                    folder=root/'canary-domains'/domain
                    folder.mkdir(parents=True,exist_ok=True)
                    atomic_json(folder/'trace.json',trace)
            return SimpleNamespace(answer='Synthetic response',error=None,confidence='LOW',sources=[],
                                   hallucination_report={'method':'nli'})
        return SimpleNamespace(query=query)


@st.cache_resource
def backend():
    return Backend()


def prepare(session):
    backend().prepared=True


def backup(folder,bucket,prefix):
    folder=Path(folder)
    files={name:(folder/name).read_bytes() for name in ('full_session.json','export_manifest.json')}
    output=root/'backups'/folder.name
    output.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        (output/name).write_bytes(data)
    if os.environ.get('CLOUDRAG_TEST_CANARY_TRACE')=='1':
        data=(root/'canary-domains'/'runner'/'trace.json').read_bytes()
        for domain in ('backups','bucket'):
            trace=root/'canary-domains'/domain
            trace.mkdir(parents=True,exist_ok=True)
            (trace/'trace.json').write_bytes(data)
    result={'status':'complete','sha256':hashlib.sha256(files['full_session.json']).hexdigest(),
        'objects':{name:{'object':'synthetic/P999/'+folder.name+'/'+name,'generation':'1',
                        'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}}
    atomic_json(folder/'backup_state.json',result)
    return result


study_runtime.preparation=lambda root:backend()
study_runtime.prepare=prepare
cloud_storage.backup_session=backup
from src.ui.views.study_page import render  # noqa: E402 - install backend doubles first

render()
