"""Process-local preparation with append-only external evidence and live residency checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import uuid

SYSTEMS = ('hybrid', 'lexical', 'semantic')
MODEL = 'granite4.1:8b'
KEEP_ALIVE = '30m'
WARM_QUERY = 'What is the SLA for AWS Lambda?'
PROJECT = Path(__file__).resolve().parents[3]


class PreparationRequired(RuntimeError):
    """No new participant inference until preparation succeeds."""


def resident_models():
    url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434').rstrip('/') + '/api/ps'
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def load_pipeline(system):
    from src.ui.components import index_loader
    return index_loader.load_pipeline(system, _hybrid_index=index_loader.load_hybrid_index())


class Preparation:
    def __init__(self, root, factory=None, probe=None, clock=None):
        self.root = Path(root).resolve()
        if self.root.is_relative_to(PROJECT.parent.parent):
            raise ValueError('Preparation evidence must be outside checkout')
        self.factory = factory or load_pipeline
        self.probe = probe or resident_models
        self.clock = clock or time.time
        self.instance = uuid.uuid4().hex
        self.pipelines = {}
        self.receipt = None
        self.bound_identity = None
        self.last_check = None
        self.lock = threading.RLock()

    def _identity(self):
        manifest = Path(os.environ['CLOUDRAG_ARTIFACT_MANIFEST'])
        return dict(pid=os.getpid(), instance=self.instance, build=os.environ['CLOUDRAG_BUILD_ID'],
                    model_digest=os.environ['CLOUDRAG_MODEL_DIGEST'],
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())

    def _event(self, kind, scope, **data):
        self.root.mkdir(parents=True, exist_ok=True)
        event_id = uuid.uuid4().hex
        event = dict(id=event_id, kind=kind, scope=scope, pid=os.getpid(), instance=self.instance,
                     at=datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(), **data)
        with (self.root / f'{event_id}.json').open('x', encoding='utf-8') as stream:
            json.dump(event, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def _resident(self, identity):
        payload = self.probe()
        models = payload.get('models', [])
        if len(models) != 1:
            raise PreparationRequired('Expected sole study model resident')
        model = models[0]
        expiry = datetime.fromisoformat(model['expires_at'])
        if (model.get('name', model.get('model')) != MODEL
                or model.get('digest', '').removeprefix('sha256:') != identity['model_digest']
                or expiry.tzinfo is None or expiry.timestamp() < self.clock() + 180
                or ('context_length' in model and model['context_length'] != 4096)):
            raise PreparationRequired('Residency identity, context or remaining lease invalid')
        return payload

    def ready(self, scope):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.receipt is None or self.receipt['scope'] != scope:
                return False
            try:
                identity = self._identity()
                if identity != self.receipt['identity'] or set(self.pipelines) != set(SYSTEMS):
                    raise PreparationRequired('Preparation identity changed')
                resident = self._resident(identity)
                self.last_check = dict(at=self.clock(), identity=identity, resident=resident)
                return True
            except Exception as exc:
                previous = self.receipt['id']
                self.receipt = None
                self._event('preparation_invalidated', scope, previous=previous,
                            error_type=type(exc).__name__)
                return False
        finally:
            self.lock.release()

    def pipeline(self, system, scope):
        with self.lock:
            if not self.ready(scope):
                raise PreparationRequired('Study deployment needs preparation')
            pipeline = self.pipelines[system]
            pipeline.interview_preparation_check = self.last_check
            return pipeline

    def prepare(self, scope):
        if not isinstance(scope, str) or not scope:
            raise ValueError('Preparation requires a session or cohort scope')
        with self.lock:
            if self.ready(scope):
                return self.receipt
            self.receipt = None
            identity = self._identity()
            if self.bound_identity is not None and identity != self.bound_identity:
                self._event('restart_required', scope, identity=identity, previous=self.bound_identity)
                raise PreparationRequired('Restart process after deployment identity changes')
            self.bound_identity = identity
            operation = self._event('preparation_started', scope, identity=identity,
                                    warm_query=WARM_QUERY, keep_alive=KEEP_ALIVE)
            started = time.perf_counter()
            stage = 'loading'
            try:
                import numpy as np
                for system in SYSTEMS:
                    stage = f'loading:{system}'
                    pipeline = self.factory(system)
                    stage = f'query:{system}'
                    query_started = time.perf_counter()
                    response = pipeline.query(WARM_QUERY).model_dump(mode='json')
                    warm_response = self._event('warmup_response', scope, operation=operation['id'],
                        system=system, response=response, elapsed_s=time.perf_counter() - query_started)
                    report = response.get('hallucination_report') or {}
                    if (response.get('error') or not response.get('answer', '').strip()
                            or response.get('confidence') == 'ERROR'
                            or report.get('method') in ('mixed', 'keyword_fallback')):
                        raise PreparationRequired(f'Warmup failed: {system}')
                    stage = f'nli:{system}'
                    model = pipeline.hallucination_detector.nli_model
                    if model is None:
                        raise PreparationRequired('NLI unavailable')
                    scores = np.asarray(model.predict(
                        [('CloudRAG is a technical test system.', 'CloudRAG is a technical test system.')],
                        batch_size=1, show_progress_bar=False, apply_softmax=True))
                    if (scores.shape != (1, 3) or not np.isfinite(scores).all()
                            or (scores < 0).any() or (scores > 1).any()
                            or not np.isclose(scores.sum(), 1)):
                        raise PreparationRequired('NLI warmup returned invalid probabilities')
                    self.pipelines[system] = pipeline
                    self._event('pipeline_prepared', scope, operation=operation['id'], system=system,
                                warm_response_id=warm_response['id'], nli_probe=scores.tolist())
                stage = 'identity_and_residency'
                if self._identity() != identity:
                    raise PreparationRequired('Identity changed while preparing')
                resident = self._resident(identity)
                self.receipt = self._event('preparation_ready', scope, operation=operation['id'],
                    status='ready', identity=identity, systems=list(SYSTEMS), resident=resident,
                    elapsed_s=time.perf_counter() - started)
                for pipeline in self.pipelines.values():
                    pipeline.interview_preparation_id = self.receipt['id']
                return self.receipt
            except Exception as exc:
                self._event('preparation_failed', scope, operation=operation['id'],
                            error_type=type(exc).__name__, stage=stage, elapsed_s=time.perf_counter() - started)
                raise
