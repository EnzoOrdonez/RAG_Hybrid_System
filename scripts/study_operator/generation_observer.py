"""Measurement-only observation at the Ollama client boundary; bytes are unchanged."""
from contextlib import contextmanager

from scripts.study_operator.stimulus_evidence import digest


class ClientObserver:
    def __init__(self, client, records):
        self.client, self.records = client, records

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, **kwargs):
        # The manager has already added num_ctx and verified the model digest.
        self.records.append(dict(request_sha256=digest(kwargs),
                                 generation_options=dict(kwargs['options'])))
        return self.client.chat(**kwargs)


@contextmanager
def observe(manager, records):
    """Restore method overrides and retain precisely the real client's state."""
    if not manager.enforce_timeout:
        raise ValueError('Measured study requires the bounded client')
    original = manager._ollama_chat
    overridden = '_ollama_chat' in vars(manager)
    previous = vars(manager).get('_ollama_chat')
    if manager._ollama_client is not None:
        manager._ollama_client = ClientObserver(manager._ollama_client, records)

    def capture(module, **kwargs):
        class ModuleObserver:
            def Client(self, *args, **values):
                return ClientObserver(module.Client(*args, **values), records)
        return original(ModuleObserver(), **kwargs)

    manager._ollama_chat = capture
    try:
        yield
    finally:
        if isinstance(manager._ollama_client, ClientObserver):
            manager._ollama_client = manager._ollama_client.client
        if overridden:
            manager._ollama_chat = previous
        else:
            del manager._ollama_chat
