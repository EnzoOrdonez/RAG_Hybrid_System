"""Host commands tolerate GPU startup without losing their finite bound."""
import subprocess
from types import SimpleNamespace

import pytest

from scripts.study_operator.stimulus_host import HostCollection


def test_host_command_waits_up_to_thirty_seconds():
    observed = []

    def invoke(argv, **kwargs):
        observed.append((argv, kwargs['timeout']))
        return SimpleNamespace(returncode=0, stdout=b'GPU ready')

    host = object.__new__(HostCollection)
    host.invoke = invoke
    assert host.command(['nvidia-smi']) == b'GPU ready'
    assert observed == [(['nvidia-smi'], 30)]


def test_host_command_timeout_is_terminal():
    def invoke(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])

    host = object.__new__(HostCollection)
    host.invoke = invoke
    with pytest.raises(subprocess.TimeoutExpired) as error:
        host.command(['nvidia-smi'])
    assert error.value.timeout == 30
