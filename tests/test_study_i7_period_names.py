"""cpu-build-02: every I7 attempt owns its build prefix and CPU names; no remnant is reread."""
import hashlib
import json
from datetime import datetime, timezone

import pytest

from scripts.study_operator import cpu_build
from tests.test_study_cpu_build import fixture

A = '09e7ae37e3b440988c4707369ba56d62'
B = '867c595a9e19400c8d27a5b1c3e4f601'


def launched_prefix(root, monkeypatch, **scope):
    root.mkdir(parents=True)
    _, cloud, _, inventory, *_ = fixture(root, monkeypatch)
    build = cpu_build.Build(root, cloud, now=lambda: datetime(2026, 10, 8, tzinfo=timezone.utc), **scope)
    build.launch('final01', inventory, b'guest')
    return json.loads((root/'cpu-build-final01-job-input.json').read_bytes())['prefix']


def test_two_i7_periods_never_share_a_build_prefix(tmp_path, monkeypatch):
    from scripts.study_operator.i7_cpu_pipeline import build_scope
    # Both attempts keep their build state in a directory named cpu, as in incident cpu-build-02.
    first = launched_prefix(tmp_path/'attempt1'/'cpu', monkeypatch, scope=build_scope({'period_id': B}))
    second = launched_prefix(tmp_path/'attempt2'/'cpu', monkeypatch, scope=build_scope({'period_id': A}))
    assert first == 'iteration4/iteration5/i7-'+B[:12]+'/final01'
    assert second == 'iteration4/iteration5/i7-'+A[:12]+'/final01'


def test_legacy_build_prefix_is_unchanged(tmp_path, monkeypatch):
    assert launched_prefix(tmp_path/'cpu', monkeypatch) == 'iteration4/iteration5/cpu/final01'


@pytest.mark.parametrize('scope', ['', 'cpu/../x', 'I7-UPPER', 'x'*33])
def test_unsafe_build_scope_is_rejected(tmp_path, scope):
    with pytest.raises(ValueError, match='Safe build scope required'):
        cpu_build.Build(tmp_path, object(), scope=scope)


def test_cpu_label_names_and_marker_carry_the_period(tmp_path):
    from scripts.study_operator.cpu_restoration import Controller
    from scripts.study_operator.i7_cpu_pipeline import cpu_label
    first, second = cpu_label({'period_id': B}), cpu_label({'period_id': A})
    assert (first, second) == ('i7cpu-'+B[:8], 'i7cpu-'+A[:8])
    for label in (first, second):
        assert Controller(tmp_path, object(), label=label).label == label
        assert cpu_build.Build(tmp_path, object(), cpu_label=label).cpu_label == label
    # The ownership marker is CloudRAG-I5-restore-<root name>-<label>; it now names the period.
    assert ('CloudRAG-I5-restore-cpu-'+second).endswith(A[:8])


def cleanup_owner(root, name, status):
    from types import SimpleNamespace
    archive = root/'private'/'legacy'/'i3-nine-originals.tar'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'synthetic archive')
    vm = {'name': name, 'id': '11', 'description': 'CloudRAG-I5-restore-cpu-'+name[len('cloudrag-i5-restore-'):]}
    state = {'cpu_resources': {'vm': vm}, 'legacy_archive_sha256': hashlib.sha256(b'synthetic archive').hexdigest()}
    calls = []
    class Cloud:
        def command(self, args, **kwargs):
            calls.append(args)
            assert args[:3] == ['compute', 'instances', 'describe'], 'No destructive call in this fixture'
            return dict(vm, status=status)
    return SimpleNamespace(root=root, cloud=Cloud(), state=state, config={'period_id': A}), calls


def test_cleanup_refuses_cpu_named_for_another_attempt(tmp_path):
    from scripts.study_operator.i7_cpu_pipeline import cleanup_cpu
    owner, calls = cleanup_owner(tmp_path, 'cloudrag-i5-restore-i7cpu01', 'TERMINATED')
    with pytest.raises(ValueError, match='^CPU_CLEANUP_IDENTITY_CHANGED$'):
        cleanup_cpu(owner)
    assert [c[:3] for c in calls] == [['compute', 'instances', 'describe']]


def test_cleanup_accepts_the_cpu_of_its_own_period(tmp_path):
    from scripts.study_operator.i7_cpu_pipeline import cleanup_cpu
    owner, _ = cleanup_owner(tmp_path, 'cloudrag-i5-restore-i7cpu-'+A[:8], 'RUNNING')
    # Identity passes; the next guard (stopped VM) is what blocks this synthetic RUNNING CPU.
    with pytest.raises(ValueError, match='^CPU_CLEANUP_REQUIRES_STOPPED_VM$'):
        cleanup_cpu(owner)
