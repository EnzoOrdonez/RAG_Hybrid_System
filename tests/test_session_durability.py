"""Publication ordering, failures and deterministic POSIX writer interruption."""
import multiprocessing
import os

import pytest

from src.ui.components import session_storage as storage


def test_publication_syncs_file_before_replace_and_parent_after(tmp_path, monkeypatch):
    events = []
    replace = os.replace
    monkeypatch.setattr(storage.os, "fsync", lambda fd: events.append("file_sync"))

    def publish(source, target):
        assert events == ["file_sync"]
        replace(source, target)
        events.append("replace")

    def parent(directory):
        assert directory == tmp_path
        assert (directory / "record.json").read_text() == '{"new":true}'
        events.append("parent_sync")

    monkeypatch.setattr(storage.os, "replace", publish)
    monkeypatch.setattr(storage, "_fsync_parent", parent)
    storage.atomic_text(tmp_path / "record.json", '{"new":true}')
    assert events == ["file_sync", "replace", "parent_sync"]


@pytest.mark.parametrize("point", ["file_sync", "replace", "parent_sync"])
def test_durability_errors_propagate_and_prepublication_preserves_old(tmp_path, monkeypatch, point):
    path = tmp_path / "record.json"
    storage.atomic_json(path, {"confirmed": True})
    before = path.read_bytes()

    def fail(*args):
        raise OSError("durability failure")

    if point == "parent_sync":
        monkeypatch.setattr(storage, "_fsync_parent", fail)
    else:
        monkeypatch.setattr(storage.os, "fsync" if point == "file_sync" else "replace", fail)
    with pytest.raises(OSError, match="durability failure"):
        storage.atomic_json(path, {"new": True})
    assert path.read_bytes() == before if point != "parent_sync" else storage.read_json(path) == {"new": True}
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory descriptors")
@pytest.mark.parametrize("fail_sync", [False, True])
def test_parent_descriptor_is_directory_and_always_closed(tmp_path, monkeypatch, fail_sync):
    calls = []
    open_fd, sync_fd, close_fd = os.open, os.fsync, os.close

    def opened(path, flags):
        assert flags & os.O_DIRECTORY
        fd = open_fd(path, flags)
        calls.append(("open", fd))
        return fd

    def synced(fd):
        calls.append(("sync", fd))
        if fail_sync:
            raise OSError("directory sync failure")
        sync_fd(fd)

    def closed(fd):
        calls.append(("close", fd))
        close_fd(fd)

    monkeypatch.setattr(storage.os, "open", opened)
    monkeypatch.setattr(storage.os, "fsync", synced)
    monkeypatch.setattr(storage.os, "close", closed)
    if fail_sync:
        with pytest.raises(OSError, match="directory sync failure"):
            storage._fsync_parent(tmp_path)
    else:
        storage._fsync_parent(tmp_path)
    assert [name for name, _ in calls] == ["open", "sync", "close"]
    assert len({fd for _, fd in calls}) == 1


def _blocked_writer(path, ready, release):
    def before_replace(source, target):
        # atomic_text has already flushed and fsynced its sibling file.
        ready.send(str(source))
        release.wait(30)
        raise RuntimeError("parent must kill writer before publication")

    storage.os.replace = before_replace
    storage.atomic_json(path, {"unconfirmed": True})


@pytest.mark.skipif(os.name != "posix", reason="POSIX deterministic kill before rename")
def test_killed_writer_preserves_last_confirmed_record(tmp_path):
    path = tmp_path / "record.json"
    storage.atomic_json(path, {"confirmed": True})
    before = path.read_bytes()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    release = context.Event()
    writer = context.Process(target=_blocked_writer, args=(path, sender, release))
    writer.start()
    try:
        assert receiver.poll(15), "writer never reached pre-rename barrier"
        temporary = receiver.recv()
        assert storage.read_json(temporary) == {"unconfirmed": True}
        writer.kill()
        writer.join(10)
        assert not writer.is_alive() and writer.exitcode < 0
        assert path.read_bytes() == before
        assert storage.read_json(path) == {"confirmed": True}
    finally:
        if writer.is_alive():
            writer.kill()
            writer.join(10)
        receiver.close()
        sender.close()
