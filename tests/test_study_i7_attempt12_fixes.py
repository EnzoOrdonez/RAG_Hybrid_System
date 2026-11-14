"""Intento 11 (ADR-0031): the host's window for the owner's acknowledgement.

In both failed stimulus boots (intento 10 boot 2, intento 11 boot 5) the owner downloaded and
verified the proof and then lost the ACK: the cold-boot host gave the owner only 60 s after
``result.json`` and its unit then ran ``shutdown -h now``. Saving the proof and finishing the ACK
alone took 34-38 s in the boots that passed, and each bridge call is a chain of gcloud calls.
These tests pin the corrected window and the grace that lets the ACK reply leave before shutdown.
"""
from pathlib import Path


class FakeClock:
    """Monotonic clock whose sleep advances time and fires scheduled file writes."""

    def __init__(self, start=1000.0):
        self.now = start
        self.events = []
        self.slept = []

    def at(self, moment, path):
        self.events.append((moment, Path(path)))

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds
        for moment, path in list(self.events):
            if self.now >= moment:
                path.write_text('{}', encoding='utf-8')
                self.events.remove((moment, path))


def test_owner_window_is_shared_by_stimulus_and_gate_and_is_ten_minutes():
    from scripts.study_operator import host_runtime

    assert host_runtime.OWNER_ACK_WINDOW_S == 600
    assert 0 < host_runtime.POST_ACK_GRACE_S <= 60


def test_stimulus_host_waits_past_one_minute_for_a_slow_owner(tmp_path):
    from scripts.study_operator.stimulus_host import await_download_ack

    fake = FakeClock()
    fake.at(fake.now + 75, tmp_path / 'download-ack.json')  # Boot 5 needed more than 60 s.
    seen = await_download_ack(tmp_path, bound=fake.now + 7200, clock=fake.clock, sleep=fake.sleep)
    assert seen is True


def test_stimulus_host_lingers_after_the_ack_so_the_bridge_reply_returns(tmp_path):
    from scripts.study_operator.host_runtime import POST_ACK_GRACE_S
    from scripts.study_operator.stimulus_host import await_download_ack

    fake = FakeClock()
    acked_at = fake.now + 40
    fake.at(acked_at, tmp_path / 'download-ack.json')
    await_download_ack(tmp_path, bound=fake.now + 7200, clock=fake.clock, sleep=fake.sleep)
    assert fake.now >= acked_at + POST_ACK_GRACE_S


def test_stimulus_host_still_stops_when_the_owner_never_acknowledges(tmp_path):
    from scripts.study_operator.host_runtime import OWNER_ACK_WINDOW_S
    from scripts.study_operator.stimulus_host import await_download_ack

    fake = FakeClock()
    began = fake.now
    seen = await_download_ack(tmp_path, bound=began + 7200, clock=fake.clock, sleep=fake.sleep)
    assert seen is False
    assert began + OWNER_ACK_WINDOW_S <= fake.now <= began + OWNER_ACK_WINDOW_S + 1


def test_stimulus_wait_never_passes_the_independent_native_bound(tmp_path):
    from scripts.study_operator.stimulus_host import await_download_ack

    fake = FakeClock()
    bound = fake.now + 30  # The 7200-s cold-boot limit is almost spent.
    fake.at(fake.now + 20, tmp_path / 'download-ack.json')
    await_download_ack(tmp_path, bound=bound, clock=fake.clock, sleep=fake.sleep)
    assert fake.now <= bound + 1


def test_gate_host_lingers_after_the_finalization_ack(tmp_path):
    from scripts.study_operator.host_runtime import POST_ACK_GRACE_S
    from scripts.study_operator.i7_gate_host import await_finalization_ack

    fake = FakeClock()
    acked_at = fake.now + 90
    fake.at(acked_at, tmp_path / 'finalization-ack.json')
    seen = await_finalization_ack(tmp_path, bound=fake.now + 9000, clock=fake.clock, sleep=fake.sleep)
    assert seen is True
    assert fake.now >= acked_at + POST_ACK_GRACE_S


def test_gate_host_keeps_its_ten_minute_window_and_host_bound(tmp_path):
    from scripts.study_operator.host_runtime import OWNER_ACK_WINDOW_S
    from scripts.study_operator.i7_gate_host import await_finalization_ack

    fake = FakeClock()
    began = fake.now
    assert await_finalization_ack(tmp_path, bound=began + 9000, clock=fake.clock, sleep=fake.sleep) is False
    assert began + OWNER_ACK_WINDOW_S <= fake.now <= began + OWNER_ACK_WINDOW_S + 1

    fake = FakeClock()
    began = fake.now
    assert await_finalization_ack(tmp_path, bound=began + 45, clock=fake.clock, sleep=fake.sleep) is False
    assert fake.now <= began + 46
