"""Validate bounded, host-written service transitions without altering ps samples."""
import math
import re


def transition(marker, observed, expected_boot, *, admission=False):
    if not isinstance(marker, dict) or marker.get('schema_version') != 1 or marker.get('mode') != 'fresh_runner':
        return False, False
    if not expected_boot or marker.get('boot_id') != expected_boot:
        return False, False
    phase = marker.get('phase')
    if phase not in {'RESETTING', 'LOADING', 'GENERATING', 'RESIDENT'}:
        return False, False
    if not re.fullmatch('[a-f0-9]{32}', marker.get('request_id', '')):
        return False, False
    started, deadline = marker.get('written_monotonic_s'), marker.get('deadline_monotonic_s')
    if not isinstance(started, (float, int)) or not math.isfinite(started) or started > observed:
        return False, False
    if phase != 'RESIDENT':
        maximum = {'RESETTING': 10, 'LOADING': 30, 'GENERATING': 600}[phase]
        if (not isinstance(deadline, (float, int)) or not math.isfinite(deadline)
                or not started < deadline <= started + maximum or not observed < deadline):
            return False, False
    elif deadline is not None or not marker.get('runner_pids'):
        return False, False
    return True, not admission and phase in {'RESETTING', 'LOADING'}
