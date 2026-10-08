"""Observe empty cold service state without rewriting actual telemetry samples."""
import math

from scripts.study_gate_environment import assess
from scripts.study_operator.stimulus_collection import cold_state


def assess_cold(rows, config, *, preparation_started, admission=False, allowed_pids=()):
    """Preserve every normal contamination check; allow only genuine initial emptiness."""
    if (type(preparation_started) not in (int, float) or not math.isfinite(preparation_started)
            or config.get('service_mode') != 'fresh_runner'):
        raise ValueError('Bounded cold preparation and fresh service required')
    reasons = set(assess(rows, config, admission=admission, allowed_pids=allowed_pids))
    normal_reasons = set()
    cold_count = 0
    generated = False
    for row in rows:
        marker = row.get('service_state', {})
        if marker.get('phase') != 'STARTING':
            generated = True
            normal_reasons.update(assess([row], config, allowed_pids=allowed_pids))
            continue
        try:
            cold_state(marker, config['service_boot_id'])
            at = row['monotonic_s']
            written = marker['written_monotonic_s']
            if (generated or type(at) not in (int, float) or not math.isfinite(at)
                    or type(written) not in (int, float) or not math.isfinite(written)
                    or not written <= at or not preparation_started <= at <= preparation_started+900
                    or row.get('ollama_ps_api', {}).get('models') != []):
                raise ValueError('Cold observation invalid')
            cold_count += 1
        except (ValueError, KeyError, TypeError):
            normal_reasons.add('cold_history_invalid')
    if cold_count:
        # These two reasons alone describe the expected cold state. All original
        # rows remain unchanged and all non-cold occurrences must still pass.
        reasons.difference_update({'service_transition', 'model_residency'})
    reasons.update(normal_reasons)
    if admission and cold_count != len(rows):
        reasons.add('cold_admission_requires_initial_empty_history')
    return sorted(reasons)
