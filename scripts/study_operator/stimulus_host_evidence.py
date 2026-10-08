"""Bind LIVE cold observations to actual host admission and bounded telemetry."""
import hashlib
import json
import math

from scripts.study_operator.cold_admission import assess_cold


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def assessment_config(inventory):
    return dict(model_digest=inventory['ollama']['digest'], service_mode='fresh_runner',
                service_boot_id=inventory['observed']['boot_id'])


def reasons(samples, config, preparation_started, *, admission=False):
    # GPU and CPU ownership is taken at each host sample, not a global exemption
    # for processes whose name contains "ollama" or "python".
    result = set(assess_cold(samples, config, preparation_started=preparation_started,
                            admission=admission,
                            allowed_pids={pid for row in samples for pid in row['owned_pids']}))
    previous_busy = set()
    for row in samples:
        owned = set(row['owned_pids'])
        if (any(type(pid) is not int or pid <= 0 for pid in owned)
                or not set(row['gpu_pids']).issubset(owned)):
            result.add('foreign_gpu_process')
        busy = {p['pid'] for p in row['processes'] if p['pid'] not in owned
                and (p.get('cpu_percent') or 0) >= 10}
        if busy & previous_busy:
            result.add('external_cpu_load')
        previous_busy = busy
    return sorted(result)


def admission_receipt(host):
    return {key: host[key] for key in ('schema_version','boot_id','boot_index','app_image_id',
        'source_commit','own_container_id','ollama_container_id','preparation_started',
        'admission_started','admission_ended','admission_samples')}


def verify_host(proof, protocol_config):
    host = proof.get('host_evidence')
    if not isinstance(host, dict):
        raise ValueError('LIVE requires host-owned admission and telemetry evidence')
    inventory = proof['inventory']
    if (host.get('schema_version') != 1 or host.get('mode') != 'LIVE'
            or host.get('boot_id') != inventory['observed']['boot_id']
            or host.get('boot_index') != proof['boot_index']
            or host.get('app_image_id') != inventory['image']['image_id']
            or host.get('source_commit') != inventory['source']['commit']
            or host.get('isolation_verified') is not True or host.get('metadata_unreachable') is not True
            or host.get('native_limit_s') != 7200 or host.get('status') != 'COMPLETE'
            or not host.get('own_container_id') or not host.get('ollama_container_id')):
        raise ValueError('Host execution identity or independent limits differ')
    times = [host[key] for key in ('preparation_started','admission_started','admission_ended','ended')]
    if (any(type(t) not in (int,float) or not math.isfinite(t) for t in times)
            or times != sorted(times) or times[2]-times[0] > 900 or times[3]-times[0] > 7200):
        raise ValueError('Host preparation or cold boot deadline exceeded')
    samples = host['samples']
    admitted = [row for row in samples if times[1] <= row['monotonic_s'] <= times[2]]
    if (admitted != host['admission_samples'] or not admitted
            or admitted[0]['monotonic_s'] != times[1] or admitted[-1]['monotonic_s'] != times[2]
            or any(row['monotonic_s'] < times[0] or row['monotonic_s'] > times[3] for row in samples)
            or any(a['monotonic_s'] >= b['monotonic_s'] for a,b in zip(samples,samples[1:]))):
        raise ValueError('Host telemetry is missing, mixed or unordered')
    config = assessment_config(inventory)
    if (reasons(admitted,config,times[0],admission=True) or reasons(samples,config,times[0])
            or sha(admission_receipt(host)) != proof.get('host_admission_sha256')):
        raise ValueError('Host admission or continuous telemetry rejected')
    calls = host['calls']
    if len(calls) != len(proof['rows']):
        raise ValueError('Host call census differs')
    previous = times[2]
    for index,(call,row) in enumerate(zip(calls,proof['rows'],strict=True),1):
        start,end = call['started'],call['ended']
        if (call['index'] != index or not previous <= start < end <= times[3]
                or end-start > 600 or row['elapsed_s'] > end-start
                or call.get('row_sha256') != sha(row)
                or not any(0 <= start-s['monotonic_s'] <= 15 for s in samples)
                or not any(0 <= s['monotonic_s']-end <= 15 for s in samples)):
            raise ValueError('Host call frontier, deadline or observation binding differs')
        previous = end
    return dict(host_admission_verified=True, continuous_telemetry_verified=True)
