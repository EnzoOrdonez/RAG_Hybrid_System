"""Read-only Windows observer. No power/model settings or other processes are changed.

Sensor failures invalidate controls; thermal/power limits and memory pressure do not.
JSONL is flushed/fsynced per observation and survives interruption.
"""
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import uuid

BALANCED = '381b4222-f694-41f0-9685-ff5bb260df2e'
BEST_PERFORMANCE = 'ded574b5-45a0-4f42-8737-46345c09c238'
GPU_FIELDS = ('utilization.gpu', 'memory.total', 'memory.used', 'memory.free',
              'temperature.gpu', 'power.draw', 'power.limit', 'clocks.current.graphics',
              'clocks.current.memory', 'clocks_event_reasons.active',
              'clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.hw_slowdown',
              'clocks_event_reasons.hw_thermal_slowdown',
              'clocks_event_reasons.hw_power_brake_slowdown',
              'clocks_event_reasons.sw_thermal_slowdown')


def command(args):
    return subprocess.check_output(args, text=True, timeout=4,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()


def windows_state():
    """Native getters avoid changing the machine and expose effective overlay, not just registry."""
    class Power(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('flag', ctypes.c_ubyte),
                    ('percent', ctypes.c_ubyte), ('reserved', ctypes.c_ubyte),
                    ('life', ctypes.c_uint32), ('full', ctypes.c_uint32)]

    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual', 'virtual_available', 'extended')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    power, memory = Power(), Memory()
    memory.length = ctypes.sizeof(memory)
    idle, system, user = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    if not (kernel.GetSystemPowerStatus(ctypes.byref(power))
            and kernel.GlobalMemoryStatusEx(ctypes.byref(memory))
            and kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user))):
        raise ctypes.WinError(ctypes.get_last_error())
    profile = ctypes.WinDLL('powrprof')
    overlay = (ctypes.c_ubyte * 16)()
    pointer = ctypes.c_void_p()
    if profile.PowerGetEffectiveOverlayScheme(ctypes.byref(overlay)):
        raise OSError('Effective overlay unavailable')
    if profile.PowerGetActiveScheme(None, ctypes.byref(pointer)):
        raise OSError('Active power scheme unavailable')
    try:
        scheme = str(uuid.UUID(bytes_le=ctypes.string_at(pointer, 16)))
    finally:
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(pointer)
    return dict(ac=power.ac == 1, ac_raw=power.ac, battery_percent=power.percent,
                scheme=scheme, overlay=str(uuid.UUID(bytes_le=bytes(overlay))),
                ram_available_bytes=memory.available, ram_total_bytes=memory.total,
                cpu_ticks=dict(idle=idle.value, total=system.value + user.value))


def process_deltas(rows, previous, elapsed, cpus):
    return [dict(pid=r['Id'], name=r['ProcessName'], ram_bytes=r['WorkingSet64'],
                 cpu_percent=(max(0, r['CPU'] - previous[r['Id']]) / elapsed / cpus * 100
                              if r.get('CPU') is not None and r['Id'] in previous and elapsed > 0 else None))
            for r in rows]


def windows_processes():
    """Toolhelp + kernel getters, avoiding a PowerShell process every five seconds."""
    class Entry(ctypes.Structure):
        _fields_ = [('size', ctypes.c_uint32), ('usage', ctypes.c_uint32), ('pid', ctypes.c_uint32),
                    ('heap', ctypes.c_size_t), ('module', ctypes.c_uint32), ('threads', ctypes.c_uint32),
                    ('parent', ctypes.c_uint32), ('priority', ctypes.c_int32), ('flags', ctypes.c_uint32),
                    ('name', ctypes.c_wchar * 260)]

    class Memory(ctypes.Structure):
        _fields_ = [('size', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in
            ('peak_working', 'working', 'peak_pool', 'pool', 'peak_nonpaged', 'nonpaged', 'page', 'peak_page')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(Entry)]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    kernel.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory), ctypes.c_uint32]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        present = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not present:
            raise ctypes.WinError(ctypes.get_last_error())
        while present:
            row = dict(Id=entry.pid, ProcessName=entry.name.removesuffix('.exe'), CPU=None, WorkingSet64=None)
            handle = kernel.OpenProcess(0x1000, False, entry.pid)  # query only
            if handle:
                try:
                    created, exited, system, user = (ctypes.c_uint64() for _ in range(4))
                    if kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                              ctypes.byref(system), ctypes.byref(user)):
                        row['CPU'] = (system.value + user.value) / 10_000_000
                    memory = Memory()
                    memory.size = ctypes.sizeof(memory)
                    if kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.size):
                        row['WorkingSet64'] = memory.working
                finally:
                    kernel.CloseHandle(handle)
            rows.append(row)
            present = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(snapshot)
    return rows


class Sampler:
    def __init__(self, host=None):
        self.host = host or os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        # Observation only: avoid Windows IPv6 fallback on this IPv4-bound local service.
        # The measured client keeps the original host and HTTP behavior.
        self.host = self.host.replace('://localhost:', '://127.0.0.1:')
        self.previous = None
        self.process_times = {}
        self.process_at = None

    def __call__(self):
        started = time.monotonic()
        row = dict(at=datetime.now(timezone.utc).isoformat(), monotonic_s=started,
                   observer_pid=os.getpid(), errors=[], cpu_temperature='unavailable: no authorized accessible sensor')
        try:
            state = windows_state()
            ticks = state['cpu_ticks']
            delta = ticks['total'] - self.previous['total'] if self.previous else 0
            row.update(state, cpu_percent=(100 * (1 - (ticks['idle'] - self.previous['idle']) / delta)
                                            if delta > 0 else None))
            self.previous = ticks
        except Exception as exc:
            row['errors'].append(f'windows: {type(exc).__name__}: {exc}')
        try:
            values = command(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
                              '--format=csv,noheader,nounits']).splitlines()
            if len(values) != 1:
                raise ValueError('Expected one reference GPU')
            row['gpu'] = dict(zip(GPU_FIELDS, (v.strip() for v in values[0].split(',')), strict=True))
            float(row['gpu']['utilization.gpu'])  # mandatory; optional unsupported sensors remain N/A
        except Exception as exc:
            row['errors'].append(f'gpu: {type(exc).__name__}: {exc}')
        try:
            rows = windows_processes()
            at = time.monotonic()
            row['processes'] = process_deltas(rows, self.process_times,
                                               at - self.process_at if self.process_at else 0, os.cpu_count())
            row['collector_pid'] = os.getpid()
            self.process_times = {r['Id']: r['CPU'] for r in rows if r.get('CPU') is not None}
            self.process_at = at
        except Exception as exc:
            row['errors'].append(f'processes: {type(exc).__name__}: {exc}')
        try:
            row['observer_ollama_host'] = self.host
            with urllib.request.urlopen(self.host.rstrip('/') + '/api/ps', timeout=3) as response:
                row['ollama_ps_api'] = json.load(response)
            executable = Path(os.environ['LOCALAPPDATA']) / 'Programs/Ollama/ollama.exe'
            row['ollama_ps_cli'] = command([str(executable), 'ps'])
        except Exception as exc:
            row['errors'].append(f'ollama_ps: {type(exc).__name__}: {exc}')
        row['observation_duration_s'] = time.monotonic() - started
        return row


def assess(rows, admission=False, allowed_pids=()):
    reasons = set()
    if not rows:
        return ['telemetry_missing']
    previous = None
    busy_previous = set()
    for index, row in enumerate(rows):
        if row.get('errors'):
            reasons.add('telemetry_error')
        if row.get('ac') is not True:
            reasons.add('ac_unavailable')
        if row.get('scheme') != BALANCED or row.get('overlay') != BEST_PERFORMANCE:
            reasons.add('power_mode_changed')
        if 'gpu' not in row or 'ram_available_bytes' not in row or 'processes' not in row:
            reasons.add('telemetry_missing')
        if index and row.get('cpu_percent') is None:
            reasons.add('cpu_telemetry_missing')
        if any(m.get('name', m.get('model')) != 'granite4.1:8b'
               for m in row.get('ollama_ps_api', {}).get('models', [])):
            reasons.add('concurrent_model')
        if previous is not None and row['monotonic_s'] - previous > 15:
            reasons.add('telemetry_gap')
        previous = row['monotonic_s']
        busy = set()
        for process in row.get('processes', []):
            name = process['name'].lower()
            if name in ('brave', 'chrome', 'msedge', 'firefox', 'opera', 'epicgameslauncher', 'steam') or 'overlay' in name:
                reasons.add('prohibited_process')
            exempt = (process['pid'] in set(allowed_pids) | {row.get('collector_pid'), row.get('observer_pid')}
                      or name in ('idle', 'system', 'registry', 'memory compression', 'ollama', 'ollama app'))
            if not exempt and (process.get('cpu_percent') or 0) >= 10:
                busy.add(process['pid'])
        if busy & busy_previous:
            reasons.add('external_cpu_load')
        busy_previous = busy
    if admission:
        if rows[-1]['monotonic_s'] - rows[0]['monotonic_s'] < 60:
            reasons.add('idle_window_incomplete')
        cpu = [r['cpu_percent'] for r in rows if r.get('cpu_percent') is not None]
        try:
            gpu = [float(r['gpu']['utilization.gpu']) for r in rows]
        except (KeyError, ValueError):
            gpu = []
        if len(cpu) < len(rows) - 1 or not cpu:
            reasons.add('cpu_telemetry_missing')
        elif sum(cpu) / len(cpu) >= 10:
            reasons.add('idle_cpu')
        if not gpu or sum(gpu) / len(gpu) >= 10:
            reasons.add('idle_gpu')
    return sorted(reasons)


class Observer:
    def __init__(self, path, sampler=None, interval=5, allowed_pids=()):
        self.path = Path(path)
        self.sampler = sampler or Sampler()
        self.interval = interval
        self.allowed_pids = allowed_pids
        self.rows = []
        self.stop_event = threading.Event()
        self.persistence_errors = []

    def sample(self):
        try:
            row = self.sampler()
        except Exception as exc:
            row = dict(at=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(),
                       errors=[f'{type(exc).__name__}: {exc}'])
        self.rows.append(row)
        try:
            with self.path.open('ab') as stream:
                stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            self.persistence_errors.append(f'{type(exc).__name__}: {exc}')

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x'):
            pass
        self.sample()  # before response clock; primes delta counters
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        deadline = time.monotonic() + self.interval
        while not self.stop_event.wait(max(0, deadline - time.monotonic())):
            self.sample()
            deadline += self.interval
            if deadline < time.monotonic():
                deadline = time.monotonic() + self.interval  # no burst of delayed subprocesses

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()  # after response clock; captures last power state
        reasons = assess(self.rows, allowed_pids=self.allowed_pids)
        if self.persistence_errors:
            reasons.append('telemetry_persistence_error')
        return dict(conditions_invalid=bool(reasons), control_reasons=reasons,
                    telemetry_path=str(self.path), telemetry_samples=len(self.rows),
                    observer_errors=self.persistence_errors)


def admission(root):
    from scripts.measure_interview_gate import write_new
    root = Path(root)
    observer = Observer(root / 'samples.jsonl', allowed_pids={os.getpid(), os.getppid()})
    observer.start()
    # No inference; the final sample must span at least 60 seconds of idle observation.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    result = observer.finish()
    reasons = assess(observer.rows, admission=True, allowed_pids=observer.allowed_pids)
    reasons += result['observer_errors']
    write_new(root / 'result.json', dict(result, admitted=not reasons, admission_reasons=reasons))
    if reasons:
        raise RuntimeError('Controlled admission failed: ' + ', '.join(reasons))
