"""Decode a limited WPR hard-fault capture without equating all faults with disk faults."""
import base64
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import uuid
import xml.etree.ElementTree as ET

NS = {'e': 'http://schemas.microsoft.com/win/2004/08/events/event',
      't': 'http://schemas.microsoft.com/win/2004/08/events/trace'}
THREAD = '{3d6fa8d1-fe05-11d0-9dda-00c04fd7ba7c}'
PROCESS = '{3d6fa8d0-fe05-11d0-9dda-00c04fd7ba7c}'
FAULT = '{3d6fa8d3-fe05-11d0-9dda-00c04fd7ba7c}'
FILE = '{90cbdc39-4a3e-11d1-84f4-0000f80464e3}'


def number(value):
    value = value.strip()
    return int(value, 16 if value.lower().startswith('0x') else 10)


def parse_trace(path):
    threads, files, processes, faults = {}, {}, {}, []
    losses, buffer_losses, decode_errors = [], [], {}
    prefix_decoded = 0
    context = ET.iterparse(Path(path), events=('start', 'end'))
    _, root = next(context)
    for event, element in context:
        if event != 'end' or element.tag != '{' + NS['e'] + '}Event':
            continue
        system = element.find('e:System', NS)
        guid = element.findtext('t:ExtendedTracingInfo/t:EventGuid', '', NS).lower()
        opcode = number(system.findtext('e:Opcode', '0', NS))
        version = number(system.findtext('e:Version', '0', NS))
        at = system.find('e:TimeCreated', NS).attrib['SystemTime']
        data = {d.attrib['Name']: (d.text or '').strip() for d in element.findall('e:EventData/e:Data', NS)}
        if 'EventsLost' in data:
            losses.append(number(data['EventsLost']))
        if 'BuffersLost' in data:
            buffer_losses.append(number(data['BuffersLost']))
        error = element.find('e:ProcessingErrorData', NS)
        if error is not None:
            decode_errors[guid] = decode_errors.get(guid, 0) + 1
            if guid == FAULT and opcode == 32:
                raise ValueError('Hard-fault event could not be decoded')
            if guid == THREAD and version == 3 and opcode in (1, 2, 3, 4):
                # Windows 11 tracerpt may fail later fields of Thread v3. Only the documented
                # uint32 ProcessId/TThreadId prefix is read; no stack/priority fields are inferred.
                # https://learn.microsoft.com/en-us/windows/win32/etw/thread-v2-typegroup1
                raw = bytes.fromhex(error.findtext('e:EventPayload', '', NS))
                if len(raw) < 8:
                    raise ValueError('Truncated thread identity prefix')
                pid, tid = struct.unpack_from('<II', raw)
                data.update(ProcessId=str(pid), TThreadId=str(tid))
                prefix_decoded += 1
        if guid == PROCESS and 'ProcessId' in data:
            pid = number(data['ProcessId'])
            if opcode in (1, 3, 4):
                processes[pid] = data.get('ImageFileName')
            elif opcode == 2:
                processes.pop(pid, None)
        elif guid == THREAD and {'ProcessId', 'TThreadId'} <= data.keys():
            tid = number(data['TThreadId'])
            if opcode in (1, 3, 4):
                threads[tid] = number(data['ProcessId'])
            elif opcode == 2:
                threads.pop(tid, None)
        elif guid == FILE and 'FileObject' in data:
            pointer = number(data['FileObject'])
            if 'FileName' in data:
                files[pointer] = data['FileName']
            elif opcode == 35:  # FileDelete
                files.pop(pointer, None)
        elif guid == FAULT and opcode == 32:
            tid = number(data['TThreadId'])
            pid = threads.get(tid)
            faults.append(dict(at=at, timestamp_s=datetime.fromisoformat(at).timestamp(),
                tid=tid, pid=pid, process=processes.get(pid), bytes_read=number(data['ByteCount']),
                file=files.get(number(data['FileObject'])), initial_filetime=number(data['InitialTime'])))
        root.clear()
    if not losses or not buffer_losses:
        raise ValueError('Missing ETW loss metadata; cannot claim complete capture')
    if any(losses) or any(buffer_losses):
        raise ValueError('ETW events/buffers lost; capture is incomplete')
    return dict(events_lost=sum(losses), buffers_lost=sum(buffer_losses), hard_faults=faults,
                hard_fault_count=len(faults), unattributed=sum(r['pid'] is None for r in faults),
                decoding_errors=decode_errors, thread_identity_prefix_decoded=prefix_decoded)


def intervals(faults, start, elapsed, width=5):
    if not math.isfinite(elapsed) or elapsed <= 0 or width <= 0:
        raise ValueError('Invalid observation interval')
    origin = datetime.fromisoformat(start).timestamp()
    rows = [dict(offset_s=i * width, duration_s=min(width, elapsed - i * width),
                 count=0, bytes_read=0, by_pid={}) for i in range(math.ceil(elapsed / width))]
    for fault in faults:
        offset = fault['timestamp_s'] - origin
        if 0 <= offset < elapsed:
            row = rows[min(int(offset // width), len(rows) - 1)]
            row['count'] += 1
            row['bytes_read'] += fault['bytes_read']
            key = str(fault['pid']) if fault['pid'] is not None else 'unknown'
            row['by_pid'][key] = row['by_pid'].get(key, 0) + 1
    for row in rows:
        row['hard_faults_per_s'] = row['count'] / row['duration_s']
    return rows


class Capture:
    """Unique WPR instance; no global cancel or stop of another recorder."""
    def __init__(self, root):
        self.root = Path(root)
        self.instance = 'CloudRAG-' + uuid.uuid4().hex
        self.active = False

    def command(self, args, timeout=90):
        from scripts.measure_interview_gate import now, write_new
        result = subprocess.run(args, capture_output=True, timeout=timeout,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        write_new(self.root / f'command-{uuid.uuid4().hex}.json',
                  dict(at=now(), args=args, returncode=result.returncode,
                       stdout_base64=base64.b64encode(result.stdout).decode('ascii'),
                       stderr_base64=base64.b64encode(result.stderr).decode('ascii'),
                       preview_encoding='utf-8 with backslashreplace; raw bytes retained above',
                       stdout=result.stdout.decode('utf-8', errors='backslashreplace'),
                       stderr=result.stderr.decode('utf-8', errors='backslashreplace')))
        if result.returncode:
            raise RuntimeError(f'Trace command failed: {args[0]} ({result.returncode})')
        return result.stdout

    def start(self):
        from scripts.gate_memory import PROFILE
        from scripts.measure_interview_gate import digest, now, write_new
        self.root.mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(self.root).free < 5 * 1024 ** 3:
            raise RuntimeError('At least 5 GiB free required for bounded trace evidence')
        write_new(self.root / 'trace-identity.json', dict(at=now(), instance=self.instance,
            owner_pid=os.getpid(), profile_sha256=digest(PROFILE), decoder_sha256=digest(__file__)))
        self.command(['wpr', '-start', f'{PROFILE}!GateMemory', '-filemode', '-recordtempto', str(self.root),
                      '-instancename', self.instance])
        self.active = True
        write_new(self.root / 'trace-active.json', dict(at=now(), instance=self.instance))

    def finish(self, start, elapsed):
        from scripts.measure_interview_gate import digest, now, write_new
        if not self.active:
            raise RuntimeError('Trace did not start')
        destination = self.root / 'trace.etl'
        if destination.exists():
            raise FileExistsError(destination)
        self.command(['wpr', '-stop', str(destination), '-skipPdbGen', '-instancename', self.instance])
        self.active = False
        write_new(self.root / 'trace-stopped.json', dict(at=now(), instance=self.instance))
        self.command(['tracerpt', str(destination), '-of', 'XML', '-o', str(self.root / 'trace.xml'),
                      '-summary', str(self.root / 'summary.txt')])
        decoded = parse_trace(self.root / 'trace.xml')
        decoded['response_intervals'] = intervals(decoded['hard_faults'], start, elapsed)
        path = self.root / 'decoded.json'
        write_new(path, dict(at=now(), trace_sha256=digest(destination), **decoded))
        return dict(hard_fault_evidence=str(path), hard_fault_evidence_sha256=digest(path),
                    hard_fault_count_in_response=sum(r['count'] for r in decoded['response_intervals']))


def smoke(root):
    """Five-second real capture lifecycle check, no inference or process configuration."""
    import time
    from scripts.measure_interview_gate import now, write_new
    capture = Capture(root)
    capture.start()
    started_at, started = now(), time.perf_counter()
    time.sleep(5)
    result = capture.finish(started_at, time.perf_counter() - started)
    write_new(Path(root) / 'smoke-result.json', dict(at=now(), **result))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', required=True, type=Path)
    smoke(parser.parse_args().smoke)
