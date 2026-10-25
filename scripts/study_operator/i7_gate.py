"""Fixed I7 pilot/cohort accounting. No inference from a subset or pilot."""
import math

from scripts.study_operator.stimulus_calendar import TASKS
from scripts.study_operator.stimulus_host_evidence import sha

HOST_LIMIT_S = 9000
VM_LIMIT_S = 10800


def calendar(tasks):
    if tuple(tasks) != TASKS:
        raise ValueError('SEALED_GATE_TASKS_REQUIRED')
    windows = []
    for window in (1, 2):
        for repetition in range(1 if window == 1 else 6, 6 if window == 1 else 11):
            for pos, query in enumerate(tasks):
                conditions = ('hybrid', 'no_rag') if (repetition+pos) % 2 else ('no_rag', 'hybrid')
                windows.extend({'phase': 'window'+str(window), 'window': window, 'repetition': repetition,
                                    'query_id': query, 'condition': c} for c in conditions)
    return [dict(slot, phase='pilot') for slot in windows[:20]]+windows


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    left = math.floor(position)
    return values[left]+(values[math.ceil(position)]-values[left])*(position-left)


def summarize(rows):
    result = {}
    for condition in ('hybrid', 'no_rag'):
        group = [r for r in rows if r['condition'] == condition]
        values = [r['elapsed_s'] for r in group if r['status'] == 'success' and r['valid']]
        result[condition] = {'n': len(values), 'failures': sum(r['status'] != 'success' for r in group),
            'invalid': sum(not r['valid'] for r in group), 'p50': percentile(values, .5), 'p95': percentile(values, .95)}
    return result


class GateRun:
    def __init__(self, tasks, identity):
        if not identity or not identity.get('boot_id') or not identity.get('image_id'):
            raise ValueError('GATE_IDENTITY_REQUIRED')
        self.slots, self.identity = calendar(tasks), identity
        self.rows, self.calls, self.phase_starts = [], [], {}
        self.terminal = False

    def record(self, index, row, *, started, ended):
        if self.terminal or index != len(self.rows)+1 or not 1 <= index <= 140:
            self.terminal = True
            raise ValueError('TERMINAL_OR_UNORDERED_GATE')
        slot = self.slots[index-1]
        phase = slot['phase']
        self.phase_starts.setdefault(phase, started)
        good_time = (all(type(t) in (int, float) and math.isfinite(t) for t in (started, ended))
                     and 0 <= ended-started <= 600
                     and (not self.calls or started >= self.calls[-1]['ended'])
                     and (phase == 'pilot' or ended-self.phase_starts[phase] <= 7200))
        duration = row.get('elapsed_s')
        good_row = (all(row.get(k) == v for k, v in slot.items()) and row.get('status') == 'success'
            and row.get('valid') is True and type(duration) in (int, float)
            and math.isfinite(duration) and 0 <= duration <= ended-started
            and ('identity' not in row or row['identity'] == self.identity))
        self.rows.append(dict(row))
        self.calls.append({'index': index, 'started': started, 'ended': ended, 'row_sha256': sha(row)})
        if not good_time or not good_row:
            self.terminal = True
            raise ValueError('GATE_ATTEMPT_TERMINAL')

    def decision(self):
        systems = summarize(self.rows[20:]) if len(self.rows) > 20 else summarize([])
        complete = len(self.rows) == 140 and not self.terminal
        passed = complete and all(g['n'] == 60 and g['failures'] == 0 and g['invalid'] == 0
                                  and g['p95'] <= 60 for g in systems.values())
        return {'decision': 'GO' if passed else 'NO-GO' if complete else
                    'INCOMPLETE_TERMINAL' if self.terminal else 'INCOMPLETE',
                    'pilot_count': min(20, len(self.rows)), 'gate_count': max(0, len(self.rows)-20), 'systems': systems}


def collect(channel, host, run):
    """Parent owns phase deadlines, telemetry, private rows and failure accounting."""
    from scripts.study_operator.service_gateway import save_state
    phase = None
    deadline = None
    try:
        for index, slot in enumerate(run.slots, 1):
            if slot['phase'] != phase:
                phase = slot['phase']
                if phase!='pilot' and hasattr(host,'revalidate'):
                    host.revalidate(channel,index)
                deadline = host.clock()+(12000 if phase == 'pilot' else 7200)
            host.poll(force=True)
            start = host.clock()
            if start >= deadline:
                raise TimeoutError('GATE_PHASE_DEADLINE')
            channel.send({'index': index})
            try:
                response = channel.receive(min(600, deadline-start))
            except BaseException:
                row = dict(slot, index=index, status='aborted', valid=False, elapsed_s=None)
                if host.root is not None:
                    save_state(host.root/'coded-P999'/f'{index:03d}.json', row)
                try:
                    run.record(index, row, started=start, ended=host.clock())
                except ValueError:
                    pass  # The terminal attempt was retained before validation.
                raise
            row = response.get('row', dict(slot, status='failed', valid=False, elapsed_s=None))
            if response.get('status') != 'OBSERVATION' or row.get('index') != index:
                row = dict(row, status='failed', valid=False)
            end = host.clock()
            # Save before validation; a rejected row is still an attempted query.
            if host.root is not None:
                save_state(host.root/'coded-P999'/f'{index:03d}.json', row)
            run.record(index, row, started=start, ended=end)
            if end > deadline:
                raise TimeoutError('GATE_PHASE_DEADLINE')
            host.poll(force=True)
        channel.send({'operation': 'complete'})
        final = channel.receive(30)
        if final != {'status': 'GATE_COMPLETE', 'rows_sha256': sha(run.rows)}:
            raise ValueError('GATE_FINAL_RECEIPT_DIFFERS')
        return run.decision()
    except BaseException:
        run.terminal = True
        raise
