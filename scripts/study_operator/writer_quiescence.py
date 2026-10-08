"""Inspect native Python writers without exposing command lines or stopping apps."""
import argparse
import ctypes
import json
import os
from pathlib import Path, PureWindowsPath
import sys

from scripts.study_operator.run_control import require_limited


def normalized(value):
    path = PureWindowsPath(value)
    return path.as_posix().casefold() if path.is_absolute() else Path(value).resolve().as_posix()


def binding():
    native = sys.executable
    if os.name == 'nt':
        buffer = ctypes.create_unicode_buffer(32768)
        count = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
        if not count or count >= len(buffer):
            raise RuntimeError('Native executable observation failed')
        native = buffer.value
    return dict(native_executable=str(Path(native).resolve()), launcher=str(Path(sys.executable).resolve()))


def coordinator_exclusions(rows, coordinator_pid, native, launcher, plan, *, parse):
    """Exclude the bound coordinator and its waiting venv redirector only."""
    observed = {row['ProcessId']: row for row in rows}
    current = observed.get(coordinator_pid)

    def invocation(row):
        argv = parse(row.get('CommandLine') or '')
        return (len(argv) == 6 and argv[1:5] ==
                ['-B', '-m', 'scripts.study_operator.finalization', '--plan']
                and normalized(argv[5]) == normalized(plan))

    if (not current or not current.get('ExecutablePath')
            or normalized(current['ExecutablePath']) != normalized(native) or not invocation(current)):
        raise ValueError('Coordinator executable and entry plan not proven')
    excluded = {coordinator_pid}
    parent = observed.get(current.get('ParentProcessId'))
    if parent and parent.get('ExecutablePath') and normalized(parent['ExecutablePath']) == normalized(launcher):
        if not invocation(parent):
            raise ValueError('Venv redirector invocation differs from coordinator')
        excluded.add(parent['ProcessId'])
    return excluded


def writer_pids(rows, executables, scopes, coordinator_pid, *, coordinator_binding=None, parse=None):
    if type(coordinator_pid) is not int or coordinator_pid <= 0 or not executables or not scopes:
        raise ValueError('Explicit native bindings, scopes and positive coordinator PID required')
    allowed = {normalized(p) for p in executables}
    scopes = [str(s).replace('\\','/').casefold() for s in scopes]
    selected, seen = [], set()
    for row in rows:
        pid = row['ProcessId']
        if type(pid) is not int or pid <= 0 or pid in seen:
            raise ValueError('Invalid or duplicate process identity')
        seen.add(pid)
    excluded = {coordinator_pid}
    if coordinator_binding is not None:
        excluded = coordinator_exclusions(rows, coordinator_pid, **coordinator_binding, parse=parse)
    for row in rows:
        pid = row['ProcessId']
        command = (row.get('CommandLine') or '').replace('\\','/').casefold()
        executable = row.get('ExecutablePath')
        if executable and normalized(executable) in allowed and any(s in command for s in scopes):
            if pid not in excluded:
                selected.append(pid)
    return dict(writer_pids=sorted(selected), command_lines_not_persisted=True,
                processes_not_stopped=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', action='store_true')
    parser.add_argument('--native')
    parser.add_argument('--launcher')
    parser.add_argument('--sdk-python')
    parser.add_argument('--package')
    parser.add_argument('--owner')
    parser.add_argument('--coordinator-pid', type=int)
    parser.add_argument('--coordinator-plan')
    args = parser.parse_args(argv)
    require_limited()
    if args.binding:
        result = binding()
    else:
        if not all((args.native,args.launcher,args.sdk_python,args.package,args.owner,args.coordinator_plan)):
            raise ValueError('All own executable and package bindings required')
        rows = json.load(sys.stdin)
        from scripts.study_operator.windows_ssh import windows_argv

        result = writer_pids(rows, [args.native,args.launcher,args.sdk_python],
            [args.package,args.owner,args.launcher,'--project=pure-loop-474323-a8'],args.coordinator_pid,
            coordinator_binding=dict(native=args.native,launcher=args.launcher,plan=args.coordinator_plan),
            parse=windows_argv)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
