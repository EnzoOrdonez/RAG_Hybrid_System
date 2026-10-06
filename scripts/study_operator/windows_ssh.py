"""Validate SDK display and use pinned native OpenSSH for private RPC."""
import base64
import ctypes
from ctypes import wintypes
import os
import hashlib
from pathlib import Path
import re
import subprocess


def api_host_key_flags(value):
    """Pin public host keys returned by the authenticated Google API."""
    # gcloud's get-guest-attributes command flattens REST queryValue.items.
    if not isinstance(value, list):
        raise ValueError('SDK guest attributes must be a row list')
    rows = value
    flags, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid SDK guest attribute row')
        if row.get('namespace') != 'hostkeys':
            continue
        algorithm = row.get('key')
        if algorithm not in {'ssh-rsa', 'ssh-ed25519', 'ecdsa-sha2-nistp256'}:
            raise ValueError('Unsupported API host key algorithm')
        if algorithm in seen:
            raise ValueError('Duplicate API host key algorithm')
        encoded = row.get('value')
        if not isinstance(encoded, str) or len(encoded) > 16384:
            raise ValueError('Invalid API host key')
        decoded = base64.b64decode(encoded, validate=True)
        prefix = len(algorithm).to_bytes(4, 'big') + algorithm.encode()
        if (not decoded.startswith(prefix) or len(decoded) <= len(prefix) + 4
                or base64.b64encode(decoded).decode() != encoded):
            raise ValueError('API host key wire algorithm differs')
        seen.add(algorithm)
        fingerprint = base64.b64encode(hashlib.sha256(decoded).digest()).decode().rstrip('=')
        flags.extend(['--ssh-flag=-hostkey', '--ssh-flag=SHA256:' + fingerprint])
    if not flags:
        raise ValueError('No authenticated public host keys')
    return flags


def windows_argv(text):
    if os.name != 'nt' or not text.strip() or '\n' in text.strip() or '\r' in text.strip():
        raise ValueError('Windows SDK command must contain one nonempty line')
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(text.strip(), ctypes.byref(count))
    if not pointer:
        raise OSError(ctypes.get_last_error(), 'Windows rejected the SDK command')
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))


def sdk_argv(content, expected_plink):
    text = content.decode('utf-8-sig').strip()
    if '\n' in text or '\r' in text:
        raise ValueError('SDK command contains multiple lines')
    # SDK dry-run surrounds space-containing fields with unescaped quotes.
    # Its nested IAP proxy field is therefore not a Windows command line.
    if ' -proxycmd ' in text:
        prefix, rest = text.split(' -proxycmd ', 1)
        if not rest.startswith('"') or rest.count('" -batch ') != 1:
            raise ValueError('IAP proxy boundary is ambiguous')
        boundary = rest.index('" -batch ')
        argv = windows_argv(prefix) + ['-proxycmd', rest[1:boundary]] + windows_argv(rest[boundary+2:])
        displayed = ' '.join('"'+arg+'"' if ' ' in arg else arg for arg in argv)
        if displayed != text:
            raise ValueError('SDK display failed its exact inverse')
    else:
        argv = windows_argv(text)
    if Path(argv[0]).resolve() != Path(expected_plink).resolve() or '-batch' not in argv:
        raise ValueError('SDK executable or host-key policy differs')
    return argv


def native_argv(plink, rows, *, sdk, target, known_hosts, ssh=None, key=None):
    """Translate only the reviewed IAP RPC; no ambient SSH configuration."""
    if (target['zone'] not in {'us-central1-a', 'us-central1-b', 'us-central1-c'}
            or not re.fullmatch(r'cloudrag-[a-z0-9-]+', target['name'])
            or not re.fullmatch(r'[0-9]+', str(target['id']))):
        raise ValueError('Managed IAP target required')
    if (Path(plink[0]).resolve() != (Path(sdk).parent/'sdk/plink.exe').resolve()
            or plink.count('-proxycmd') != 1 or '-batch' not in plink or '-T' not in plink):
        raise ValueError('Unexpected SDK IAP executable or flags')
    pins = [plink[i+1] for i, value in enumerate(plink[:-1]) if value == '-hostkey']
    expected = [flag.removeprefix('--ssh-flag=') for flag in api_host_key_flags(rows)
                if flag.startswith('--ssh-flag=SHA256:')]
    if pins != expected:
        raise ValueError('SDK host keys differ from authenticated API')
    aliases = [i for i, value in enumerate(plink)
               if re.fullmatch(r'[a-z_][a-z0-9_-]*@compute\.[0-9]+', value)]
    if len(aliases) != 1:
        raise ValueError('Ambiguous managed SSH identity')
    position = aliases[0]
    user, alias = plink[position].split('@')
    remote = plink[position+1:]
    if alias != 'compute.'+str(target['id']) or remote[:2] != ['sudo', '-n']:
        raise ValueError('SDK VM identity or RPC command differs')
    proxy = plink[plink.index('-proxycmd')+1]
    if not all(value in proxy for value in ('start-iap-tunnel', target['name'],
            '--zone='+target['zone'], '--project=pure-loop-474323-a8')):
        raise ValueError('SDK IAP route differs')
    sdk_root = Path(sdk).parent.parent
    tunnel = subprocess.list2cmdline([str(sdk_root/'platform/bundledpython/python.exe'),
        '-S', str(sdk_root/'lib/gcloud.py'), 'compute', 'start-iap-tunnel', target['name'],
        '22', '--listen-on-stdin', '--project=pure-loop-474323-a8',
        '--zone='+target['zone'], '--verbosity=warning'])
    ssh = Path(ssh) if ssh else Path(os.environ.get('SystemRoot', 'C:/Windows'))/'System32/OpenSSH/ssh.exe'
    key = Path(key) if key else Path.home()/'.ssh/google_compute_engine'
    if not ssh.is_file() or not key.is_file():
        raise ValueError('Existing Windows OpenSSH or SDK private key missing')
    public = ''.join(alias+' '+row['key']+' '+row['value']+'\n'
                     for row in rows if row['namespace'] == 'hostkeys')
    with Path(known_hosts).open('x', encoding='ascii', newline='\n') as stream:
        stream.write(public)
    options = ['BatchMode=yes', 'StrictHostKeyChecking=yes',
        'UserKnownHostsFile='+str(known_hosts), 'GlobalKnownHostsFile=NUL',
        'HostKeyAlias='+alias, 'IdentitiesOnly=yes', 'PasswordAuthentication=no',
        'KbdInteractiveAuthentication=no', 'GSSAPIAuthentication=no', 'UpdateHostKeys=no',
        'CheckHostIP=no', 'ConnectTimeout=30', 'ProxyCommand='+tunnel]
    argv = [str(ssh), '-F', 'NUL', '-T', '-i', str(key)]
    for option in options:
        argv.extend(['-o', option])
    return [*argv, user+'@'+alias, *remote]


def native_run(argv, *, input, capture_output, timeout, env):
    """Bound the owned SSH child and its IAP descendants; private bytes stay RAM."""
    if capture_output is not True:
        raise ValueError('RPC capture policy required')
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        stdout, stderr = process.communicate(timeout=15)
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
