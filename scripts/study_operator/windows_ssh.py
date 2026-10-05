"""Recover the installed SDK's Plink argv without mediating private stdin."""
import base64
import ctypes
from ctypes import wintypes
import os
import hashlib
from pathlib import Path


def api_host_key_flags(value):
    """Pin public host keys returned by the authenticated Google API."""
    rows = value['queryValue']['items']
    flags, seen = [], set()
    for row in rows:
        if row.get('namespace') != 'hostkeys':
            continue
        algorithm = row['key']
        if algorithm not in {'ssh-rsa', 'ssh-ed25519', 'ecdsa-sha2-nistp256'}:
            raise ValueError('Unsupported API host key algorithm')
        if algorithm in seen:
            raise ValueError('Duplicate API host key algorithm')
        encoded = row['value']
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
