"""Recover the installed SDK's Plink argv without mediating private stdin."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path


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
