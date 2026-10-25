"""Windows system commit and paging counters; totals are not hard-fault counts."""
import ctypes
import mmap
import os
from pathlib import Path
import weakref

PROFILE = Path(__file__).with_name('gate_memory.wprp')


def commit_metrics(total, limit, physical, available, page_size):
    if min(total, physical, available) < 0 or limit <= 0 or page_size <= 0:
        raise ValueError('Invalid memory counters')
    return dict(committed_bytes=total * page_size, commit_limit_bytes=limit * page_size,
                commit_fraction=total / limit, ram_total_bytes=physical * page_size,
                ram_available_bytes=available * page_size)


def system_memory():
    class Performance(ctypes.Structure):
        _fields_ = [('size', ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in
            ('commit', 'limit', 'peak', 'physical', 'available', 'cache', 'kernel', 'paged', 'nonpaged', 'page_size')]
        _fields_ += [(name, ctypes.c_uint32) for name in ('handles', 'processes', 'threads')]

    info = Performance()
    info.size = ctypes.sizeof(info)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = [ctypes.POINTER(Performance), ctypes.c_uint32]
    if not api.GetPerformanceInfo(ctypes.byref(info), info.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return commit_metrics(info.commit, info.limit, info.physical, info.available, info.page_size)


class PagingCounters:
    """PDH uses English counter paths, independent of Windows display language."""
    PATHS = {'page_faults_total_per_s': r'\Memory\Page Faults/sec',
             'page_read_operations_per_s': r'\Memory\Page Reads/sec',
             'pages_input_per_s': r'\Memory\Pages Input/sec'}

    def __init__(self):
        self.api = ctypes.WinDLL('pdh')
        self.api.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.api.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t,
                                                  ctypes.POINTER(ctypes.c_void_p)]
        self.api.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.api.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self.query = ctypes.c_void_p()
        self.check(self.api.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        self.finalizer = weakref.finalize(self, self.api.PdhCloseQuery, self.query)
        self.counters = {}
        for name, path in self.PATHS.items():
            handle = ctypes.c_void_p()
            self.check(self.api.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)))
            self.counters[name] = handle
        self.primed = False

    @staticmethod
    def check(status):
        if status:
            raise OSError(f'PDH status 0x{status & 0xffffffff:08x}')

    def sample(self):
        class Value(ctypes.Structure):
            _fields_ = [('status', ctypes.c_uint32), ('value', ctypes.c_double)]

        self.check(self.api.PdhCollectQueryData(self.query))
        if not self.primed:
            self.primed = True
            return dict.fromkeys(self.counters)
        self.api.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                        ctypes.c_void_p, ctypes.POINTER(Value)]
        result = {}
        for name, handle in self.counters.items():
            value = Value()
            self.check(self.api.PdhGetFormattedCounterValue(handle, 0x200, None, ctypes.byref(value)))
            if value.status not in (0, 1):
                raise OSError(f'Invalid PDH counter {name}: {value.status}')
            result[name] = value.value
        return result


def synthetic_hard_faults(path):
    """Create an uncached 16 MiB zero file, then map/read it. No global cache trimming."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel.VirtualAlloc.restype = ctypes.c_void_p
    kernel.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                 ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    size = 16 * 1024 * 1024
    handle = kernel.CreateFileW(str(path), 0x40000000, 1, None, 1, 0xa0000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = None
    try:
        buffer = kernel.VirtualAlloc(None, size, 0x3000, 4)
        if not buffer:
            raise ctypes.WinError(ctypes.get_last_error())
        written = ctypes.c_uint32()
        if not kernel.WriteFile(handle, buffer, size, ctypes.byref(written), None) or written.value != size:
            raise OSError('Uncached synthetic write failed')
    finally:
        if buffer:
            kernel.VirtualFree(buffer, 0, 0x8000)
        kernel.CloseHandle(handle)
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        checksum = sum(mapped[i] for i in range(0, size, 4096))
    return dict(pid=os.getpid(), bytes=size, checksum=checksum, path=str(path))
