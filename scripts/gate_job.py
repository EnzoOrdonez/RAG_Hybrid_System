"""Own the measurement process tree with a Windows kill-on-close job."""
import ctypes
import os

_HANDLE = None


def enter():
    global _HANDLE
    if _HANDLE is not None:
        raise RuntimeError('Measurement job already established')

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', ctypes.c_uint32), ('minimum', ctypes.c_size_t), ('maximum', ctypes.c_size_t),
                    ('active', ctypes.c_uint32), ('affinity', ctypes.c_size_t),
                    ('priority', ctypes.c_uint32), ('scheduling', ctypes.c_uint32)]

    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    process = ctypes.c_void_p(-1)
    created, exited, system, user = (ctypes.c_uint64() for _ in range(4))
    try:
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetProcessTimes(process, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(handle, process):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    # Keep the sole non-inherited handle until process exit, including abnormal death.
    _HANDLE = handle
    return dict(pid=os.getpid(), creation_filetime=created.value)
