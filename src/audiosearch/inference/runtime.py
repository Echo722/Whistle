"""Runtime plumbing shared by the encoders: device choice, dtypes and metering.

Heavy imports (``torch``, ``transformers``) happen lazily inside functions so the
rest of the package — vectors, fusion, indexing — imports cleanly without a GPU
stack installed. Missing stacks raise ``ModelUnavailableError`` instead of
ImportError, so a service can report ``model_unavailable`` truthfully.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .spaces import ModelUnavailableError


def require_torch():
    """Import torch or raise a typed model-unavailable error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ModelUnavailableError(
            "PyTorch is not installed; install the inference extra before encoding"
        ) from exc
    return torch


def require_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ModelUnavailableError(
            "transformers is not installed; install the inference extra before encoding"
        ) from exc
    return transformers


@dataclass(frozen=True, slots=True)
class DevicePlan:
    """Where inference runs and in what precision."""

    device: str
    dtype: str
    reason: str

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def plan_device(preference: str = "auto") -> DevicePlan:
    """Resolve ``auto``/``cpu``/``cuda`` without pretending a GPU is present."""
    if preference == "cpu":
        return DevicePlan("cpu", "float32", "explicitly requested")
    torch = require_torch()
    if preference not in {"auto", "cuda"}:
        raise ValueError(f"unknown device preference: {preference}")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - broken driver
        if preference == "cuda":
            raise ModelUnavailableError(f"CUDA is unusable on this host: {exc}") from exc
        return DevicePlan("cpu", "float32", f"CUDA probe failed ({type(exc).__name__})")
    if available:
        return DevicePlan("cuda", "float32", "CUDA device available")
    if preference == "cuda":
        raise ModelUnavailableError("CUDA was requested but no GPU is available")
    return DevicePlan("cpu", "float32", "no CUDA device")


@dataclass
class ResourceMeter:
    """Peak memory and wall-clock bookkeeping for one encoding run."""

    label: str
    started: float = field(default_factory=time.perf_counter)
    gpu_peak_bytes: int | None = None
    _peak_extra: int = 0

    def reset_gpu_peak(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def sample_gpu(self) -> int | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated()
        self.gpu_peak_bytes = max(self.gpu_peak_bytes or 0, peak)
        return peak

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "gpu_peak_bytes": self.gpu_peak_bytes,
        }


@contextmanager
def peak_memory():
    """Process peak RSS (Windows/Linux) plus CUDA peak while the block runs."""
    start_rss = _rss_bytes()
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        end_rss = _rss_bytes()
        gpu_peak = None
        if torch is not None and torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated()
        _LAST_PEAK["rss_peak_bytes"] = max(start_rss, end_rss) if end_rss else start_rss
        _LAST_PEAK["gpu_peak_bytes"] = gpu_peak


_LAST_PEAK: dict[str, int | None] = {"rss_peak_bytes": None, "gpu_peak_bytes": None}


def last_peak() -> dict[str, int | None]:
    return dict(_LAST_PEAK)


def peak_rss_bytes() -> int | None:
    """Process peak resident set (Windows working set / POSIX ru_maxrss)."""
    return _rss_bytes()


def _rss_bytes() -> int | None:
    try:
        import resource  # POSIX only
    except ImportError:
        pass
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage) * (1024 if _is_linux() else 1)
    if _is_windows():
        return _windows_peak_rss()
    return None


def _is_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def _windows_peak_rss() -> int | None:
    """Peak working set via kernel32; no third-party dependency."""
    import ctypes
    import ctypes.wintypes as wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without an explicit restype the pseudo-handle -1 is truncated to 32 bits.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    process = kernel32.GetCurrentProcess()
    if not kernel32.K32GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def release_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "DevicePlan",
    "ResourceMeter",
    "last_peak",
    "peak_memory",
    "peak_rss_bytes",
    "plan_device",
    "release_memory",
    "require_torch",
    "require_transformers",
]
