"""Pin the verified version of a source file while REAPER is about to use it.

The catalog resolver checks size + mtime (the agreed on-disk version of an
``AssetLocation``) when the command is prepared, but the command may then wait
in the queue. To close that window the adapter opens the file once more and,
*through that open handle*, re-checks size and mtime against the catalog
record:

* On Windows the handle is opened with ``GENERIC_READ`` and share mode
  ``FILE_SHARE_READ`` only. While it is open nobody can open the file for
  writing, truncate, rename or delete it (sharing violation), while REAPER can
  still open it for reading. So the version verified through the handle is the
  version REAPER reads, for as long as the guard is held.
* Elsewhere (tests on non-Windows hosts) the re-check still happens, but no
  lock is possible; the Lua side's size check is the only later check.

The guard is held until the receipt is in (insert) or while the preview
generation is current (preview start). Residual boundary: once the guard is
released, the file can change while REAPER keeps referencing it (an inserted
item, a preview that outlives its guard); that is outside a single command.

No content hashing and no copying of assets.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from audiosearch.contracts.models import AssetRecord, ErrorCode

from .sources import SourceUnavailableError

_SHARING_VIOLATION = 32
_LOCK_VIOLATION = 33
_NOT_FOUND = {2, 3}


class SourceGuard:
    """An open read-only handle on a verified source version. ``release()`` is idempotent."""

    def __init__(self, path: Path, fd: int, size: int, mtime_ns: int, locked: bool) -> None:
        self.path = path
        self.size = size
        self.mtime_ns = mtime_ns
        self.locked = locked  # True: writes/renames/deletes are refused while held
        self._fd: int | None = fd
        self._lock = threading.Lock()

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        with self._lock:
            fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> SourceGuard:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def _open_windows(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    generic_read, file_share_read, open_existing, normal = 0x80000000, 0x1, 3, 0x80
    handle = create(str(path), generic_read, file_share_read, None, open_existing, normal, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        if error in (_SHARING_VIOLATION, _LOCK_VIOLATION):
            raise SourceUnavailableError(
                ErrorCode.ASSET_CHANGED, "源文件正被其他程序写入或占用，无法确认版本，未执行"
            )
        if error in _NOT_FOUND:
            raise SourceUnavailableError(ErrorCode.ASSET_OFFLINE, "源文件已不存在，未执行")
        raise SourceUnavailableError(
            ErrorCode.ASSET_OFFLINE, f"源文件无法以只读保护方式打开（错误 {error}），未执行"
        )
    return msvcrt.open_osfhandle(handle, os.O_RDONLY)


def guard_source(path: Path, asset: AssetRecord) -> SourceGuard:
    """Open ``path`` read-only (write/delete-locked on Windows) and verify its version.

    The version is the catalog's size + mtime of one of the asset's recorded
    locations. Raises ``SourceUnavailableError`` (``asset_changed`` /
    ``asset_offline``) when it cannot be confirmed; nothing is left open then.
    """
    if sys.platform == "win32":
        fd = _open_windows(path)
        locked = True
    else:
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError as exc:
            raise SourceUnavailableError(ErrorCode.ASSET_OFFLINE, "源文件已不存在，未执行") from exc
        except OSError as exc:
            raise SourceUnavailableError(
                ErrorCode.ASSET_OFFLINE, f"源文件无法打开（{type(exc).__name__}），未执行"
            ) from exc
        locked = False
    try:
        stat = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise SourceUnavailableError(ErrorCode.ASSET_OFFLINE, "无法读取源文件状态，未执行") from exc
    versions = {(loc.size_bytes, loc.mtime_ns) for loc in asset.locations}
    if (stat.st_size, stat.st_mtime_ns) not in versions:
        os.close(fd)
        raise SourceUnavailableError(
            ErrorCode.ASSET_CHANGED, "源文件大小/修改时间与目录记录的版本不一致，未执行"
        )
    return SourceGuard(Path(path), fd, stat.st_size, stat.st_mtime_ns, locked)


__all__ = ["SourceGuard", "guard_source"]
