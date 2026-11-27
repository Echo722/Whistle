"""In-app on-demand CUDA 12.8 runtime upgrader for frozen Windows Whistle bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiosearch.browser.runtime_paths import is_frozen
from audiosearch.browser.updater import resolve_app_root
from audiosearch.inference.runtime import probe_cuda

CUDA_UPGRADE_TARGET_VERSION = "2.9.1+cu128"
CUDA_UPGRADE_MIN_DRIVER = "570.65"
DEFAULT_CUDA_WHEEL_BASE_URL = "https://download.pytorch.org/whl/cu128"
CUDA_STAGE_DIR_NAME = "_cuda_upgrade_stage"
CUDA_CACHE_DIR_NAME = "_cuda_upgrade_cache"
CUDA_BACKUP_DIR_NAME = "_cuda_backup"
DEFAULT_MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB
_CHUNK_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class CudaWheelSpec:
    name: str
    url_filename: str
    local_filename: str
    sha256: str
    approx_bytes: int
    package_dir: str
    dist_info_prefix: str


DEFAULT_CUDA_WHEEL_SPECS: tuple[CudaWheelSpec, ...] = (
    CudaWheelSpec(
        name="torch",
        url_filename="torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl",
        local_filename="torch-2.9.1+cu128-cp312-cp312-win_amd64.whl",
        sha256="3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0",
        approx_bytes=2_867_000_000,
        package_dir="torch",
        dist_info_prefix="torch-",
    ),
    CudaWheelSpec(
        name="torchvision",
        url_filename="torchvision-0.24.1%2Bcu128-cp312-cp312-win_amd64.whl",
        local_filename="torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl",
        sha256="33ecea57afa1daeedfed443a8a0cb8e4b0b403fdf18c2a328ba6f9069d403384",
        approx_bytes=9_650_000,
        package_dir="torchvision",
        dist_info_prefix="torchvision-",
    ),
)


@dataclass(frozen=True, slots=True)
class CudaRuntimeInfo:
    is_frozen: bool
    internal_dir: Path | None
    can_upgrade_in_place: bool
    torch_dist_version: str
    supports_sm120: bool
    cuda_available: bool
    cuda_usable: bool
    gpu_name: str
    gpu_capability: str
    probe_reason: str
    upgrade_recommended: bool
    upgrade_available: bool


@dataclass(frozen=True, slots=True)
class CudaStageResult:
    target_version: str
    stage_dir: Path
    payload_dir: Path
    apply_script_path: Path


def resolve_cuda_wheel_base_url(base_url: str | None = None) -> str:
    if base_url and base_url.strip():
        return base_url.strip().rstrip("/")
    env_url = os.environ.get("WHISTLE_CUDA_WHEEL_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return DEFAULT_CUDA_WHEEL_BASE_URL


def _dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".cuda_write_probe.{os.getpid()}"
        probe.write_bytes(b"1")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _detect_torch_dist_version(internal_dir: Path | None, torch_mod: Any | None) -> str:
    if internal_dir is not None and internal_dir.is_dir():
        for entry in sorted(internal_dir.iterdir()):
            if entry.is_dir():
                m = re.match(r"^torch-(.+)\.dist-info$", entry.name)
                if m:
                    return m.group(1)
    try:
        return importlib.metadata.version("torch")
    except Exception:
        pass
    if torch_mod is not None:
        return str(getattr(torch_mod, "__version__", "") or "")
    return ""


def inspect_cuda_runtime(
    app_root: Path | None = None,
    *,
    frozen_override: bool | None = None,
    torch_mod: Any | None = None,
) -> CudaRuntimeInfo:
    """Inspect bundled PyTorch CUDA compatibility and whether in-place cu128 upgrade applies."""
    root = resolve_app_root(app_root)
    frozen = is_frozen() if frozen_override is None else bool(frozen_override)
    cand_internal = root / "_internal"
    internal_dir = cand_internal if (frozen and (cand_internal / "torch").is_dir()) else None

    if torch_mod is None:
        try:
            import torch as _torch

            torch_mod = _torch
        except Exception:
            torch_mod = None

    torch_dist_version = _detect_torch_dist_version(internal_dir, torch_mod)
    cuda_available = False
    cuda_usable = False
    gpu_name = ""
    gpu_capability = ""
    probe_reason = "PyTorch 未安装"
    arch_list: list[str] = []

    if torch_mod is not None:
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
        except Exception:
            cuda_available = False
        try:
            arch_list = [str(x) for x in torch_mod.cuda.get_arch_list()]
        except Exception:
            arch_list = []
        if cuda_available:
            try:
                gpu_name = str(torch_mod.cuda.get_device_name(0) or "")
            except Exception:
                gpu_name = ""
            try:
                maj, min_ = torch_mod.cuda.get_device_capability(0)
                gpu_capability = f"sm_{int(maj)}{int(min_)}"
            except Exception:
                gpu_capability = ""
        cuda_usable, probe_reason = probe_cuda(torch_mod, force=True)

    supports_sm120 = ("sm_120" in arch_list) or ("cu128" in torch_dist_version.lower())
    writable = bool(internal_dir is not None and _dir_writable(root) and _dir_writable(internal_dir))
    can_upgrade_in_place = bool(frozen and os.name == "nt" and internal_dir is not None and writable)
    upgrade_available = bool(can_upgrade_in_place and not supports_sm120)
    upgrade_recommended = bool(upgrade_available and cuda_available and not cuda_usable)

    return CudaRuntimeInfo(
        is_frozen=frozen,
        internal_dir=internal_dir,
        can_upgrade_in_place=can_upgrade_in_place,
        torch_dist_version=torch_dist_version,
        supports_sm120=supports_sm120,
        cuda_available=cuda_available,
        cuda_usable=cuda_usable,
        gpu_name=gpu_name,
        gpu_capability=gpu_capability,
        probe_reason=probe_reason,
        upgrade_recommended=upgrade_recommended,
        upgrade_available=upgrade_available,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def _open_wheel_response(url: str, *, offset: int, timeout: float):
    headers = {"User-Agent": "Whistle-CUDA-Updater/1.0"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def _download_wheel_with_resume(
    base_url: str,
    spec: CudaWheelSpec,
    dest_path: Path,
    *,
    bytes_before: int,
    total_bytes: int,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    timeout: float = 30.0,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        if progress_cb is not None:
            progress_cb(bytes_before + dest_path.stat().st_size, total_bytes, f"正在校验缓存 {spec.name}…")
        if _sha256_file(dest_path) == spec.sha256.lower():
            return dest_path
        dest_path.unlink(missing_ok=True)

    part_path = dest_path.with_name(dest_path.name + ".part")
    candidates = [f"{base_url}/{spec.url_filename}"]
    if spec.local_filename != spec.url_filename:
        candidates.append(f"{base_url}/{spec.local_filename}")

    resp = None
    last_exc: Exception | None = None
    offset = part_path.stat().st_size if part_path.is_file() else 0

    for url in candidates:
        attempt_offset = offset
        for _ in range(2):
            try:
                resp = _open_wheel_response(url, offset=attempt_offset, timeout=timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and attempt_offset > 0:
                    part_path.unlink(missing_ok=True)
                    attempt_offset = 0
                    offset = 0
                    continue
                last_exc = exc
                if exc.code == 404:
                    break
                raise
            except urllib.error.URLError as exc:
                last_exc = exc
                break
        if resp is not None:
            offset = attempt_offset
            break

    if resp is None:
        raise RuntimeError(f"无法下载 {spec.name} 运行库：{last_exc}")

    try:
        with resp:
            status = getattr(resp, "status", 200)
            mode = "ab" if (offset > 0 and status == 206) else "wb"
            downloaded = offset if mode == "ab" else 0
            if mode == "wb" and part_path.exists():
                part_path.unlink(missing_ok=True)
            with part_path.open(mode) as out:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("已取消下载")
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(
                            bytes_before + downloaded,
                            total_bytes,
                            f"正在下载 {spec.name} 运行库…",
                        )
    except Exception:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("已取消下载") from None
        raise

    if progress_cb is not None:
        progress_cb(bytes_before + downloaded, total_bytes, f"正在校验 {spec.name} SHA-256…")
    actual_sha = _sha256_file(part_path)
    if actual_sha != spec.sha256.lower():
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"{spec.name} SHA-256 校验失败（期望 {spec.sha256[:12]}…，实际 {actual_sha[:12]}…）"
        )
    os.replace(part_path, dest_path)
    return dest_path


def _extract_wheel_safe(wheel_path: Path, payload_dir: Path, spec: CudaWheelSpec) -> None:
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_resolved = payload_dir.resolve()

    with zipfile.ZipFile(wheel_path, "r") as zf:
        members: list[tuple[zipfile.ZipInfo, Path]] = []
        matched_dist_dirs: set[str] = set()
        for info in zf.infolist():
            norm = info.filename.replace("\\", "/")
            while norm.startswith("./"):
                norm = norm[2:]
            if not norm:
                continue
            if norm.startswith("/") or ":" in norm.split("/")[0]:
                raise ValueError(f"运行库包包含非法路径：{info.filename}")
            parts = [p for p in norm.split("/") if p]
            if any(p == ".." for p in parts):
                raise ValueError(f"运行库包包含越界路径：{info.filename}")
            top = parts[0]
            is_pkg = top == spec.package_dir
            is_dist = top.startswith(spec.dist_info_prefix) and top.endswith(".dist-info")
            if not (is_pkg or is_dist):
                continue
            dest = (payload_dir / Path(*parts)).resolve()
            if dest != payload_resolved and payload_resolved not in dest.parents:
                raise ValueError(f"运行库包包含越界路径：{info.filename}")
            if is_dist:
                matched_dist_dirs.add(top)
            members.append((info, dest))

        for info, dest in members:
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK_BYTES)

    if not (payload_dir / spec.package_dir).is_dir() or not matched_dist_dirs:
        raise ValueError(f"运行库包结构不完整：缺少 {spec.package_dir} 或 .dist-info")


def _cmd_quote(value: str | Path) -> str:
    return str(value).replace("%", "%%")


def write_apply_cuda_script(
    app_root: Path,
    payload_dir: Path,
    *,
    target_pid: int | None = None,
    relaunch_exe: Path | None = None,
    log_path: Path | None = None,
) -> Path:
    """Write a CRLF Windows batch script that atomically swaps CUDA libs after Whistle exits."""
    root = Path(app_root).resolve()
    stage_dir = root / CUDA_STAGE_DIR_NAME
    stage_dir.mkdir(parents=True, exist_ok=True)
    script_path = stage_dir / "apply_cuda_upgrade.cmd"
    internal_dir = root / "_internal"
    backup_dir = internal_dir / CUDA_BACKUP_DIR_NAME
    resolved_payload = Path(payload_dir).resolve()
    resolved_log = (
        Path(log_path).resolve()
        if log_path is not None
        else (root / "data" / "logs" / "cuda-upgrade.log")
    )
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    pid_val = int(target_pid or 0)
    exe_str = _cmd_quote(Path(relaunch_exe).resolve()) if relaunch_exe is not None else ""

    lines = [
        "@echo off",
        "setlocal enabledelayedexpansion",
        "chcp 65001 >nul",
        "title Whistle - 正在升级 CUDA 12.8 运行库",
        f'set "INTERNAL={_cmd_quote(internal_dir)}"',
        f'set "PAYLOAD={_cmd_quote(resolved_payload)}"',
        f'set "BACKUP={_cmd_quote(backup_dir)}"',
        f'set "LOGFILE={_cmd_quote(resolved_log)}"',
        f'set "EXE={exe_str}"',
        "",
        'echo [%date% %time%] Start CUDA 12.8 runtime swap>>"%LOGFILE%"',
        "echo 正在等待 Whistle 主程序退出释放文件锁...",
    ]

    if pid_val > 0:
        lines.extend(
            [
                "for /L %%i in (1,1,30) do (",
                f'    tasklist /FI "PID eq {pid_val}" /FO CSV /NH 2>nul | findstr /C:",\\"{pid_val}\\"," >nul',
                "    if errorlevel 1 goto :do_swap",
                "    ping -n 2 127.0.0.1 >nul",
                ")",
            ]
        )

    lines.extend(
        [
            ":do_swap",
            "echo 正在备份旧版运行库并切入 CUDA 12.8...",
            'if exist "%BACKUP%" rmdir /s /q "%BACKUP%"',
            'mkdir "%BACKUP%"',
            'if errorlevel 1 goto :fail_early',
            'if exist "%INTERNAL%\\torch" move /y "%INTERNAL%\\torch" "%BACKUP%\\torch" >nul',
            "if errorlevel 1 goto :rollback",
            'if exist "%INTERNAL%\\torchvision" move /y "%INTERNAL%\\torchvision" "%BACKUP%\\torchvision" >nul',
            "if errorlevel 1 goto :rollback",
            'for /d %%D in ("%INTERNAL%\\torch-*.dist-info") do (',
            '    move /y "%%~fD" "%BACKUP%\\%%~nxD" >nul',
            "    if errorlevel 1 goto :rollback",
            ")",
            'for /d %%D in ("%INTERNAL%\\torchvision-*.dist-info") do (',
            '    move /y "%%~fD" "%BACKUP%\\%%~nxD" >nul',
            "    if errorlevel 1 goto :rollback",
            ")",
            "",
            'if not exist "%PAYLOAD%\\torch" goto :rollback',
            'move /y "%PAYLOAD%\\torch" "%INTERNAL%\\torch" >nul',
            "if errorlevel 1 goto :rollback",
            'if exist "%PAYLOAD%\\torchvision" (',
            '    move /y "%PAYLOAD%\\torchvision" "%INTERNAL%\\torchvision" >nul',
            "    if errorlevel 1 goto :rollback",
            ")",
            'for /d %%D in ("%PAYLOAD%\\*.dist-info") do (',
            '    move /y "%%~fD" "%INTERNAL%\\%%~nxD" >nul',
            "    if errorlevel 1 goto :rollback",
            ")",
            "",
            'echo [%date% %time%] CUDA 12.8 swap succeeded>>"%LOGFILE%"',
            'rmdir /s /q "%BACKUP%" >nul 2>nul',
            'rmdir /s /q "%PAYLOAD%" >nul 2>nul',
            'if not "%EXE%"=="" start "" "%EXE%"',
            "exit /b 0",
            "",
            ":fail_early",
            'echo [%date% %time%] Failed to create backup dir>>"%LOGFILE%"',
            'if not "%EXE%"=="" start "" "%EXE%"',
            "exit /b 1",
            "",
            ":rollback",
            'echo [%date% %time%] Swap failed, rolling back>>"%LOGFILE%"',
            'for /d %%D in ("%BACKUP%\\*") do (',
            '    if exist "%INTERNAL%\\%%~nxD" rmdir /s /q "%INTERNAL%\\%%~nxD" >nul 2>nul',
            '    move /y "%%~fD" "%INTERNAL%\\%%~nxD" >nul',
            ")",
            'rmdir /s /q "%BACKUP%" >nul 2>nul',
            'if not "%EXE%"=="" start "" "%EXE%"',
            "exit /b 1",
        ]
    )

    script_path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return script_path


def download_and_stage_cuda_upgrade(
    *,
    app_root: Path | None = None,
    base_url: str | None = None,
    specs: tuple[CudaWheelSpec, ...] = DEFAULT_CUDA_WHEEL_SPECS,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    target_pid: int | None = None,
    relaunch_exe: Path | None = None,
) -> CudaStageResult:
    """Download, verify, extract, and stage cu128 PyTorch/TorchVision runtime."""
    root = resolve_app_root(app_root).resolve()
    internal_dir = root / "_internal"
    if not internal_dir.is_dir():
        raise FileNotFoundError(f"未找到便携版运行库目录：{internal_dir}")
    if not (_dir_writable(root) and _dir_writable(internal_dir)):
        raise PermissionError("当前程序目录不可写，无法原地升级 CUDA 运行库")

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < min_free_bytes:
        need_gb = min_free_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        raise OSError(f"磁盘可用空间不足（需要至少 {need_gb:.1f} GB，当前仅剩 {free_gb:.1f} GB）")

    resolved_base = resolve_cuda_wheel_base_url(base_url)
    cache_dir = root / CUDA_CACHE_DIR_NAME
    stage_dir = root / CUDA_STAGE_DIR_NAME
    payload_dir = stage_dir / "payload"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(max(1, int(s.approx_bytes)) for s in specs)
    done_before = 0
    downloaded_wheels: list[tuple[CudaWheelSpec, Path]] = []

    for spec in specs:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("已取消下载")
        wheel_path = cache_dir / spec.local_filename
        _download_wheel_with_resume(
            resolved_base,
            spec,
            wheel_path,
            bytes_before=done_before,
            total_bytes=total_bytes,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
        )
        done_before += max(int(spec.approx_bytes), wheel_path.stat().st_size)
        downloaded_wheels.append((spec, wheel_path))

    for spec, wheel_path in downloaded_wheels:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("已取消下载")
        if progress_cb is not None:
            progress_cb(total_bytes, total_bytes, f"正在解压 {spec.name} 运行库…")
        _extract_wheel_safe(wheel_path, payload_dir, spec)

    for _, wheel_path in downloaded_wheels:
        wheel_path.unlink(missing_ok=True)
    try:
        if cache_dir.is_dir() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    except OSError:
        pass

    script_path = write_apply_cuda_script(
        root,
        payload_dir,
        target_pid=target_pid if target_pid is not None else os.getpid(),
        relaunch_exe=relaunch_exe,
    )
    (stage_dir / "ready.json").write_text(
        json.dumps(
            {
                "target_version": CUDA_UPGRADE_TARGET_VERSION,
                "payload_dir": str(payload_dir),
                "script_path": str(script_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return CudaStageResult(
        target_version=CUDA_UPGRADE_TARGET_VERSION,
        stage_dir=stage_dir,
        payload_dir=payload_dir,
        apply_script_path=script_path,
    )


def launch_apply_cuda_script(script_path: Path) -> None:
    """Launch the staged swap script in a detached console window before exiting Whistle."""
    cmd = ["cmd.exe", "/c", str(Path(script_path).resolve())]
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    subprocess.Popen(cmd, creationflags=flags)  # noqa: S603


__all__ = [
    "CUDA_BACKUP_DIR_NAME",
    "CUDA_CACHE_DIR_NAME",
    "CUDA_STAGE_DIR_NAME",
    "CUDA_UPGRADE_MIN_DRIVER",
    "CUDA_UPGRADE_TARGET_VERSION",
    "DEFAULT_CUDA_WHEEL_BASE_URL",
    "DEFAULT_CUDA_WHEEL_SPECS",
    "DEFAULT_MIN_FREE_BYTES",
    "CudaRuntimeInfo",
    "CudaStageResult",
    "CudaWheelSpec",
    "download_and_stage_cuda_upgrade",
    "inspect_cuda_runtime",
    "launch_apply_cuda_script",
    "resolve_cuda_wheel_base_url",
    "write_apply_cuda_script",
]
