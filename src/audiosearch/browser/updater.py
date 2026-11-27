"""Lightweight <1 MB GitHub Releases patch updater for Whistle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiosearch.browser.runtime_paths import executable_dir, is_frozen

APP_NAME = "Whistle"
APP_VERSION = "0.2.0"
RUNTIME_ABI = 2
DEFAULT_GITHUB_REPO = "Echo722/Whistle"

PATCH_DIR_NAME = "app_patch"
PATCH_PREV_DIR_NAME = "app_patch.prev"
PATCH_NEXT_DIR_NAME = "app_patch.next"
PATCH_FAILED_DIR_NAME = "app_patch.failed"
PATCH_MANIFEST_FILENAME = "patch_manifest.json"
BOOT_SENTINEL_FILENAME = ".boot_attempt"

_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class UpdateManifest:
    version: str
    min_runtime_abi: int
    patch_url: str
    patch_sha256: str
    patch_size: int
    release_url: str
    notes: str
    published_at: str


@dataclass(frozen=True, slots=True)
class UpdateCheckResult:
    status: str  # "up_to_date" | "patch_available" | "full_update_required" | "error"
    current_version: str
    current_runtime_abi: int
    manifest: UpdateManifest | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class PatchInstallResult:
    version: str
    patch_dir: Path
    prev_backup_dir: Path | None


def parse_version(raw: str) -> tuple[int, ...]:
    """Parse ``v0.1.2`` or ``0.1.2`` into a comparable tuple of ints."""
    cleaned = str(raw or "").strip().lstrip("vV")
    if not cleaned:
        return (0,)
    parts: list[int] = []
    for seg in cleaned.split("."):
        digits = []
        for ch in seg:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        parts.append(int("".join(digits)) if digits else 0)
    return tuple(parts) or (0,)


def is_newer_version(candidate: str, current: str) -> bool:
    cand = list(parse_version(candidate))
    cur = list(parse_version(current))
    width = max(len(cand), len(cur))
    cand.extend([0] * (width - len(cand)))
    cur.extend([0] * (width - len(cur)))
    return tuple(cand) > tuple(cur)


def resolve_app_root(app_root: Path | None = None) -> Path:
    if app_root is not None:
        return Path(app_root)
    return executable_dir()


def _active_patch_dir(app_root: Path | None = None) -> Path:
    env_dir = os.environ.get("WHISTLE_PATCH_DIR", "").strip()
    if env_dir and app_root is None:
        return Path(env_dir)
    return resolve_app_root(app_root) / PATCH_DIR_NAME


def active_patch_info(app_root: Path | None = None) -> dict[str, Any] | None:
    patch_dir = _active_patch_dir(app_root)
    manifest_path = patch_dir / PATCH_MANIFEST_FILENAME
    if not (patch_dir / "audiosearch" / "__init__.py").is_file() or not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def effective_version(app_root: Path | None = None) -> str:
    info = active_patch_info(app_root)
    if info is not None:
        ver = str(info.get("version") or "").strip().lstrip("vV")
        if ver:
            return ver
    return APP_VERSION


def can_rollback(app_root: Path | None = None) -> bool:
    root = resolve_app_root(app_root)
    patch_dir = root / PATCH_DIR_NAME
    return patch_dir.is_dir() and (patch_dir / "audiosearch" / "__init__.py").is_file()


def default_manifest_url(repo: str = DEFAULT_GITHUB_REPO) -> str:
    override = os.environ.get("WHISTLE_UPDATE_URL", "").strip()
    if override:
        return override
    clean_repo = repo.strip().strip("/") or DEFAULT_GITHUB_REPO
    return f"https://github.com/{clean_repo}/releases/latest/download/latest.json"


def _parse_manifest_payload(data: dict[str, Any]) -> UpdateManifest:
    version = str(data.get("version") or "").strip().lstrip("vV")
    if not version:
        raise ValueError("更新清单缺少 version 字段")
    return UpdateManifest(
        version=version,
        min_runtime_abi=int(data.get("min_runtime_abi") or 1),
        patch_url=str(data.get("patch_url") or "").strip(),
        patch_sha256=str(data.get("patch_sha256") or "").strip().lower(),
        patch_size=max(0, int(data.get("patch_size") or 0)),
        release_url=str(data.get("release_url") or "").strip(),
        notes=str(data.get("notes") or "").strip(),
        published_at=str(data.get("published_at") or "").strip(),
    )


def _urlopen_with_ua(url: str, *, current_version: str, timeout: float):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"{APP_NAME}/{current_version}"},
    )
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def check_for_updates(
    *,
    repo: str = DEFAULT_GITHUB_REPO,
    manifest_url: str | None = None,
    current_version: str | None = None,
    current_runtime_abi: int = RUNTIME_ABI,
    app_root: Path | None = None,
    timeout: float = 8.0,
) -> UpdateCheckResult:
    """Fetch remote/local ``latest.json`` and compare against active version and runtime ABI."""
    cur_ver = (
        current_version.strip().lstrip("vV")
        if current_version is not None
        else effective_version(app_root)
    )
    url = manifest_url or default_manifest_url(repo)

    try:
        with _urlopen_with_ua(url, current_version=cur_ver, timeout=timeout) as resp:
            raw_bytes = resp.read(2 * 1024 * 1024)
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("更新清单格式无效")
        manifest = _parse_manifest_payload(payload)
    except urllib.error.HTTPError as exc:
        return UpdateCheckResult(
            status="error",
            current_version=cur_ver,
            current_runtime_abi=current_runtime_abi,
            message=f"检查更新失败（HTTP {exc.code}）",
        )
    except Exception as exc:
        return UpdateCheckResult(
            status="error",
            current_version=cur_ver,
            current_runtime_abi=current_runtime_abi,
            message=f"无法连接更新源：{exc}",
        )

    if not is_newer_version(manifest.version, cur_ver):
        return UpdateCheckResult(
            status="up_to_date",
            current_version=cur_ver,
            current_runtime_abi=current_runtime_abi,
            manifest=manifest,
            message=f"当前已是最新版本 v{cur_ver}",
        )

    if manifest.min_runtime_abi > current_runtime_abi or not manifest.patch_url or not manifest.patch_sha256:
        return UpdateCheckResult(
            status="full_update_required",
            current_version=cur_ver,
            current_runtime_abi=current_runtime_abi,
            manifest=manifest,
            message=f"发现新版本 v{manifest.version}（含底层依赖升级，需前往发布页下载完整版）",
        )

    return UpdateCheckResult(
        status="patch_available",
        current_version=cur_ver,
        current_runtime_abi=current_runtime_abi,
        manifest=manifest,
        message=f"发现新版本 v{manifest.version}（增量热更新补丁）",
    )


def _rename_dir_with_retry(
    src: Path,
    dst: Path,
    *,
    attempts: int = 12,
    base_delay: float = 0.05,
) -> None:
    """Retry transient Windows Defender / indexer ``PermissionError`` locks on directory rename."""
    for idx in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if idx + 1 >= attempts:
                raise
            time.sleep(base_delay * (idx + 1))


def _rmtree_with_retry(
    target: Path,
    *,
    attempts: int = 8,
    base_delay: float = 0.05,
) -> None:
    """Retry transient Windows handle locks when clearing patch staging directories."""
    for idx in range(attempts):
        if not target.exists():
            return
        try:
            shutil.rmtree(target)
            return
        except PermissionError:
            if idx + 1 >= attempts:
                raise
            time.sleep(base_delay * (idx + 1))


def _validate_and_extract_zip(zip_path: Path, target_dir: Path) -> None:
    if target_dir.exists():
        _rmtree_with_retry(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members: list[tuple[zipfile.ZipInfo, Path]] = []
        has_audiosearch_pkg = False
        for info in zf.infolist():
            norm_name = info.filename.replace("\\", "/")
            while norm_name.startswith("./"):
                norm_name = norm_name[2:]
            if not norm_name:
                continue
            if norm_name.startswith("/") or ":" in norm_name.split("/")[0]:
                raise ValueError(f"补丁包包含非法路径：{info.filename}")
            parts = [p for p in norm_name.split("/") if p]
            if any(p == ".." for p in parts):
                raise ValueError(f"补丁包包含越界路径：{info.filename}")
            dest = (target_dir / Path(*parts)).resolve()
            if dest != target_resolved and target_resolved not in dest.parents:
                raise ValueError(f"补丁包包含越界路径：{info.filename}")
            if parts == ["audiosearch", "__init__.py"]:
                has_audiosearch_pkg = True
            members.append((info, dest))

        if not has_audiosearch_pkg:
            raise ValueError("补丁包结构无效：缺少 audiosearch/__init__.py")

        for info, dest in members:
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK_BYTES)


def apply_patch_zip(
    zip_path: Path,
    manifest: UpdateManifest,
    *,
    app_root: Path | None = None,
) -> PatchInstallResult:
    """Verify SHA-256 of ``zip_path``, extract safely, and swap into ``app_patch``."""
    digest = hashlib.sha256()
    with Path(zip_path).open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    actual_sha = digest.hexdigest().lower()
    if actual_sha != manifest.patch_sha256.strip().lower():
        raise ValueError(
            f"补丁文件 SHA-256 校验失败（期望 {manifest.patch_sha256[:12]}…，实际 {actual_sha[:12]}…）"
        )

    root = resolve_app_root(app_root)
    patch_dir = root / PATCH_DIR_NAME
    prev_dir = root / PATCH_PREV_DIR_NAME
    next_dir = root / PATCH_NEXT_DIR_NAME

    _validate_and_extract_zip(Path(zip_path), next_dir)
    manifest_payload = {
        "version": manifest.version,
        "min_runtime_abi": manifest.min_runtime_abi,
        "notes": manifest.notes,
        "published_at": manifest.published_at,
        "patch_sha256": actual_sha,
    }
    (next_dir / PATCH_MANIFEST_FILENAME).write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (next_dir / BOOT_SENTINEL_FILENAME).write_text("0", encoding="utf-8")

    if prev_dir.exists():
        _rmtree_with_retry(prev_dir)
    moved_prev: Path | None = None
    if patch_dir.exists():
        _rename_dir_with_retry(patch_dir, prev_dir)
        moved_prev = prev_dir
    try:
        _rename_dir_with_retry(next_dir, patch_dir)
    except Exception:
        if moved_prev is not None and moved_prev.exists() and not patch_dir.exists():
            try:
                _rename_dir_with_retry(moved_prev, patch_dir)
            except OSError:
                pass
        raise

    return PatchInstallResult(
        version=manifest.version,
        patch_dir=patch_dir,
        prev_backup_dir=moved_prev,
    )


def download_and_stage_patch(
    manifest: UpdateManifest,
    *,
    app_root: Path | None = None,
    current_version: str = APP_VERSION,
    timeout: float = 30.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> PatchInstallResult:
    """Download ``manifest.patch_url`` to temp file, verify SHA-256, and stage ``app_patch``."""
    if not manifest.patch_url or not manifest.patch_sha256:
        raise ValueError("更新清单缺少补丁下载地址或校验值")

    root = resolve_app_root(app_root)
    root.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="whistle-patch-", suffix=".zip", dir=str(root))
    os.close(tmp_fd)
    tmp_zip = Path(tmp_name)
    try:
        with _urlopen_with_ua(manifest.patch_url, current_version=current_version, timeout=timeout) as resp:
            hdr_len = resp.headers.get("Content-Length") if resp.headers else None
            total = int(hdr_len) if hdr_len and hdr_len.isdigit() else manifest.patch_size
            downloaded = 0
            with tmp_zip.open("wb") as out:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total)
        return apply_patch_zip(tmp_zip, manifest, app_root=root)
    finally:
        tmp_zip.unlink(missing_ok=True)


def mark_patch_boot_healthy(app_root: Path | None = None) -> None:
    """Clear pending-boot sentinel once MainWindow starts cleanly."""
    patch_dir = _active_patch_dir(app_root)
    sentinel = patch_dir / BOOT_SENTINEL_FILENAME
    try:
        sentinel.unlink(missing_ok=True)
    except OSError:
        pass


def rollback_patch(app_root: Path | None = None) -> bool:
    """Revert active ``app_patch`` to ``app_patch.prev`` (or built-in runtime)."""
    root = resolve_app_root(app_root)
    patch_dir = root / PATCH_DIR_NAME
    prev_dir = root / PATCH_PREV_DIR_NAME
    failed_dir = root / PATCH_FAILED_DIR_NAME

    if not patch_dir.exists() and not prev_dir.exists():
        return False
    if failed_dir.exists():
        shutil.rmtree(failed_dir, ignore_errors=True)
    if patch_dir.exists():
        try:
            _rename_dir_with_retry(patch_dir, failed_dir)
        except OSError:
            shutil.rmtree(patch_dir, ignore_errors=True)
    if prev_dir.exists():
        _rename_dir_with_retry(prev_dir, patch_dir)
    return True


def is_running_from_patch() -> bool:
    return bool(os.environ.get("WHISTLE_PATCH_DIR", "").strip()) or (
        is_frozen() and can_rollback()
    )


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "BOOT_SENTINEL_FILENAME",
    "DEFAULT_GITHUB_REPO",
    "PATCH_DIR_NAME",
    "PATCH_MANIFEST_FILENAME",
    "PATCH_NEXT_DIR_NAME",
    "PATCH_PREV_DIR_NAME",
    "PatchInstallResult",
    "RUNTIME_ABI",
    "UpdateCheckResult",
    "UpdateManifest",
    "active_patch_info",
    "apply_patch_zip",
    "can_rollback",
    "check_for_updates",
    "default_manifest_url",
    "download_and_stage_patch",
    "effective_version",
    "is_newer_version",
    "is_running_from_patch",
    "mark_patch_boot_healthy",
    "parse_version",
    "resolve_app_root",
    "rollback_patch",
]
