"""Cross-platform zero-dependency builder for Whistle <1 MB hot-update patches."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UPDATER_PY = _REPO_ROOT / "src" / "audiosearch" / "browser" / "updater.py"
_EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _read_default_metadata() -> tuple[str, int]:
    version = "0.1.0"
    min_abi = 1
    if _UPDATER_PY.is_file():
        text = _UPDATER_PY.read_text(encoding="utf-8")
        m_ver = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m_ver:
            version = m_ver.group(1).strip().lstrip("vV") or version
        m_abi = re.search(r"^RUNTIME_ABI\s*=\s*(\d+)", text, re.MULTILINE)
        if m_abi:
            min_abi = int(m_abi.group(1))
    return version, min_abi


def build_patch(
    *,
    version: str | None = None,
    notes: str = "",
    repo: str = "Echo722/Whistle",
    min_runtime_abi: int | None = None,
    out_dir: str | Path = ".local/dist/patches",
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Build ``whistle-patch-vX.Y.Z.zip`` and ``latest.json`` cleanly with stdlib only."""
    root = (repo_root or _REPO_ROOT).resolve()
    default_ver, default_abi = _read_default_metadata()

    clean_ver = (version if version is not None else default_ver).strip().lstrip("vV")
    if not clean_ver:
        raise ValueError("Version 不能为空。")
    abi = default_abi if min_runtime_abi is None else int(min_runtime_abi)

    out_path = Path(out_dir)
    out_abs = out_path if out_path.is_absolute() else (root / out_path)
    out_abs.mkdir(parents=True, exist_ok=True)

    src_pkg = root / "src" / "audiosearch"
    if not (src_pkg / "__init__.py").is_file():
        raise FileNotFoundError(f"未找到包目录：{src_pkg}")

    built_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch_manifest = json.dumps(
        {
            "version": clean_ver,
            "min_runtime_abi": abi,
            "notes": notes,
            "built_at": built_at,
        },
        ensure_ascii=False,
        indent=2,
    )

    zip_name = f"whistle-patch-v{clean_ver}.zip"
    zip_path = out_abs / zip_name
    tmp_zip = out_abs / f".{zip_name}.{os.getpid()}.tmp"
    if tmp_zip.exists():
        tmp_zip.unlink()

    files_to_pack: list[tuple[Path, str]] = []
    for file_path in sorted(src_pkg.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(src_pkg)
        if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if file_path.suffix.lower() in _EXCLUDED_SUFFIXES:
            continue
        files_to_pack.append((file_path, f"audiosearch/{rel.as_posix()}"))

    src_lua = root / "reaper" / "audiosearch_spot_bridge.lua"
    if src_lua.is_file():
        files_to_pack.append((src_lua, "reaper/audiosearch_spot_bridge.lua"))

    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for src_file, arcname in files_to_pack:
                zf.write(src_file, arcname=arcname)
            zf.writestr("patch_manifest.json", patch_manifest.encode("utf-8"))
        os.replace(tmp_zip, zip_path)
    except Exception:
        tmp_zip.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256()
    with zip_path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    patch_sha256 = digest.hexdigest().lower()
    patch_size = zip_path.stat().st_size

    clean_repo = repo.strip().strip("/") or "Echo722/Whistle"
    latest_payload = {
        "version": clean_ver,
        "min_runtime_abi": abi,
        "patch_url": f"https://github.com/{clean_repo}/releases/download/v{clean_ver}/{zip_name}",
        "patch_sha256": patch_sha256,
        "patch_size": patch_size,
        "release_url": f"https://github.com/{clean_repo}/releases/tag/v{clean_ver}",
        "notes": notes,
        "published_at": built_at,
    }
    latest_path = out_abs / "latest.json"
    latest_path.write_text(
        json.dumps(latest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return zip_path, latest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Whistle <1 MB hot-update patch and latest.json")
    parser.add_argument("--version", default=None, help="Release version (default: APP_VERSION from updater.py)")
    parser.add_argument("--notes", default="", help="Release notes")
    parser.add_argument("--repo", default="Echo722/Whistle", help="GitHub repository slug (owner/name)")
    parser.add_argument("--min-runtime-abi", type=int, default=None, help="Minimum compatible runtime ABI")
    parser.add_argument("--out-dir", default=".local/dist/patches", help="Output directory for zip and latest.json")
    args = parser.parse_args(argv)

    zip_path, latest_path = build_patch(
        version=args.version,
        notes=args.notes,
        repo=args.repo,
        min_runtime_abi=args.min_runtime_abi,
        out_dir=args.out_dir,
    )
    size_kb = round(zip_path.stat().st_size / 1024.0, 1)
    print(f"[OK] Patch zip: {zip_path} ({size_kb} KB)")
    print(f"[OK] Manifest:  {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
