# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Resolved paths must stay inside the mapped root.

``relative_path`` cannot contain ``..`` (the contract rejects it), but a junction
or symlink placed inside the root can still point elsewhere. The catalog
resolves the path and refuses anything that lands outside the resolved root.

Everything happens under ``tmp_path``; no real sound library is touched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from wav_fixtures import write_wav

from audiosearch.audioio import AssetOfflineError, ManifestCatalog
from audiosearch.contracts.models import AssetLocation, AssetRecord, SampleManifest


def _link_directory(link: Path, target: Path) -> bool:
    """Create a directory link; junctions need no privilege on Windows."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        pass
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    return False


def _catalog(root: Path, relative_path: str, *, size: int, mtime: int) -> ManifestCatalog:
    asset = AssetRecord(
        asset_id="11111111-1111-5111-8111-111111111111",
        content_revision=f"stat-v1:{size}:{mtime}",
        name=Path(relative_path).name,
        library_id="test-library",
        locations=[
            AssetLocation(
                location_id="22222222-2222-5222-8222-222222222222",
                root_id="test-root",
                relative_path=relative_path,
                availability="online",
                size_bytes=size,
                mtime_ns=mtime,
            )
        ],
        duration_seconds=1.0,
        sample_rate_hz=8000,
        channel_count=1,
        bit_depth=16,
    )
    manifest = SampleManifest(
        manifest_id="sha256:" + "5" * 64,
        generated_at="2026-11-10T00:00:00Z",
        selection_method="sha256-relative-path-v1",
        selection_limit=8,
        library_id="test-library",
        root_id="test-root",
        assets=[asset],
    )
    return ManifestCatalog(manifest, {"test-root": root})


def test_path_inside_the_root_is_accepted(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    path = write_wav(root / "ok.wav", np.zeros((800, 1), dtype=np.float32), rate=8000)
    stat = path.stat()
    catalog = _catalog(root, "ok.wav", size=stat.st_size, mtime=stat.st_mtime_ns)
    resolved = catalog.path_for(catalog.assets[0])
    assert resolved == path.resolve()


def test_junction_escaping_the_root_is_refused(tmp_path):
    root = tmp_path / "library"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = write_wav(outside / "secret.wav", np.zeros((800, 1)), rate=8000)
    if not _link_directory(root / "escape", outside):
        pytest.skip("this host cannot create directory links (needs junction/symlink support)")

    stat = target.stat()
    catalog = _catalog(root, "escape/secret.wav", size=stat.st_size, mtime=stat.st_mtime_ns)
    with pytest.raises(AssetOfflineError, match="outside its root"):
        catalog.path_for(catalog.assets[0])


def test_symlinked_file_escaping_the_root_is_refused(tmp_path):
    root = tmp_path / "library"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = write_wav(outside / "secret.wav", np.zeros((800, 1)), rate=8000)
    link = root / "linked.wav"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("file symlinks are not permitted on this host")

    stat = target.stat()
    catalog = _catalog(root, "linked.wav", size=stat.st_size, mtime=stat.st_mtime_ns)
    with pytest.raises(AssetOfflineError, match="outside its root"):
        catalog.path_for(catalog.assets[0])


def test_missing_root_is_reported_as_offline(tmp_path):
    catalog = _catalog(tmp_path / "gone", "a.wav", size=1, mtime=1)
    with pytest.raises(AssetOfflineError, match="not available"):
        catalog.path_for(catalog.assets[0])


def test_root_that_is_a_file_is_rejected(tmp_path):
    root = tmp_path / "not-a-dir"
    root.write_text("x", encoding="utf-8")
    catalog = _catalog(root, "a.wav", size=1, mtime=1)
    with pytest.raises(AssetOfflineError, match="not a directory"):
        catalog.path_for(catalog.assets[0])
