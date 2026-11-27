"""SQLite + FTS5 schema and connection setup for the standalone audio browser."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    is_online INTEGER NOT NULL DEFAULT 1,
    file_count INTEGER NOT NULL DEFAULT 0,
    last_scan_at REAL,
    scan_cursor TEXT,
    status TEXT NOT NULL DEFAULT 'idle',
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    file_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(root_id, rel_path)
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
    rel_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    sample_rate INTEGER NOT NULL DEFAULT 0,
    channels INTEGER NOT NULL DEFAULT 0,
    bit_depth INTEGER NOT NULL DEFAULT 0,
    package TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    embedded_text TEXT NOT NULL DEFAULT '',
    bext_json TEXT NOT NULL DEFAULT '{}',
    ixml_json TEXT NOT NULL DEFAULT '{}',
    is_available INTEGER NOT NULL DEFAULT 1,
    indexed_at REAL NOT NULL DEFAULT 0.0,
    UNIQUE(root_id, rel_path)
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filename,
    rel_path,
    package,
    description,
    embedded_text,
    content='files',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, filename, rel_path, package, description, embedded_text)
    VALUES (new.id, new.filename, new.rel_path, new.package, new.description, new.embedded_text);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, rel_path, package, description, embedded_text)
    VALUES ('delete', old.id, old.filename, old.rel_path, old.package, old.description, old.embedded_text);
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, rel_path, package, description, embedded_text)
    VALUES ('delete', old.id, old.filename, old.rel_path, old.package, old.description, old.embedded_text);
    INSERT INTO files_fts(rowid, filename, rel_path, package, description, embedded_text)
    VALUES (new.id, new.filename, new.rel_path, new.package, new.description, new.embedded_text);
END;

CREATE TABLE IF NOT EXISTS waveform_cache (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    mtime_ns INTEGER NOT NULL,
    peaks_blob BLOB NOT NULL,
    points INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS waveform_thumbs (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    mtime_ns INTEGER NOT NULL,
    thumb_blob BLOB NOT NULL,
    spectral_blob BLOB
);

CREATE TABLE IF NOT EXISTS user_meta (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    favorite INTEGER NOT NULL DEFAULT 0,
    rating INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS metadata_fields (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    dirty INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY(file_id, field)
);

CREATE TABLE IF NOT EXISTS write_batches (
    batch_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    file_count INTEGER NOT NULL,
    backup_dir TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rename_batches (
    batch_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    file_count INTEGER NOT NULL,
    template TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
    folder_rel_path TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    added_at REAL NOT NULL,
    UNIQUE(collection_id, root_id, folder_rel_path)
);

CREATE TABLE IF NOT EXISTS collection_files (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    added_at REAL NOT NULL,
    PRIMARY KEY(collection_id, file_id)
);

CREATE TABLE IF NOT EXISTS recent_plays (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    played_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'keyword',
    result_count INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    last_used_at REAL NOT NULL,
    UNIQUE(query, mode)
);

CREATE TABLE IF NOT EXISTS hidden_files (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    reason TEXT NOT NULL DEFAULT 'duplicate',
    hidden_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    group_key TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_members (
    group_id INTEGER NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    is_primary INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(group_id, file_id)
);

CREATE INDEX IF NOT EXISTS idx_folders_root_parent ON folders(root_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_id);
CREATE INDEX IF NOT EXISTS idx_files_root_folder ON files(root_id, folder_id);
CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder_id);
CREATE INDEX IF NOT EXISTS idx_files_root_relpath ON files(root_id, rel_path);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_files_description ON files(description COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_files_duration ON files(duration_sec);
CREATE INDEX IF NOT EXISTS idx_files_sample_rate ON files(sample_rate);
CREATE INDEX IF NOT EXISTS idx_files_channels ON files(channels);
CREATE INDEX IF NOT EXISTS idx_files_bit_depth ON files(bit_depth);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_files_size ON files(size_bytes);
CREATE INDEX IF NOT EXISTS idx_files_package ON files(package COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_user_meta_fav ON user_meta(favorite);
CREATE INDEX IF NOT EXISTS idx_recent_plays_time ON recent_plays(played_at DESC);
CREATE INDEX IF NOT EXISTS idx_coll_files_file ON collection_files(file_id);
CREATE INDEX IF NOT EXISTS idx_coll_folders_coll ON collection_folders(collection_id);
CREATE INDEX IF NOT EXISTS idx_coll_folders_root ON collection_folders(root_id);
CREATE INDEX IF NOT EXISTS idx_coll_folders_folder ON collection_folders(folder_id);
CREATE INDEX IF NOT EXISTS idx_dup_groups_mode ON duplicate_groups(mode);
CREATE INDEX IF NOT EXISTS idx_dup_members_file ON duplicate_members(file_id);
CREATE INDEX IF NOT EXISTS idx_metadata_fields_dirty ON metadata_fields(file_id) WHERE dirty = 1;
"""


def open_connection(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a configured SQLite connection with WAL and memory pragmas."""
    path = Path(db_path)
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA temp_store = MEMORY")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables, trigram FTS5 virtual table, triggers, and indexes if missing."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'files_fts'"
    ).fetchone()
    rebuild_trigram = row is not None and "trigram" not in str(row["sql"] or "").lower()
    if rebuild_trigram:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS files_ai;
            DROP TRIGGER IF EXISTS files_ad;
            DROP TRIGGER IF EXISTS files_au;
            DROP TABLE IF EXISTS files_fts;
            """
        )
    conn.executescript(SCHEMA_SQL)
    if rebuild_trigram:
        conn.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
    user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if user_version < 2:
        conn.execute("DELETE FROM waveform_thumbs")
        conn.execute("PRAGMA user_version = 2")
    cols = {
        str(r["name"])
        for r in conn.execute("PRAGMA table_info(waveform_thumbs)").fetchall()
    }
    if "spectral_blob" not in cols:
        conn.execute("ALTER TABLE waveform_thumbs ADD COLUMN spectral_blob BLOB")
    cf_cols = {
        str(r["name"])
        for r in conn.execute("PRAGMA table_info(collection_folders)").fetchall()
    }
    if "sort_order" not in cf_cols:
        conn.execute(
            "ALTER TABLE collection_folders ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0"
        )
    root_cols = {
        str(r["name"])
        for r in conn.execute("PRAGMA table_info(roots)").fetchall()
    }
    if "sort_order" not in root_cols:
        conn.execute(
            "ALTER TABLE roots ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0"
        )
    if user_version < 3:
        conn.execute("PRAGMA user_version = 3")
    if user_version < 4:
        conn.execute("PRAGMA user_version = 4")
    conn.commit()
