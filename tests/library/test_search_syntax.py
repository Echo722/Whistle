"""Behavioral tests for ME search syntax compiled to SQLite FTS5 + SQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audiosearch.library.schema import ensure_schema, open_connection
from audiosearch.library.search_syntax import build_where_clause


def _seed_db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_connection(tmp_path / "lib.sqlite3")
    ensure_schema(conn)
    conn.execute("INSERT INTO roots(path, label, is_online) VALUES ('C:/SFX', 'SFX', 1)")
    rows = [
        (1, None, "Magic/whoosh_fire_01.wav", "whoosh_fire_01.wav", "wav", 100, 1, 1.2, 48000, 2, 24, "Magic", "魔法火焰呼啸声", ""),
        (1, None, "Magic/water_splash_02.wav", "water_splash_02.wav", "wav", 100, 1, 0.8, 48000, 2, 24, "Magic", "水流溅射魔法", ""),
        (1, None, "UI/click_metal_loop.wav", "click_metal_loop.wav", "wav", 100, 1, 0.2, 48000, 1, 16, "UI", "sharp metal click loop", ""),
        (1, None, "Impact/explosion_big.flac", "explosion_big.flac", "flac", 100, 1, 2.5, 96000, 2, 24, "Impact", "heavy sub boom explosion", ""),
        (1, None, "Misc/8BitRetroSFXPack1_Contemporary_Misc06.wav", "8BitRetroSFXPack1_Contemporary_Misc06.wav", "wav", 100, 1, 0.6, 96000, 2, 24, "8BitRetro", "", ""),
    ]
    conn.executemany(
        """
        INSERT INTO files(
            root_id, folder_id, rel_path, filename, ext, size_bytes, mtime_ns,
            duration_sec, sample_rate, channels, bit_depth, package, description, embedded_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn


def _run_query(conn: sqlite3.Connection, query: str) -> list[str]:
    join_fts, clauses, params = build_where_clause(query, alias="f")
    sql = "SELECT f.filename FROM files f"
    if join_fts:
        sql += " JOIN files_fts ON files_fts.rowid = f.id"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY f.filename"
    return [str(r[0]) for r in conn.execute(sql, params).fetchall()]


def test_and_or_not_phrase_and_wildcards(tmp_path):
    conn = _seed_db(tmp_path)
    try:
        assert _run_query(conn, "whoosh fire") == ["whoosh_fire_01.wav"]
        assert _run_query(conn, "fire OR water") == ["water_splash_02.wav", "whoosh_fire_01.wav"]
        assert _run_query(conn, "metal -loop") == []
        assert _run_query(conn, "metal NOT click") == []
        assert _run_query(conn, "-loop") == [
            "8BitRetroSFXPack1_Contemporary_Misc06.wav",
            "explosion_big.flac",
            "water_splash_02.wav",
            "whoosh_fire_01.wav",
        ]
        assert _run_query(conn, '"heavy sub"') == ["explosion_big.flac"]
        assert _run_query(conn, "expl*") == ["explosion_big.flac"]
        assert _run_query(conn, "*splash*") == ["water_splash_02.wav"]
    finally:
        conn.close()


def test_substring_matching_and_legacy_fts_migration(tmp_path):
    conn = _seed_db(tmp_path)
    try:
        # Substrings inside concatenated filenames (ME behavior)
        assert _run_query(conn, "06") == ["8BitRetroSFXPack1_Contemporary_Misc06.wav"]
        assert _run_query(conn, "Retro") == ["8BitRetroSFXPack1_Contemporary_Misc06.wav"]
        assert _run_query(conn, "Misc06") == ["8BitRetroSFXPack1_Contemporary_Misc06.wav"]
        assert _run_query(conn, "火焰") == ["whoosh_fire_01.wav"]
        assert _run_query(conn, "魔法 -火焰") == ["water_splash_02.wav"]
        assert _run_query(conn, "whoosh_fire_01.wav") == ["whoosh_fire_01.wav"]
        assert _run_query(conn, "((:invalid--fts))") == []

        # Simulate legacy unicode61 FTS table and verify ensure_schema rebuilds it to trigram.
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS files_ai;
            DROP TRIGGER IF EXISTS files_ad;
            DROP TRIGGER IF EXISTS files_au;
            DROP TABLE IF EXISTS files_fts;
            CREATE VIRTUAL TABLE files_fts USING fts5(
                filename, rel_path, package, description, embedded_text,
                content='files', content_rowid='id', tokenize='unicode61'
            );
            """
        )
        ensure_schema(conn)
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'files_fts'"
        ).fetchone()
        assert sql_row is not None and "trigram" in sql_row[0].lower()
        assert _run_query(conn, "Retro") == ["8BitRetroSFXPack1_Contemporary_Misc06.wav"]
    finally:
        conn.close()


def test_relative_backslash_path_fragments(tmp_path):
    conn = open_connection(tmp_path / "rel_slash.sqlite3")
    ensure_schema(conn)
    try:
        conn.execute("INSERT INTO roots(path, label, is_online) VALUES ('F:/Sound Effects', 'SFX', 1)")
        rows = [
            (1, None, "2496 Sound Effects/Raw Noise1/Automotive Noise Collection 1.wav", "Automotive Noise Collection 1.wav", "wav", 100, 1, 1.0, 48000, 2, 24, "2496", "", ""),
            (1, None, "科幻/激光/Laser_Beam_01.wav", "Laser_Beam_01.wav", "wav", 100, 1, 1.0, 48000, 2, 24, "SciFi", "", ""),
            (1, None, "ui/btn.wav", "btn.wav", "wav", 100, 1, 0.3, 48000, 1, 16, "UI", "", ""),
        ]
        conn.executemany(
            """
            INSERT INTO files(
                root_id, folder_id, rel_path, filename, ext, size_bytes, mtime_ns,
                duration_sec, sample_rate, channels, bit_depth, package, description, embedded_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

        assert _run_query(conn, r"Raw Noise1\Automotive") == ["Automotive Noise Collection 1.wav"]
        assert _run_query(conn, r"科幻\激光") == ["Laser_Beam_01.wav"]
        assert _run_query(conn, "ui\\") == ["btn.wav"]
    finally:
        conn.close()
