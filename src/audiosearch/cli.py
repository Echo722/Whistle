"""Command-line entry points for AudioSearchTool."""

from __future__ import annotations

import argparse
from pathlib import Path

from audiosearch import __version__
from audiosearch.browser.runtime_paths import default_db_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audiosearch")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    default_db = default_db_path()

    browser_cmd = sub.add_parser("browser", help="启动独立音效浏览器（默认）")
    browser_cmd.add_argument("--db", type=Path, default=default_db)
    browser_cmd.add_argument("--add-root", type=Path, default=None)

    scan_cmd = sub.add_parser("scan", help="扫描并索引本地音效目录")
    scan_cmd.add_argument("root", type=Path)
    scan_cmd.add_argument("--db", type=Path, default=default_db)

    index_cmd = sub.add_parser("index", help="后台构建 AI 文本与声波向量索引")
    index_cmd.add_argument("--db", type=Path, default=default_db)
    index_cmd.add_argument("--profile", choices=["large", "small"], default="large")
    index_cmd.add_argument("--limit", type=int, default=None)
    index_cmd.add_argument("--text-only", action="store_true")
    index_cmd.add_argument("--audio-only", action="store_true")

    args = parser.parse_args(argv)
    if args.command in (None, "browser"):
        from audiosearch.browser.__main__ import launch_browser

        db_path = getattr(args, "db", default_db)
        add_root = getattr(args, "add_root", None)
        return launch_browser(db_path=db_path, add_root=add_root)

    if args.command == "scan":
        from audiosearch.library.scanner import LibraryScanner
        from audiosearch.library.store import LibraryStore

        store = LibraryStore(args.db)
        root = store.add_root(args.root)
        scanner = LibraryScanner(args.db)
        report = scanner.scan_root(root.id)
        print(
            f"scanned={report.scanned} added={report.added} "
            f"updated={report.updated} skipped={report.skipped} errors={report.errors}"
        )
        return 0

    if args.command == "index":
        from audiosearch.ai.indexer import IndexProgress, VectorIndexer

        last_pct = {"val": -1}

        def _on_prog(p: IndexProgress) -> None:
            pct = int(
                round(
                    ((p.text_done + p.audio_done) / max(1, 2 * p.total_files)) * 100.0
                )
            )
            if p.phase in ("done", "cancelled") or pct >= last_pct["val"] + 10:
                last_pct["val"] = pct
                print(
                    f"[{p.phase}] {pct}% text={p.text_done}/{p.total_files} "
                    f"audio={p.audio_done}/{p.total_files} ({p.items_per_sec:.1f}/s) {p.message}"
                )

        indexer = VectorIndexer(args.db, profile=args.profile)
        report = indexer.run(
            limit=args.limit,
            text_only=args.text_only,
            audio_only=args.audio_only,
            on_progress=_on_prog,
        )
        print(
            f"profile={report.profile} status={report.status} total={report.total_files} "
            f"encoded_text={report.encoded_text} reused_text={report.reused_text} "
            f"encoded_audio={report.encoded_audio} reused_audio={report.reused_audio} "
            f"windows={report.audio_windows_added} failed={report.failed} "
            f"pruned_text={report.pruned_text} pruned_audio={report.pruned_audio} "
            f"elapsed={report.elapsed_seconds}s"
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
