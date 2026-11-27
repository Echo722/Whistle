"""Virtualized QTableView + paged SQLite model replicating ME columns and behavior."""

from __future__ import annotations

import array
import ctypes
import os
import subprocess
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QMimeData,
    QModelIndex,
    QObject,
    QPoint,
    QPointF,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QGuiApplication,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QWidget,
)

from audiosearch.browser import theme
from audiosearch.browser.file_drag import start_file_drag, start_files_drag
from audiosearch.browser.formatting import (
    format_bit_depth,
    format_channels,
    format_duration,
    format_sample_rate,
    format_size,
)
from audiosearch.browser.waveform_delegate import (
    ThumbnailManager,
    WaveformItemDelegate,
    paint_selection_chrome,
)
from audiosearch.library.store import FileRecord, FolderRecord, LibraryStore, QueryFilter

PAGE_SIZE = 200
MAX_CACHED_PAGES = 64
RAM_INDEX_MIN_ROWS = 2_000
RAM_INDEX_MAX_CACHED_VIEWS = 3

WAVEFORM_COL_KEY = "waveform"
WAVEFORM_COL = 1
FAV_COL_KEY = "favorite"
FAV_COL = 2
MARK_COL_KEY = "mark"
MARK_COL = 3
NON_SORTABLE_COL_KEYS = frozenset({WAVEFORM_COL_KEY, MARK_COL_KEY})

# Default columns modeled on REAPER Media Explorer + Soundminer waveform column.
COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("filename", "名称", 230),
    ("waveform", "波形", 150),
    ("favorite", "★", 34),
    ("mark", "Mark", 44),
    ("relevance", "相关度", 68),
    ("hit_reason", "命中说明", 156),
    ("description", "描述", 220),
    ("duration_sec", "时长", 72),
    ("sample_rate", "采样率", 66),
    ("channels", "声道", 54),
    ("bit_depth", "位深", 54),
    ("ext", "格式", 56),
    ("size_bytes", "大小", 76),
    ("package", "素材包", 120),
    ("rel_path", "路径", 260),
)
DEFAULT_COLUMN_ORDER: tuple[str, ...] = (
    MARK_COL_KEY,
    *(key for key, _, _ in COLUMNS if key != MARK_COL_KEY),
)
AI_COL_KEYS = frozenset({"relevance", "hit_reason"})
NUMERIC_COL_KEYS = frozenset(
    {"duration_sec", "sample_rate", "channels", "bit_depth", "size_bytes"}
)
NUMERIC_COLS = {idx for idx, (key, _, _) in enumerate(COLUMNS) if key in NUMERIC_COL_KEYS}


_INLINE_EDITABLE_COL_MAP: dict[str, str] = {
    "description": "description",
    "package": "library",
}


class _PageLoaderSignals(QObject):
    pageLoaded = Signal(int, int, list)  # (snapshot_gen, page_idx, list[FileRecord])


class _PageLoader:
    """Single background worker thread loading virtual table pages without blocking UI."""

    def __init__(self, store: LibraryStore, signals: _PageLoaderSignals) -> None:
        self._store = store
        self._signals = signals
        self._cond = threading.Condition()
        self._queue: deque[tuple[int, int, dict[str, Any]]] = deque()
        self._latest_gen = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="FileTablePageLoader", daemon=True
            )
            self._thread.start()

    def submit(
        self, gen: int, page_idx: int, params: dict[str, Any]
    ) -> list[tuple[int, int]]:
        evicted: list[tuple[int, int]] = []
        with self._cond:
            if self._stop or gen < self._latest_gen:
                return evicted
            self._queue = deque(
                item for item in self._queue if not (item[0] == gen and item[1] == page_idx)
            )
            self._queue.appendleft((gen, page_idx, params))
            while len(self._queue) > 4:
                ev_gen, ev_page, _ = self._queue.pop()
                evicted.append((ev_gen, ev_page))
            self._ensure_thread_locked()
            self._cond.notify()
        return evicted

    def clear_pending(self, gen: int) -> list[int]:
        dropped_pages: list[int] = []
        with self._cond:
            kept: deque[tuple[int, int, dict[str, Any]]] = deque()
            for item in self._queue:
                if item[0] == gen:
                    dropped_pages.append(item[1])
                else:
                    kept.append(item)
            self._queue = kept
        return dropped_pages

    def invalidate(self, gen: int) -> None:
        with self._cond:
            self._latest_gen = max(self._latest_gen, gen)
            self._queue.clear()

    def shutdown(self, timeout: float = 1.5) -> None:
        with self._cond:
            thread = self._thread
            self._stop = True
            self._queue.clear()
            self._cond.notify_all()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                gen, page_idx, params = self._queue.popleft()
                if gen < self._latest_gen:
                    continue
            try:
                if "page_ids" in params:
                    records = self._store.query_page_by_ids(params["page_ids"])
                else:
                    records = self._store.query_page(**params)
            except Exception:
                records = []
            with self._cond:
                stale = self._stop or gen < self._latest_gen
            if not stale:
                try:
                    self._signals.pageLoaded.emit(gen, page_idx, records)
                except RuntimeError:
                    return


class _IdIndexSignals(QObject):
    indexReady = Signal(int, object, object)  # (snapshot_gen, cache_key, array.array)


class _IdIndexBuilder:
    """Builds a compact int64 sorted ID array in the background for ~1ms random-access scrolling."""

    def __init__(self, store: LibraryStore, signals: _IdIndexSignals) -> None:
        self._store = store
        self._signals = signals
        self._cond = threading.Condition()
        self._pending: tuple[int, tuple[Any, ...], dict[str, Any]] | None = None
        self._active_cancel: threading.Event | None = None
        self._latest_gen = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="FileTableIdIndexBuilder", daemon=True
            )
            self._thread.start()

    def submit(self, gen: int, cache_key: tuple[Any, ...], params: dict[str, Any]) -> None:
        with self._cond:
            if self._stop or gen < self._latest_gen:
                return
            self._latest_gen = max(self._latest_gen, gen)
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._pending = (gen, cache_key, params)
            self._ensure_thread_locked()
            self._cond.notify()

    def invalidate(self, gen: int) -> None:
        with self._cond:
            self._latest_gen = max(self._latest_gen, gen)
            self._pending = None
            if self._active_cancel is not None:
                self._active_cancel.set()

    def shutdown(self, timeout: float = 1.5) -> None:
        with self._cond:
            thread = self._thread
            self._stop = True
            self._pending = None
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._cond.notify_all()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                gen, cache_key, params = self._pending
                self._pending = None
                if gen < self._latest_gen:
                    continue
                cancel_ev = threading.Event()
                self._active_cancel = cancel_ev
            try:
                ids = self._store.query_sorted_id_array(**params, cancel_event=cancel_ev)
            except Exception:
                ids = None
            with self._cond:
                if self._active_cancel is cancel_ev:
                    self._active_cancel = None
                stale = self._stop or gen < self._latest_gen or cancel_ev.is_set()
            if not stale and ids is not None:
                try:
                    self._signals.indexReady.emit(gen, cache_key, ids)
                except RuntimeError:
                    return


class _FileRowDelegate(QStyledItemDelegate):
    """Paints standard table cells without focus rectangle plus unified selection chrome."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, opt, index)
        paint_selection_chrome(painter, opt, index)


class FileTableModel(QAbstractTableModel):
    """Paged virtual model backed directly by LibraryStore."""

    sortChanged = Signal(str, bool)  # (sort_column_key, ascending)
    inlineMetadataEdited = Signal(int, str, str)  # (file_id, canonical_key, new_value)
    pageReady = Signal(int)  # page_idx
    idIndexReady = Signal()

    def __init__(
        self,
        store: LibraryStore,
        thumbnail_manager: ThumbnailManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._thumbnail_manager = thumbnail_manager
        self._total_count = 0
        self._file_total_count = 0
        self._query = ""
        self._root_id: int | None = None
        self._folder_id: int | None = None
        self._folder_rel_path: str = ""
        self._include_subfolders = True
        self._filters: QueryFilter | None = None
        self._sort_column = "filename"
        self._sort_asc = True
        self._pages: OrderedDict[int, list[FileRecord]] = OrderedDict()
        self._snapshot_gen = 0
        self._in_flight_pages: set[int] = set()
        self._scroll_throttled = False
        self._page_signals = _PageLoaderSignals(self)
        self._page_signals.pageLoaded.connect(self._on_page_loaded)
        self._loader = _PageLoader(store, self._page_signals)
        self._sorted_ids: array.array | None = None
        self._id_cache: OrderedDict[tuple[Any, ...], array.array] = OrderedDict()
        self._id_signals = _IdIndexSignals(self)
        self._id_signals.indexReady.connect(self._on_id_index_ready)
        self._id_builder = _IdIndexBuilder(store, self._id_signals)
        self._mono = theme.mono_font(theme.ui_scale_spec().mono_pt)
        self._playing_file_id: int | None = None
        self._played_mark_ids: set[int] = set()
        self._view_key: tuple[Any, ...] | None = None
        self._ai_mode = False
        self._dup_mode = False
        self._folders: list[tuple[FolderRecord, str, bool]] = []
        self._ai_records: list[FileRecord] = []
        self._ai_hits: dict[int, Any] = {}
        self._dup_rows: dict[int, Any] = {}
        self._icon_folder_online: QIcon | None = None
        self._icon_folder_offline: QIcon | None = None

    @staticmethod
    def _build_folder_icon(color: QColor) -> QIcon:
        pm = QPixmap(14, 14)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color, 1.2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF(
            [
                QPointF(1.5, 3.0),
                QPointF(5.4, 3.0),
                QPointF(6.6, 4.4),
                QPointF(12.5, 4.4),
                QPointF(12.5, 11.2),
                QPointF(1.5, 11.2),
            ]
        )
        p.drawPolygon(poly)
        p.end()
        return QIcon(pm)

    def _folder_icon(self, is_online: bool) -> QIcon:
        if is_online:
            if self._icon_folder_online is None:
                self._icon_folder_online = self._build_folder_icon(QColor("#9E998E"))
            return self._icon_folder_online
        if self._icon_folder_offline is None:
            self._icon_folder_offline = self._build_folder_icon(theme.OFFLINE)
        return self._icon_folder_offline

    @property
    def folder_count(self) -> int:
        return 0 if (self._ai_mode or self._dup_mode) else len(self._folders)

    def folder_at(self, row: int) -> tuple[FolderRecord, str, bool] | None:
        if self._ai_mode or self._dup_mode:
            return None
        if 0 <= row < len(self._folders):
            return self._folders[row]
        return None

    def is_folder_row(self, row: int) -> bool:
        return self.folder_at(row) is not None

    @property
    def sort_column(self) -> str:
        return self._sort_column

    @property
    def sort_asc(self) -> bool:
        return self._sort_asc

    @property
    def is_ai_mode(self) -> bool:
        return self._ai_mode

    @property
    def is_duplicate_mode(self) -> bool:
        return self._dup_mode

    def apply_ui_scale(self, scale_key: str) -> None:
        spec = theme.ui_scale_spec(scale_key)
        self._mono = theme.mono_font(spec.mono_pt)
        if self._total_count > 0 and len(COLUMNS) > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self._total_count - 1, len(COLUMNS) - 1),
                [Qt.ItemDataRole.FontRole],
            )
            self.layoutChanged.emit()

    def ai_hit_for(self, file_id: int) -> Any | None:
        if not self._ai_mode:
            return None
        return self._ai_hits.get(file_id)

    def duplicate_row_for(self, file_id: int) -> Any | None:
        if not self._dup_mode:
            return None
        return self._dup_rows.get(file_id)

    def apply_snapshot(
        self,
        *,
        total_count: int,
        first_page: list[FileRecord],
        query: str,
        root_id: int | None,
        folder_id: int | None,
        folder_rel_path: str,
        include_subfolders: bool,
        sort_column: str,
        sort_asc: bool,
        filters: QueryFilter | None = None,
        folders: list[tuple[FolderRecord, str, bool]] | None = None,
        extra_pages: dict[int, list[FileRecord]] | None = None,
    ) -> None:
        new_view_key = (
            query,
            root_id,
            folder_id,
            folder_rel_path,
            include_subfolders,
            filters,
        )
        same_view = (
            not self._ai_mode
            and not self._dup_mode
            and self._view_key == new_view_key
        )
        self.beginResetModel()
        if not same_view:
            self._played_mark_ids.clear()
        self._snapshot_gen += 1
        self._in_flight_pages.clear()
        self._loader.invalidate(self._snapshot_gen)
        self._id_builder.invalidate(self._snapshot_gen)
        self._sorted_ids = None
        self._view_key = new_view_key
        self._ai_mode = False
        self._dup_mode = False
        self._ai_records.clear()
        self._ai_hits.clear()
        self._dup_rows.clear()
        self._folders = list(folders) if folders else []
        self._file_total_count = max(0, int(total_count))
        self._total_count = len(self._folders) + self._file_total_count
        self._query = query
        self._root_id = root_id
        self._folder_id = folder_id
        self._folder_rel_path = folder_rel_path
        self._include_subfolders = include_subfolders
        self._filters = filters
        self._sort_column = sort_column
        self._sort_asc = sort_asc
        self._pages.clear()
        if first_page:
            self._pages[0] = first_page
        if extra_pages:
            for page_idx, page_recs in extra_pages.items():
                if page_recs:
                    self._pages[int(page_idx)] = page_recs
        recent_filter = bool(filters is not None and filters.recent_only)
        if (
            self._file_total_count >= RAM_INDEX_MIN_ROWS
            and sort_column != "played_at"
            and not recent_filter
        ):
            cache_key = (
                new_view_key,
                sort_column,
                bool(sort_asc),
                self._file_total_count,
                self._store.index_version,
            )
            cached_ids = self._id_cache.get(cache_key)
            if cached_ids is not None and len(cached_ids) == self._file_total_count:
                self._id_cache.move_to_end(cache_key)
                self._sorted_ids = cached_ids
            else:
                self._id_builder.submit(
                    self._snapshot_gen,
                    cache_key,
                    {
                        "query": query,
                        "root_id": root_id,
                        "folder_id": folder_id,
                        "folder_rel_path": folder_rel_path,
                        "include_subfolders": include_subfolders,
                        "sort_column": sort_column,
                        "sort_asc": sort_asc,
                        "filters": filters,
                    },
                )
        self.endResetModel()
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(COLUMNS) - 1)

    def apply_ai_results(
        self,
        records: list[FileRecord],
        hits: dict[int, Any],
        *,
        sort_column: str = "relevance",
        sort_asc: bool = False,
    ) -> None:
        self.beginResetModel()
        self._played_mark_ids.clear()
        self._snapshot_gen += 1
        self._in_flight_pages.clear()
        self._loader.invalidate(self._snapshot_gen)
        self._id_builder.invalidate(self._snapshot_gen)
        self._sorted_ids = None
        self._view_key = None
        self._ai_mode = True
        self._dup_mode = False
        self._folders.clear()
        self._file_total_count = 0
        self._dup_rows.clear()
        self._pages.clear()
        self._ai_hits = dict(hits)
        self._ai_records = list(records)
        self._sort_column = sort_column
        self._sort_asc = sort_asc
        self._sort_ai_list(sort_column, sort_asc)
        self._total_count = len(self._ai_records)
        self.endResetModel()
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(COLUMNS) - 1)

    def apply_duplicate_results(
        self, dup_rows: list[Any], *, reset_marks: bool = False
    ) -> None:
        self.beginResetModel()
        if not self._dup_mode or reset_marks:
            self._played_mark_ids.clear()
        self._snapshot_gen += 1
        self._in_flight_pages.clear()
        self._loader.invalidate(self._snapshot_gen)
        self._id_builder.invalidate(self._snapshot_gen)
        self._sorted_ids = None
        self._view_key = None
        self._ai_mode = False
        self._dup_mode = True
        self._folders.clear()
        self._file_total_count = 0
        self._ai_hits.clear()
        self._pages.clear()
        self._dup_rows = {int(d.file.id): d for d in dup_rows}
        self._ai_records = [d.file for d in dup_rows]
        self._sort_column = "hit_reason"
        self._sort_asc = True
        self._total_count = len(self._ai_records)
        self.endResetModel()
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(COLUMNS) - 1)

    @property
    def has_ram_index(self) -> bool:
        return self._sorted_ids is not None

    def invalidate_id_cache(self) -> None:
        self._id_cache.clear()
        self._sorted_ids = None

    def clear_pending_pages(self) -> None:
        for p in self._loader.clear_pending(self._snapshot_gen):
            self._in_flight_pages.discard(p)

    def _on_id_index_ready(
        self, gen: int, cache_key: tuple[Any, ...], ids: array.array
    ) -> None:
        self._id_cache[cache_key] = ids
        self._id_cache.move_to_end(cache_key)
        while len(self._id_cache) > RAM_INDEX_MAX_CACHED_VIEWS:
            self._id_cache.popitem(last=False)
        if gen == self._snapshot_gen and len(ids) == self._file_total_count:
            self._sorted_ids = ids
            self.idIndexReady.emit()

    def set_scroll_throttled(self, throttled: bool) -> None:
        self._scroll_throttled = bool(throttled)

    def request_page_async(self, page_idx: int, *, force: bool = False) -> None:
        if self._ai_mode or self._dup_mode or page_idx < 0:
            return
        if page_idx * PAGE_SIZE >= self._file_total_count:
            return
        if self._scroll_throttled and not force:
            return
        if page_idx in self._pages:
            self._pages.move_to_end(page_idx)
            return
        if page_idx in self._in_flight_pages:
            return
        self._in_flight_pages.add(page_idx)
        offset = page_idx * PAGE_SIZE
        if self._sorted_ids is not None:
            params: dict[str, Any] = {
                "page_ids": self._sorted_ids[offset : offset + PAGE_SIZE].tolist()
            }
        else:
            params = {
                "offset": offset,
                "limit": PAGE_SIZE,
                "query": self._query,
                "root_id": self._root_id,
                "folder_id": self._folder_id,
                "folder_rel_path": self._folder_rel_path,
                "include_subfolders": self._include_subfolders,
                "sort_column": self._sort_column,
                "sort_asc": self._sort_asc,
                "filters": self._filters,
                "total_count": self._file_total_count,
            }
        evicted = self._loader.submit(self._snapshot_gen, page_idx, params)
        for ev_gen, ev_page in evicted:
            if ev_gen == self._snapshot_gen:
                self._in_flight_pages.discard(ev_page)

    def _on_page_loaded(self, gen: int, page_idx: int, records: list[FileRecord]) -> None:
        if gen != self._snapshot_gen:
            return
        self._in_flight_pages.discard(page_idx)
        self._pages[page_idx] = records
        self._pages.move_to_end(page_idx)
        while len(self._pages) > MAX_CACHED_PAGES:
            self._pages.popitem(last=False)
        start_row = len(self._folders) + page_idx * PAGE_SIZE
        end_row = min(self._total_count - 1, start_row + max(len(records), PAGE_SIZE) - 1)
        if 0 <= start_row <= end_row:
            self.dataChanged.emit(
                self.index(start_row, 0),
                self.index(end_row, len(COLUMNS) - 1),
            )
        self.pageReady.emit(page_idx)

    def shutdown(self) -> None:
        self._id_builder.shutdown()
        self._loader.shutdown()

    def sort_ai_records(self, sort_column: str, ascending: bool) -> None:
        if not self._ai_mode and not self._dup_mode:
            return
        self.beginResetModel()
        self._sort_column = sort_column
        self._sort_asc = ascending
        self._sort_ai_list(sort_column, ascending)
        self.endResetModel()

    def _sort_ai_list(self, sort_column: str, ascending: bool) -> None:
        if not self._ai_records:
            return
        if self._dup_mode and sort_column in ("relevance", "hit_reason"):
            self._ai_records.sort(
                key=lambda r: (
                    int(getattr(self._dup_rows.get(r.id), "group_index", 0)),
                    0 if getattr(self._dup_rows.get(r.id), "is_primary", False) else 1,
                    r.id,
                ),
                reverse=not ascending,
            )
            return
        if sort_column == "relevance":
            self._ai_records.sort(
                key=lambda r: (
                    float(getattr(self._ai_hits.get(r.id), "score", 0.0)),
                    -int(getattr(self._ai_hits.get(r.id), "rank", 0)),
                ),
                reverse=not ascending,
            )
            return
        if sort_column == "hit_reason":
            self._ai_records.sort(
                key=lambda r: str(getattr(self._ai_hits.get(r.id), "hit_reason", "")).casefold(),
                reverse=not ascending,
            )
            return
        attr_map = {
            "filename": lambda r: r.filename.casefold(),
            "favorite": lambda r: 1 if r.favorite else 0,
            "description": lambda r: r.description.casefold(),
            "duration_sec": lambda r: r.duration_sec,
            "sample_rate": lambda r: r.sample_rate,
            "channels": lambda r: r.channels,
            "bit_depth": lambda r: r.bit_depth,
            "ext": lambda r: r.ext.casefold(),
            "size_bytes": lambda r: r.size_bytes,
            "package": lambda r: r.package.casefold(),
            "rel_path": lambda r: r.rel_path.casefold(),
        }
        key_fn = attr_map.get(sort_column, attr_map["filename"])
        self._ai_records.sort(key=key_fn, reverse=not ascending)

    def refresh_file_meta(self, file_ids: set[int]) -> None:
        if not file_ids:
            return
        fresh = self._store.get_files_by_ids(list(file_ids))
        if not fresh:
            return
        if self._ai_mode or self._dup_mode:
            for row, rec in enumerate(self._ai_records):
                if rec.id in fresh:
                    self._ai_records[row] = fresh[rec.id]
                    if rec.id in self._dup_rows:
                        old_dup = self._dup_rows[rec.id]
                        from dataclasses import replace

                        self._dup_rows[rec.id] = replace(old_dup, file=fresh[rec.id])
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, len(COLUMNS) - 1),
                    )
            return
        folder_offset = len(self._folders)
        for page_idx, page in self._pages.items():
            base = folder_offset + page_idx * PAGE_SIZE
            for offset, rec in enumerate(page):
                if rec.id in fresh:
                    page[offset] = fresh[rec.id]
                    row = base + offset
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, len(COLUMNS) - 1),
                    )

    def is_played_marked(self, file_id: int) -> bool:
        return file_id in self._played_mark_ids

    def played_mark_count(self) -> int:
        return len(self._played_mark_ids)

    def _emit_mark_changed_for_ids(self, target_ids: set[int]) -> None:
        if not target_ids:
            return
        roles = [
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ForegroundRole,
        ]
        if self._ai_mode or self._dup_mode:
            for row, rec in enumerate(self._ai_records):
                if rec.id in target_ids:
                    idx = self.index(row, MARK_COL)
                    self.dataChanged.emit(idx, idx, roles)
            return
        folder_offset = len(self._folders)
        for page_idx, page in self._pages.items():
            base = folder_offset + page_idx * PAGE_SIZE
            for offset, rec in enumerate(page):
                if rec.id in target_ids:
                    idx = self.index(base + offset, MARK_COL)
                    self.dataChanged.emit(idx, idx, roles)

    def mark_played(self, file_id: int, marked: bool = True) -> bool:
        fid = int(file_id)
        if marked:
            if fid in self._played_mark_ids:
                return False
            self._played_mark_ids.add(fid)
        else:
            if fid not in self._played_mark_ids:
                return False
            self._played_mark_ids.discard(fid)
        self._emit_mark_changed_for_ids({fid})
        return True

    def mark_played_many(self, file_ids: list[int], marked: bool = True) -> int:
        changed: set[int] = set()
        for raw_id in file_ids:
            fid = int(raw_id)
            if marked:
                if fid not in self._played_mark_ids:
                    self._played_mark_ids.add(fid)
                    changed.add(fid)
            else:
                if fid in self._played_mark_ids:
                    self._played_mark_ids.discard(fid)
                    changed.add(fid)
        self._emit_mark_changed_for_ids(changed)
        return len(changed)

    def clear_played_marks(self) -> int:
        count = len(self._played_mark_ids)
        if count == 0:
            return 0
        self._played_mark_ids.clear()
        if self._total_count > 0:
            self.dataChanged.emit(
                self.index(0, MARK_COL),
                self.index(self._total_count - 1, MARK_COL),
                [
                    Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.ForegroundRole,
                ],
            )
        return count

    def set_playing_file(self, file_id: int | None) -> None:
        old = self._playing_file_id
        self._playing_file_id = file_id
        if old == file_id:
            return
        targets = {x for x in (old, file_id) if x is not None}
        if not targets:
            return
        if self._ai_mode or self._dup_mode:
            for row, rec in enumerate(self._ai_records):
                if rec.id in targets:
                    idx0 = self.index(row, 0)
                    idx_mark = self.index(row, MARK_COL)
                    self.dataChanged.emit(idx0, idx0, [Qt.ItemDataRole.ForegroundRole])
                    self.dataChanged.emit(idx_mark, idx_mark, [Qt.ItemDataRole.ForegroundRole])
            return
        folder_offset = len(self._folders)
        for page_idx, page in self._pages.items():
            base = folder_offset + page_idx * PAGE_SIZE
            for offset, rec in enumerate(page):
                if rec.id in targets:
                    row = base + offset
                    idx0 = self.index(row, 0)
                    idx_mark = self.index(row, MARK_COL)
                    self.dataChanged.emit(idx0, idx0, [Qt.ItemDataRole.ForegroundRole])
                    self.dataChanged.emit(idx_mark, idx_mark, [Qt.ItemDataRole.ForegroundRole])

    def thumbnail_for(self, rec: FileRecord) -> bytes | None:
        if rec.thumb_blob is not None:
            return rec.thumb_blob
        if self._thumbnail_manager is not None:
            return self._thumbnail_manager.get_cached(rec.id, rec.mtime_ns)
        return None

    def spectral_thumbnail_for(self, rec: FileRecord) -> bytes | None:
        if rec.spectral_thumb_blob is not None:
            return rec.spectral_thumb_blob
        if self._thumbnail_manager is not None:
            return self._thumbnail_manager.get_cached_spectral(rec.id, rec.mtime_ns)
        return None

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return self._total_count

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(COLUMNS)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(COLUMNS):
            key, title, _ = COLUMNS[section]
            if role == Qt.ItemDataRole.DisplayRole:
                if self._dup_mode:
                    if key == "relevance":
                        return "重复状态"
                    if key == "hit_reason":
                        return "重复分组"
                return title
            if role == Qt.ItemDataRole.ToolTipRole and key == MARK_COL_KEY:
                return "当前展示中已试听标记（•），切换目录/搜索自动清零，右键可手动切换或清空"
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:  # noqa: N802
        if not 0 <= column < len(COLUMNS):
            return
        col_key = COLUMNS[column][0]
        if col_key in NON_SORTABLE_COL_KEYS:
            return
        asc = order == Qt.SortOrder.AscendingOrder
        self.sortChanged.emit(col_key, asc)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        if not index.isValid() or self.is_folder_row(index.row()):
            return base
        col = index.column()
        extra = Qt.ItemFlag.ItemIsDragEnabled
        if 0 <= col < len(COLUMNS) and COLUMNS[col][0] in _INLINE_EDITABLE_COL_MAP:
            extra |= Qt.ItemFlag.ItemIsEditable
        return base | extra

    def setData(  # noqa: N802
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role != Qt.ItemDataRole.EditRole or not index.isValid() or self.is_folder_row(index.row()):
            return False
        col = index.column()
        if not 0 <= col < len(COLUMNS):
            return False
        canonical_key = _INLINE_EDITABLE_COL_MAP.get(COLUMNS[col][0])
        if canonical_key is None:
            return False
        rec = self.record_at(index.row())
        if rec is None:
            return False
        self.inlineMetadataEdited.emit(rec.id, canonical_key, str(value or ""))
        return True

    def supportedDragActions(self) -> Qt.DropAction:  # noqa: N802
        return Qt.DropAction.CopyAction | Qt.DropAction.MoveAction | Qt.DropAction.LinkAction

    def mimeTypes(self) -> list[str]:  # noqa: N802
        return ["text/uri-list"]

    def mimeData(self, indexes: list[QModelIndex]) -> QMimeData:  # noqa: N802
        mime = QMimeData()
        seen_rows: set[int] = set()
        urls: list[QUrl] = []
        for idx in indexes:
            if not idx.isValid() or idx.row() in seen_rows:
                continue
            seen_rows.add(idx.row())
            rec = self.record_at(idx.row())
            if rec is not None and rec.is_available:
                resolved = rec.full_path.resolve()
                if resolved.exists():
                    urls.append(QUrl.fromLocalFile(str(resolved)))
        if urls:
            mime.setUrls(urls)
        return mime

    def peek_record_at(self, row: int) -> FileRecord | None:
        if not 0 <= row < self._total_count:
            return None
        if self._ai_mode or self._dup_mode:
            return self._ai_records[row] if row < len(self._ai_records) else None
        file_row = row - len(self._folders)
        if file_row < 0:
            return None
        page = self._pages.get(file_row // PAGE_SIZE)
        if page is None:
            return None
        in_page = file_row % PAGE_SIZE
        return page[in_page] if in_page < len(page) else None

    def record_at(self, row: int) -> FileRecord | None:
        if not 0 <= row < self._total_count:
            return None
        if self._ai_mode or self._dup_mode:
            return self._ai_records[row] if row < len(self._ai_records) else None
        file_row = row - len(self._folders)
        if file_row < 0:
            return None
        page_idx = file_row // PAGE_SIZE
        in_page = file_row % PAGE_SIZE
        page = self._pages.get(page_idx)
        if page is None:
            offset = page_idx * PAGE_SIZE
            if self._sorted_ids is not None:
                page = self._store.query_page_by_ids(
                    self._sorted_ids[offset : offset + PAGE_SIZE].tolist()
                )
            else:
                page = self._store.query_page(
                    offset=offset,
                    limit=PAGE_SIZE,
                    query=self._query,
                    root_id=self._root_id,
                    folder_id=self._folder_id,
                    folder_rel_path=self._folder_rel_path,
                    include_subfolders=self._include_subfolders,
                    sort_column=self._sort_column,
                    sort_asc=self._sort_asc,
                    filters=self._filters,
                    total_count=self._file_total_count,
                )
            self._pages[page_idx] = page
            while len(self._pages) > MAX_CACHED_PAGES:
                self._pages.popitem(last=False)
        else:
            self._pages.move_to_end(page_idx)
        if in_page >= len(page):
            return None
        return page[in_page]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        col = index.column()
        if not 0 <= col < len(COLUMNS):
            return None
        col_key = COLUMNS[col][0]

        folder_entry = self.folder_at(index.row())
        if folder_entry is not None:
            folder, full_path, is_online = folder_entry
            if role == Qt.ItemDataRole.DecorationRole and col_key == "filename":
                return self._folder_icon(is_online)
            if role == Qt.ItemDataRole.DisplayRole:
                if col_key == "filename":
                    return folder.name
                if col_key == "description":
                    return f"文件夹 · {folder.file_count:,} 条"
                if col_key == "ext":
                    return "文件夹"
                if col_key == "rel_path":
                    return folder.rel_path
                return None
            if role == Qt.ItemDataRole.ForegroundRole:
                if not is_online:
                    return theme.OFFLINE
                if col_key == "filename":
                    return QColor("#ECE9E2")
                if col_key in ("description", "ext", "rel_path"):
                    return theme.TEXT_DIM
                return None
            return None

        rec = self.peek_record_at(index.row())
        if rec is None:
            if not self._ai_mode and not self._dup_mode:
                file_row = index.row() - len(self._folders)
                if 0 <= file_row < self._file_total_count:
                    self.request_page_async(file_row // PAGE_SIZE)
                    if role == Qt.ItemDataRole.DisplayRole and col_key == "filename":
                        return "加载中…"
                    if role == Qt.ItemDataRole.ForegroundRole and col_key == "filename":
                        return theme.TEXT_FAINT
            return None

        if role == Qt.ItemDataRole.EditRole:
            if col_key == "description":
                return rec.description
            if col_key == "package":
                return rec.package
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            if col_key == "filename":
                return rec.filename
            if col_key == WAVEFORM_COL_KEY:
                return None
            if col_key == FAV_COL_KEY:
                return "★" if rec.favorite else "☆"
            if col_key == MARK_COL_KEY:
                return "•" if rec.id in self._played_mark_ids else ""
            if col_key == "relevance":
                if self._dup_mode:
                    dup = self._dup_rows.get(rec.id)
                    if dup is None:
                        return "—"
                    if dup.file.is_hidden:
                        return "已隐藏"
                    return "主版本" if dup.is_primary else "副本"
                hit = self.ai_hit_for(rec.id)
                return f"{hit.relevance_pct}%" if hit is not None else "—"
            if col_key == "hit_reason":
                if self._dup_mode:
                    dup = self._dup_rows.get(rec.id)
                    return f"第 {dup.group_index} 组 · {dup.group_size} 条" if dup is not None else "—"
                hit = self.ai_hit_for(rec.id)
                return hit.hit_reason if hit is not None else "—"
            if col_key == "description":
                return rec.description or "—"
            if col_key == "duration_sec":
                return format_duration(rec.duration_sec)
            if col_key == "sample_rate":
                return format_sample_rate(rec.sample_rate)
            if col_key == "channels":
                return format_channels(rec.channels)
            if col_key == "bit_depth":
                return format_bit_depth(rec.bit_depth)
            if col_key == "ext":
                return rec.ext.upper()
            if col_key == "size_bytes":
                return format_size(rec.size_bytes)
            if col_key == "package":
                return rec.package or "—"
            if col_key == "rel_path":
                return rec.rel_path
            return None

        if role == Qt.ItemDataRole.FontRole and (
            col_key in NUMERIC_COL_KEYS or (col_key == "relevance" and not self._dup_mode)
        ):
            return self._mono

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col_key in (FAV_COL_KEY, MARK_COL_KEY):
                return int(Qt.AlignmentFlag.AlignCenter)
            if col_key in NUMERIC_COL_KEYS:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None

        if role == Qt.ItemDataRole.ForegroundRole:
            if not rec.is_available:
                return theme.OFFLINE
            if col_key == FAV_COL_KEY:
                return theme.PLAY if rec.favorite else theme.TEXT_FAINT
            if col_key == MARK_COL_KEY:
                if self._playing_file_id is not None and rec.id == self._playing_file_id:
                    return theme.PLAY
                return QColor("#D5D0C7")
            if (
                self._playing_file_id is not None
                and rec.id == self._playing_file_id
                and col_key == "filename"
            ):
                return theme.PLAY_SELECTED
            if self._dup_mode:
                dup = self._dup_rows.get(rec.id)
                if dup is not None:
                    if dup.file.is_hidden:
                        return theme.TEXT_FAINT
                    if col_key == "relevance":
                        return theme.PLAY if dup.is_primary else theme.SELECT_MARK
                return None
            if self._ai_mode and col_key == "relevance":
                return theme.SELECT_MARK
            return None

        return None


class FileTableView(QTableView):
    """Virtual audio table with multi-select, ★ toggle, ME audition, waveform seek, spot, and drag."""

    auditionRowRequested = Signal(int)
    waveformSeekRequested = Signal(int, float)
    spotRowRequested = Signal(int)
    spotFilesRequested = Signal(list)  # list[int]
    findSimilarRowRequested = Signal(int)
    columnVisibilityChanged = Signal()
    toggleFavoriteRequested = Signal(int)  # file_id
    batchFavoriteRequested = Signal(list, bool)  # (list[int], favorite)
    setPlayedMarkRequested = Signal(list, bool)  # (list[int], marked)
    clearPlayedMarksRequested = Signal()
    addFilesToCollectionRequested = Signal(list, int)  # (list[int], collection_id)
    removeFilesFromCollectionRequested = Signal(list)  # list[int]
    hideDuplicatesRequested = Signal(list)  # list[int]
    setDuplicatePrimaryRequested = Signal(int, int)  # (group_id, file_id)
    folderActivated = Signal(object)  # FolderRecord
    navigateUpRequested = Signal()
    folderRelocateRequested = Signal(int, int)  # (root_id, folder_id)
    folderAddToCollectionRequested = Signal(int, int, object, str, str)
    navigateToFileFolderRequested = Signal(int)  # file_id
    editMetadataRequested = Signal(list)  # list[int]
    parseUcsFilenameRequested = Signal(list)  # list[int]
    writeBackMetadataRequested = Signal(list)  # list[int]
    aiEnrichMetadataRequested = Signal(list)  # list[int]
    batchRenameRequested = Signal(list)  # list[int]
    csvMetadataRequested = Signal(list)  # list[int]
    statusMessageRequested = Signal(str)

    def __init__(
        self,
        model: FileTableModel,
        thumbnail_manager: ThumbnailManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._file_model = model
        self._thumbnail_manager = thumbnail_manager
        self._press_pos: QPoint | None = None
        self._press_row: int = -1
        self._press_in_waveform = False
        self._press_in_favorite = False
        self._deferred_single_select = False
        self._press_seek_ratio = 0.0
        self._hover_row = -1
        self._hover_ratio: float | None = None
        self._playing_file_id: int | None = None
        self._playing_ratio: float | None = None
        self._spectral_peaks: bool = False
        self.drag_path_resolver: Callable[[FileRecord], Path | None] | None = None
        self.collections_provider: Callable[[], list[tuple[int, str]]] | None = None
        self.active_collection_id_provider: Callable[[], int | None] | None = None

        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(40)
        self._thumb_timer.timeout.connect(self.schedule_visible_thumbnails)

        self._scrollbar_dragging = False
        self._drag_page_timer = QTimer(self)
        self._drag_page_timer.setSingleShot(True)
        self._drag_page_timer.setInterval(45)
        self._drag_page_timer.timeout.connect(self._on_drag_page_timeout)

        self.setObjectName("fileTable")
        self.setModel(model)
        self.setItemDelegate(_FileRowDelegate(self))
        self._waveform_delegate = WaveformItemDelegate(self)
        self.setItemDelegateForColumn(WAVEFORM_COL, self._waveform_delegate)

        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.horizontalScrollBar().setSingleStep(24)
        self.setSortingEnabled(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.viewport().setMouseTracking(True)
        self.verticalHeader().setVisible(False)
        spec = theme.ui_scale_spec()
        self.verticalHeader().setDefaultSectionSize(spec.table_row_h)

        header = self.horizontalHeader()
        header.setMinimumHeight(max(24, spec.table_row_h - 1))
        header.setSectionsMovable(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setStretchLastSection(False)
        header.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        for idx, (_, _, width) in enumerate(COLUMNS):
            self.setColumnWidth(idx, width)
        header.moveSection(header.visualIndex(MARK_COL), 0)
        self._first_visible_visual_col_cached = 0
        self._refresh_first_visible_col()

        header.sectionClicked.connect(self._on_header_section_clicked)
        header.customContextMenuRequested.connect(self._show_header_context_menu)
        header.sectionResized.connect(self._on_section_resized)
        header.sectionMoved.connect(lambda *_args: self._refresh_first_visible_col())
        vbar = self.verticalScrollBar()
        vbar.valueChanged.connect(self._on_vscroll_value_changed)
        vbar.sliderPressed.connect(self._on_vscroll_pressed)
        vbar.sliderMoved.connect(self._on_vscroll_moved)
        vbar.sliderReleased.connect(self._on_vscroll_released)
        self._file_model.modelReset.connect(self._on_model_reset)
        self._file_model.pageReady.connect(self._on_model_page_ready)
        self._file_model.idIndexReady.connect(self._on_id_index_ready)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def apply_ui_scale(self, scale_key: str) -> None:
        spec = theme.ui_scale_spec(scale_key)
        self._file_model.apply_ui_scale(scale_key)
        self.verticalHeader().setDefaultSectionSize(spec.table_row_h)
        self.horizontalHeader().setMinimumHeight(max(24, spec.table_row_h - 1))
        self.viewport().update()

    def _refresh_first_visible_col(self) -> None:
        header = self.horizontalHeader()
        count = header.count()
        first = count
        for logical in range(count):
            if not self.isColumnHidden(logical):
                vis = header.visualIndex(logical)
                if 0 <= vis < first:
                    first = vis
        self._first_visible_visual_col_cached = first

    def _on_vscroll_value_changed(self, _value: int) -> None:
        if not self._scrollbar_dragging:
            self._thumb_timer.start()

    def _on_vscroll_pressed(self) -> None:
        self._scrollbar_dragging = True
        self._thumb_timer.stop()
        self._file_model.set_scroll_throttled(True)

    def _on_vscroll_moved(self, _value: int) -> None:
        if self._scrollbar_dragging:
            interval = 16 if self._file_model.has_ram_index else 120
            if self._drag_page_timer.interval() != interval:
                self._drag_page_timer.setInterval(interval)
            self._drag_page_timer.start()

    def _on_vscroll_released(self) -> None:
        self._scrollbar_dragging = False
        self._drag_page_timer.stop()
        self._file_model.set_scroll_throttled(False)
        self._file_model.clear_pending_pages()
        self._request_visible_pages(prefetch_neighbors=True, force=True)
        self._thumb_timer.start()

    def _on_id_index_ready(self) -> None:
        if not self._scrollbar_dragging:
            self._request_visible_pages(prefetch_neighbors=True, force=True)

    def _on_drag_page_timeout(self) -> None:
        self._request_visible_pages(prefetch_neighbors=False, force=True)

    def _on_model_page_ready(self, _page_idx: int) -> None:
        if not self._scrollbar_dragging:
            self._thumb_timer.start()

    def _request_visible_pages(self, *, prefetch_neighbors: bool, force: bool = False) -> None:
        if self._file_model.is_ai_mode or self._file_model.is_duplicate_mode:
            return
        vis = self._visible_row_range()
        if not vis:
            return
        folder_offset = self._file_model.folder_count
        pages: list[int] = []
        seen: set[int] = set()
        for row in vis:
            file_row = row - folder_offset
            if file_row >= 0:
                p = file_row // PAGE_SIZE
                if p not in seen:
                    seen.add(p)
                    pages.append(p)
        if not pages:
            return
        for p in pages:
            self._file_model.request_page_async(p, force=force)
        if prefetch_neighbors:
            min_p = min(pages)
            max_p = max(pages)
            if min_p > 0:
                self._file_model.request_page_async(min_p - 1, force=force)
            self._file_model.request_page_async(max_p + 1, force=force)

    def _on_model_reset(self) -> None:
        self._drag_page_timer.stop()
        self._scrollbar_dragging = False
        self._file_model.set_scroll_throttled(False)
        self._clear_hover()
        self._thumb_timer.start()

    def _on_section_resized(self, logical_index: int, _old: int, _new: int) -> None:
        if logical_index == WAVEFORM_COL:
            self._thumb_timer.start()

    def _on_header_section_clicked(self, logical_index: int) -> None:
        if (
            not 0 <= logical_index < len(COLUMNS)
            or COLUMNS[logical_index][0] not in NON_SORTABLE_COL_KEYS
        ):
            return
        col_keys = [c[0] for c in COLUMNS]
        sort_key = self._file_model.sort_column
        sort_idx = col_keys.index(sort_key) if sort_key in col_keys else 0
        sort_order = (
            Qt.SortOrder.AscendingOrder if self._file_model.sort_asc else Qt.SortOrder.DescendingOrder
        )
        header = self.horizontalHeader()
        blocked = header.blockSignals(True)
        try:
            header.setSortIndicator(sort_idx, sort_order)
        finally:
            header.blockSignals(blocked)

    def _show_header_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        for idx, (key, title, _) in enumerate(COLUMNS):
            act = QAction(title, menu)
            act.setCheckable(True)
            act.setChecked(not self.isColumnHidden(idx))
            if key == "filename":
                act.setEnabled(False)
            act.toggled.connect(
                lambda checked, col_idx=idx, col_key=key: self._toggle_column_visible(
                    col_idx, col_key, checked
                )
            )
            menu.addAction(act)
        mark_count = self._file_model.played_mark_count()
        if mark_count > 0:
            menu.addSeparator()
            clear_mark_act = QAction(f"清空全部已听标记 ({mark_count:,} 条)", menu)
            clear_mark_act.triggered.connect(
                lambda _checked=False: self.clearPlayedMarksRequested.emit()
            )
            menu.addAction(clear_mark_act)
        menu.exec(self.horizontalHeader().mapToGlobal(pos))

    def _toggle_column_visible(self, col_idx: int, col_key: str, visible: bool) -> None:
        if col_key == "filename":
            return
        self.setColumnHidden(col_idx, not visible)
        self._refresh_first_visible_col()
        self.columnVisibilityChanged.emit()
        if col_key == WAVEFORM_COL_KEY and visible:
            self.schedule_visible_thumbnails()

    def set_ai_columns_visible(self, visible: bool) -> None:
        for idx, (key, _, default_width) in enumerate(COLUMNS):
            if key in AI_COL_KEYS:
                self.setColumnHidden(idx, not visible)
                if visible and self.columnWidth(idx) < 24:
                    self.setColumnWidth(idx, default_width)
        self._refresh_first_visible_col()

    def _visible_row_range(self) -> range:
        total = self._file_model.rowCount()
        if total <= 0:
            return range(0)
        vp = self.viewport().rect()
        first = self.rowAt(vp.top())
        if first < 0:
            first = 0
        last = self.rowAt(vp.bottom())
        if last < 0:
            last = min(total - 1, first + 40)
        return range(first, min(total, last + 1))

    def _update_waveform_cell(self, row: int) -> None:
        if row < 0 or self.isColumnHidden(WAVEFORM_COL):
            return
        rect = self.visualRect(self._file_model.index(row, WAVEFORM_COL))
        if rect.isValid():
            self.viewport().update(rect)

    def set_playing_progress(self, file_id: int | None, ratio: float | None) -> None:
        old_id = self._playing_file_id
        self._playing_file_id = file_id
        self._playing_ratio = ratio
        self._waveform_delegate.set_playing_state(file_id, ratio)
        if self.isColumnHidden(WAVEFORM_COL):
            return
        for row in self._visible_row_range():
            rec = self._file_model.peek_record_at(row)
            if rec is not None and (rec.id == file_id or (old_id is not None and rec.id == old_id)):
                self._update_waveform_cell(row)

    def repaint_file_waveforms(self, file_ids: set[int]) -> None:
        if not file_ids:
            return
        for fid in file_ids:
            self._waveform_delegate.invalidate_file(fid)
        if self.isColumnHidden(WAVEFORM_COL):
            return
        for row in self._visible_row_range():
            rec = self._file_model.peek_record_at(row)
            if rec is not None and rec.id in file_ids:
                self._update_waveform_cell(row)

    def repaint_file_waveform(self, file_id: int) -> None:
        self.repaint_file_waveforms({file_id})

    def set_spectral_peaks(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._spectral_peaks == enabled:
            return
        self._spectral_peaks = enabled
        self._waveform_delegate.set_spectral_peaks(enabled)
        self.viewport().update()
        if enabled:
            self.schedule_visible_thumbnails()

    def schedule_visible_thumbnails(self) -> None:
        if (
            self._scrollbar_dragging
            or self._thumbnail_manager is None
            or self.isColumnHidden(WAVEFORM_COL)
        ):
            return
        total = self._file_model.rowCount()
        if total <= 0:
            return
        vis = self._visible_row_range()
        first = vis.start
        last = vis.stop - 1
        row_order: list[int] = list(vis)
        row_order.extend(range(last + 1, min(total, last + 49)))
        row_order.extend(range(max(0, first - 24), first))

        items: list[tuple[int, int, Path]] = []
        for row in row_order:
            rec = self._file_model.peek_record_at(row)
            if rec is None or not rec.is_available:
                continue
            has_thumb = self._file_model.thumbnail_for(rec) is not None
            has_spec = self._file_model.spectral_thumbnail_for(rec) is not None
            if has_thumb and (not self._spectral_peaks or has_spec):
                continue
            items.append((rec.id, rec.mtime_ns, rec.full_path))
        if items:
            self._thumbnail_manager.request_visible(items, require_spectral=self._spectral_peaks)

    def _clear_hover(self) -> None:
        old_row = self._hover_row
        if old_row < 0 and self._hover_ratio is None:
            return
        self._hover_row = -1
        self._hover_ratio = None
        self._waveform_delegate.set_hover_state(-1, None)
        self.viewport().unsetCursor()
        if old_row >= 0:
            self._update_waveform_cell(old_row)

    def _update_hover(self, pos_f) -> None:
        if self.isColumnHidden(WAVEFORM_COL):
            self._clear_hover()
            return
        idx = self.indexAt(pos_f.toPoint())
        new_row = -1
        new_ratio: float | None = None
        if idx.isValid() and idx.column() == WAVEFORM_COL:
            rec = self._file_model.peek_record_at(idx.row())
            dur = rec.duration_sec if rec is not None else 0.0
            ratio = WaveformItemDelegate.seek_ratio_for_x(self.visualRect(idx), pos_f.x(), dur)
            if ratio is not None:
                new_row = idx.row()
                new_ratio = ratio
                self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.viewport().unsetCursor()
        else:
            self.viewport().unsetCursor()

        if new_row != self._hover_row or new_ratio != self._hover_ratio:
            old_row = self._hover_row
            self._hover_row = new_row
            self._hover_ratio = new_ratio
            self._waveform_delegate.set_hover_state(new_row, new_ratio)
            if old_row >= 0 and old_row != new_row:
                self._update_waveform_cell(old_row)
            if new_row >= 0:
                self._update_waveform_cell(new_row)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_thumb_timer"):
            self._thumb_timer.start()

    def viewportEvent(self, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Leave and hasattr(self, "_hover_row"):
            self._clear_hover()
        return super().viewportEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
        if hasattr(self, "_hover_row"):
            self._clear_hover()
        super().leaveEvent(event)

    def selected_rows(self) -> list[int]:
        sel_model = self.selectionModel()
        if sel_model is None:
            return []
        return sorted({idx.row() for idx in sel_model.selectedRows()})

    def selected_records(self) -> list[FileRecord]:
        out: list[FileRecord] = []
        for r in self.selected_rows():
            rec = self._file_model.record_at(r)
            if rec is not None:
                out.append(rec)
        return out

    def selected_file_ids(self) -> list[int]:
        return [rec.id for rec in self.selected_records()]

    def _start_drag_for_row(self, row: int) -> None:
        sel_rows = self.selected_rows()
        if len(sel_rows) > 1 and row in sel_rows:
            targets: list[Path] = []
            file_ids: list[int] = []
            for r in sel_rows:
                rec = self._file_model.record_at(r)
                if rec is not None and rec.is_available:
                    target = (
                        self.drag_path_resolver(rec)
                        if self.drag_path_resolver is not None
                        else rec.full_path
                    )
                    if target is not None:
                        targets.append(target)
                        file_ids.append(rec.id)
            if len(targets) > 1:
                start_files_drag(self, targets, file_ids=file_ids)
                return
            if len(targets) == 1:
                start_file_drag(self, targets[0])
                return
            return

        self.selectRow(row)
        rec = self._file_model.record_at(row)
        if rec is not None and rec.is_available:
            target = (
                self.drag_path_resolver(rec)
                if self.drag_path_resolver is not None
                else rec.full_path
            )
            if target is not None:
                start_file_drag(self, target)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self.indexAt(event.position().toPoint())
            if idx.isValid():
                sel_model = self.selectionModel()
                was_selected = (
                    sel_model.isRowSelected(idx.row(), QModelIndex())
                    if sel_model is not None
                    else False
                )
                sel_count = len(self.selected_rows())
                self._press_pos = event.position().toPoint()
                self._press_row = idx.row()
                self._press_in_favorite = False
                self._deferred_single_select = False
                is_folder = self._file_model.is_folder_row(idx.row())

                if (
                    not is_folder
                    and idx.column() == FAV_COL
                    and event.modifiers() == Qt.KeyboardModifier.NoModifier
                ):
                    self._press_in_waveform = False
                    self._press_in_favorite = True
                    super().mousePressEvent(event)
                    return

                if not is_folder and idx.column() == WAVEFORM_COL:
                    rec = self._file_model.record_at(idx.row())
                    dur = rec.duration_sec if rec is not None else 0.0
                    seek_ratio = WaveformItemDelegate.seek_ratio_for_x(
                        self.visualRect(idx), event.position().x(), dur
                    )
                    if seek_ratio is not None:
                        self._press_in_waveform = True
                        self._press_seek_ratio = seek_ratio
                        super().mousePressEvent(event)
                        return

                self._press_in_waveform = False
                if (
                    was_selected
                    and sel_count > 1
                    and event.modifiers() == Qt.KeyboardModifier.NoModifier
                ):
                    self._deferred_single_select = True
                    event.accept()
                    return
                super().mousePressEvent(event)
                if not was_selected and not is_folder:
                    self.auditionRowRequested.emit(idx.row())
                return
        self._press_pos = None
        self._press_row = -1
        self._press_in_waveform = False
        self._press_in_favorite = False
        self._deferred_single_select = False
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self.indexAt(event.position().toPoint())
            if idx.isValid():
                folder_entry = self._file_model.folder_at(idx.row())
                if folder_entry is not None:
                    self._press_pos = None
                    self._press_row = -1
                    self._press_in_waveform = False
                    self._press_in_favorite = False
                    self._deferred_single_select = False
                    self.folderActivated.emit(folder_entry[0])
                    event.accept()
                    return
                if idx.column() == FAV_COL:
                    event.accept()
                    return
                if idx.column() == WAVEFORM_COL:
                    rec = self._file_model.record_at(idx.row())
                    dur = rec.duration_sec if rec is not None else 0.0
                    seek_ratio = WaveformItemDelegate.seek_ratio_for_x(
                        self.visualRect(idx), event.position().x(), dur
                    )
                    if seek_ratio is not None:
                        event.accept()
                        return
                self._press_pos = None
                self._press_row = -1
                self._press_in_waveform = False
                self._press_in_favorite = False
                self._deferred_single_select = False
                self.spotRowRequested.emit(idx.row())
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._press_pos is not None
            and self._press_row >= 0
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            dist = (event.position().toPoint() - self._press_pos).manhattanLength()
            if dist >= QApplication.startDragDistance() and not self._press_in_favorite:
                row = self._press_row
                self._press_pos = None
                self._press_row = -1
                self._press_in_waveform = False
                self._deferred_single_select = False
                self._start_drag_for_row(row)
                return
            if self._press_in_waveform:
                event.accept()
                return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_hover(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        was_wf_press = self._press_in_waveform
        was_fav_press = self._press_in_favorite
        was_deferred_single = self._deferred_single_select
        press_row = self._press_row
        seek_ratio = self._press_seek_ratio
        self._press_pos = None
        self._press_row = -1
        self._press_in_waveform = False
        self._press_in_favorite = False
        self._deferred_single_select = False

        if was_deferred_single and press_row >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self.selectRow(press_row)
            self.auditionRowRequested.emit(press_row)
            event.accept()
            return

        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and press_row >= 0:
            if was_wf_press:
                self.waveformSeekRequested.emit(press_row, seek_ratio)
            elif was_fav_press:
                rel_idx = self.indexAt(event.position().toPoint())
                if rel_idx.isValid() and rel_idx.row() == press_row and rel_idx.column() == FAV_COL:
                    rec = self._file_model.record_at(press_row)
                    if rec is not None:
                        self.toggleFavoriteRequested.emit(rec.id)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            before = self.currentIndex().row()
            super().keyPressEvent(event)
            after = self.currentIndex().row()
            if after >= 0 and after != before:
                self.auditionRowRequested.emit(after)
            event.accept()
            return
        if key == Qt.Key.Key_Backspace:
            self.navigateUpRequested.emit()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            idx = self.currentIndex()
            if idx.isValid():
                folder_entry = self._file_model.folder_at(idx.row())
                if folder_entry is not None:
                    self.folderActivated.emit(folder_entry[0])
                else:
                    self.spotRowRequested.emit(idx.row())
                event.accept()
                return
        super().keyPressEvent(event)

    def startDrag(self, supported_actions) -> None:  # noqa: N802
        row = self._press_row if self._press_row >= 0 else self.currentIndex().row()
        if row >= 0 and not self._file_model.is_folder_row(row):
            self._start_drag_for_row(row)

    def _show_context_menu(self, pos) -> None:
        idx = self.indexAt(pos)
        if not idx.isValid():
            return
        folder_entry = self._file_model.folder_at(idx.row())
        if folder_entry is not None:
            folder, full_path, _is_online = folder_entry
            menu = QMenu(self)
            open_act = QAction("打开此目录 (Enter)", menu)
            open_act.triggered.connect(
                lambda _checked=False, f=folder: self.folderActivated.emit(f)
            )
            menu.addAction(open_act)
            if full_path and Path(full_path).exists():
                reveal_dir = QAction("在资源管理器中打开", menu)
                reveal_dir.triggered.connect(
                    lambda _checked=False, p=Path(full_path): self._handle_reveal_path(p)
                )
                menu.addAction(reveal_dir)
            collections = self.collections_provider() if self.collections_provider is not None else []
            coll_menu = menu.addMenu("加入合集分组")
            for cid, cname in collections:
                c_act = QAction(cname, coll_menu)
                c_act.triggered.connect(
                    lambda _checked=False, x=cid, f=folder: self.folderAddToCollectionRequested.emit(
                        x, f.root_id, f.id, f.rel_path, f.name
                    )
                )
                coll_menu.addAction(c_act)
            if collections:
                coll_menu.addSeparator()
            new_c = QAction("+ 新建合集并加入…", coll_menu)
            new_c.triggered.connect(
                lambda _checked=False, f=folder: self.folderAddToCollectionRequested.emit(
                    0, f.root_id, f.id, f.rel_path, f.name
                )
            )
            coll_menu.addAction(new_c)
            menu.addSeparator()
            relocate_act = QAction("重定位此目录路径…", menu)
            relocate_act.triggered.connect(
                lambda _checked=False, rid=folder.root_id, fid=folder.id: self.folderRelocateRequested.emit(
                    rid, fid
                )
            )
            menu.addAction(relocate_act)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        rec = self._file_model.record_at(idx.row())
        if rec is None:
            return
        selected_recs = self.selected_records()
        if all(r.id != rec.id for r in selected_recs):
            self.selectRow(idx.row())
            selected_recs = [rec]
        selected_ids = [r.id for r in selected_recs]

        menu = QMenu(self)
        full_path = rec.full_path

        if len(selected_ids) > 1:
            spot_act = QAction(f"连续插入 REAPER 光标处 ({len(selected_ids)} 条 · Enter)", menu)
            spot_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.spotFilesRequested.emit(fids)
            )
        else:
            spot_act = QAction("插入到 REAPER 光标处 (Enter)", menu)
            spot_act.triggered.connect(
                lambda _checked=False, r=idx.row(): self.spotRowRequested.emit(r)
            )
        menu.addAction(spot_act)

        similar_act = QAction("以该音频找同类 (AI)", menu)
        similar_act.triggered.connect(
            lambda _checked=False, r=idx.row(): self.findSimilarRowRequested.emit(r)
        )
        menu.addAction(similar_act)
        menu.addSeparator()

        if len(selected_ids) == 1:
            fav_label = "取消收藏 (★)" if rec.favorite else "加入收藏 (★)"
            fav_act = QAction(fav_label, menu)
            fav_act.triggered.connect(
                lambda _checked=False, fid=rec.id: self.toggleFavoriteRequested.emit(fid)
            )
            menu.addAction(fav_act)
        else:
            fav_on_act = QAction(f"★ 批量设为收藏 ({len(selected_ids)} 条)", menu)
            fav_on_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.batchFavoriteRequested.emit(
                    fids, True
                )
            )
            menu.addAction(fav_on_act)

            fav_off_act = QAction(f"取消收藏 ({len(selected_ids)} 条)", menu)
            fav_off_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.batchFavoriteRequested.emit(
                    fids, False
                )
            )
            menu.addAction(fav_off_act)

        collections = self.collections_provider() if self.collections_provider is not None else []
        if collections:
            coll_menu = menu.addMenu(
                f"加入我的合集 ({len(selected_ids)} 条)" if len(selected_ids) > 1 else "加入我的合集"
            )
            for cid, cname in collections:
                c_act = QAction(cname, coll_menu)
                c_act.triggered.connect(
                    lambda _checked=False, x=cid, fids=list(selected_ids): self.addFilesToCollectionRequested.emit(
                        fids, x
                    )
                )
                coll_menu.addAction(c_act)

        active_cid = (
            self.active_collection_id_provider()
            if self.active_collection_id_provider is not None
            else None
        )
        if active_cid is not None:
            rm_coll_act = QAction(
                f"从当前合集移出 ({len(selected_ids)} 条)"
                if len(selected_ids) > 1
                else "从当前合集移出",
                menu,
            )
            rm_coll_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.removeFilesFromCollectionRequested.emit(
                    fids
                )
            )
            menu.addAction(rm_coll_act)

        if self._file_model.is_duplicate_mode:
            menu.addSeparator()
            dup = self._file_model.duplicate_row_for(rec.id)
            if dup is not None and not dup.is_primary and len(selected_ids) == 1:
                primary_act = QAction("设为保留主版本", menu)
                primary_act.triggered.connect(
                    lambda _checked=False, gid=dup.group_id, fid=rec.id: self.setDuplicatePrimaryRequested.emit(
                        gid, fid
                    )
                )
                menu.addAction(primary_act)
            hide_label = (
                f"在库中隐藏所选副本 ({len(selected_ids)} 条 · 不删文件)"
                if len(selected_ids) > 1
                else "在库中隐藏此副本（不删文件）"
            )
            if dup is None or not dup.file.is_hidden or len(selected_ids) > 1:
                hide_act = QAction(hide_label, menu)
                hide_act.triggered.connect(
                    lambda _checked=False, fids=list(selected_ids): self.hideDuplicatesRequested.emit(
                        fids
                    )
                )
                menu.addAction(hide_act)

        menu.addSeparator()
        if len(selected_ids) == 1:
            is_marked = self._file_model.is_played_marked(rec.id)
            mark_label = "取消已听标记 (•)" if is_marked else "标记为已听 (•)"
            mark_act = QAction(mark_label, menu)
            mark_act.triggered.connect(
                lambda _checked=False, fid=rec.id, nxt=not is_marked: self.setPlayedMarkRequested.emit(
                    [fid], nxt
                )
            )
            menu.addAction(mark_act)
        else:
            mark_on_act = QAction(f"标记为已听 ({len(selected_ids)} 条)", menu)
            mark_on_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.setPlayedMarkRequested.emit(
                    fids, True
                )
            )
            menu.addAction(mark_on_act)

            mark_off_act = QAction(f"取消已听标记 ({len(selected_ids)} 条)", menu)
            mark_off_act.triggered.connect(
                lambda _checked=False, fids=list(selected_ids): self.setPlayedMarkRequested.emit(
                    fids, False
                )
            )
            menu.addAction(mark_off_act)

        mark_count = self._file_model.played_mark_count()
        if mark_count > 0:
            clear_mark_act = QAction(f"清空全部已听标记 ({mark_count:,} 条)", menu)
            clear_mark_act.triggered.connect(
                lambda _checked=False: self.clearPlayedMarksRequested.emit()
            )
            menu.addAction(clear_mark_act)

        meta_menu = menu.addMenu(
            f"元数据 ({len(selected_ids)} 条)" if len(selected_ids) > 1 else "元数据"
        )
        edit_meta_act = QAction(
            f"批量编辑元数据 / UCS ({len(selected_ids)} 条)…"
            if len(selected_ids) > 1
            else "编辑元数据 / UCS 分类…",
            meta_menu,
        )
        edit_meta_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.editMetadataRequested.emit(fids)
        )
        meta_menu.addAction(edit_meta_act)

        parse_ucs_act = QAction("从文件名解析 UCS 字段（仅存库）", meta_menu)
        parse_ucs_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.parseUcsFilenameRequested.emit(fids)
        )
        meta_menu.addAction(parse_ucs_act)

        ai_enrich_act = QAction("AI 自动补全 UCS / 描述 / 翻译…", meta_menu)
        ai_enrich_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.aiEnrichMetadataRequested.emit(fids)
        )
        meta_menu.addAction(ai_enrich_act)

        meta_menu.addSeparator()

        rename_act = QAction(
            f"模板批量改名 ({len(selected_ids)} 条)…"
            if len(selected_ids) > 1
            else "模板改名…",
            meta_menu,
        )
        rename_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.batchRenameRequested.emit(fids)
        )
        meta_menu.addAction(rename_act)

        csv_act = QAction("CSV 导入 / 导出元数据…", meta_menu)
        csv_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.csvMetadataRequested.emit(fids)
        )
        meta_menu.addAction(csv_act)

        meta_menu.addSeparator()

        write_meta_act = QAction("显式写回源文件（预览与备份）…", meta_menu)
        write_meta_act.triggered.connect(
            lambda _checked=False, fids=list(selected_ids): self.writeBackMetadataRequested.emit(
                fids
            )
        )
        meta_menu.addAction(write_meta_act)

        menu.addSeparator()

        jump_dir_act = QAction("跳转到所在目录", menu)
        jump_dir_act.triggered.connect(
            lambda _checked=False, fid=rec.id: self.navigateToFileFolderRequested.emit(fid)
        )
        menu.addAction(jump_dir_act)

        reveal_act = QAction("在资源管理器中打开", menu)
        reveal_act.triggered.connect(
            lambda _checked=False, p=full_path: self._handle_reveal_path(p)
        )
        menu.addAction(reveal_act)

        copy_name = QAction("复制文件名", menu)
        copy_name.triggered.connect(
            lambda _checked=False, name=rec.filename: QGuiApplication.clipboard().setText(name)
        )
        menu.addAction(copy_name)

        copy_path = QAction("复制完整路径", menu)
        copy_path.triggered.connect(
            lambda _checked=False, p=str(full_path): QGuiApplication.clipboard().setText(p)
        )
        menu.addAction(copy_path)

        menu.exec(self.viewport().mapToGlobal(pos))

    def _handle_reveal_path(self, path: Path) -> None:
        ok, error_msg = reveal_in_explorer(path)
        if not ok and error_msg:
            self.statusMessageRequested.emit(error_msg)


def _is_win32_low_integrity() -> bool:
    """Return True when the current Windows process token runs below Medium integrity."""
    if os.name != "nt":
        return False
    h_token = ctypes.c_void_p()
    kernel32 = None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        advapi32.OpenProcessToken.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        advapi32.OpenProcessToken.restype = ctypes.c_int
        advapi32.GetTokenInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        advapi32.GetTokenInformation.restype = ctypes.c_int
        advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(ctypes.c_uint32)

        token_query = 0x0008
        token_integrity_level = 25
        medium_rid = 0x2000

        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), token_query, ctypes.byref(h_token)
        ):
            return False

        needed = ctypes.c_uint32(0)
        advapi32.GetTokenInformation(
            h_token, token_integrity_level, None, 0, ctypes.byref(needed)
        )
        if needed.value == 0:
            return False

        buf = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            h_token, token_integrity_level, buf, needed.value, ctypes.byref(needed)
        ):
            return False

        psid = ctypes.c_void_p.from_buffer(buf).value
        if not psid:
            return False
        count_ptr = advapi32.GetSidSubAuthorityCount(psid)
        if not count_ptr or count_ptr[0] == 0:
            return False
        rid_ptr = advapi32.GetSidSubAuthority(psid, int(count_ptr[0]) - 1)
        if not rid_ptr:
            return False
        return int(rid_ptr[0]) < medium_rid
    except Exception:
        return False
    finally:
        if kernel32 is not None and h_token.value:
            try:
                kernel32.CloseHandle(h_token)
            except Exception:
                pass


def _win_path_exists(path: Path) -> bool:
    try:
        if path.exists():
            return True
    except OSError:
        pass
    if os.name == "nt":
        raw = str(path)
        if len(raw) >= 248 and not raw.startswith("\\\\?\\"):
            try:
                return os.path.exists(f"\\\\?\\{raw}")
            except OSError:
                return False
    return False


def _to_win32_short_path(path_str: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        kernel32.GetShortPathNameW.restype = ctypes.c_uint
        src = f"\\\\?\\{path_str}" if not path_str.startswith("\\\\?\\") else path_str
        need = int(kernel32.GetShortPathNameW(src, None, 0))
        if need <= 0:
            return None
        buf = ctypes.create_unicode_buffer(need + 1)
        written = int(kernel32.GetShortPathNameW(src, buf, len(buf)))
        if written <= 0:
            return None
        short = buf.value
        if short.startswith("\\\\?\\"):
            short = short[4:]
        return short or None
    except Exception:
        return None


def _win32_shell_execute_explorer(target: str, *, select: bool = True) -> bool:
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.ShellExecuteW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        shell32.ShellExecuteW.restype = ctypes.c_void_p
        if select:
            res = shell32.ShellExecuteW(None, "open", "explorer.exe", f'/select,"{target}"', None, 1)
        else:
            res = shell32.ShellExecuteW(None, "open", target, None, None, 1)
        return int(res or 0) > 32
    except Exception:
        return False


def _open_nearest_existing_folder(folder: Path) -> bool:
    cand = folder
    for _ in range(16):
        if _win_path_exists(cand):
            target = str(cand)
            if os.name == "nt" and len(target) >= 255:
                short = _to_win32_short_path(target)
                if short:
                    target = short
            if _win32_shell_execute_explorer(target, select=False):
                return True
            try:
                os.startfile(target)  # noqa: S606
                return True
            except OSError:
                return False
        parent = cand.parent
        if parent == cand:
            break
        cand = parent
    return False


def reveal_in_explorer(path: Path) -> tuple[bool, str]:
    """Open Windows Explorer and select ``path`` (or nearest existing folder)."""
    if os.name != "nt":
        return False, "当前系统不支持定位到资源管理器"
    if _is_win32_low_integrity():
        return (
            False,
            "当前进程处于低完整性权限，Windows 阻止启动资源管理器；请在 PowerShell 执行 "
            'icacls ".\\.venv\\Scripts\\*.exe" /setintegritylevel Medium 后重启',
        )
    if _reveal_in_explorer_impl(path):
        return True, ""
    return False, f"无法打开目录：{path}"


def _reveal_in_explorer(path: Path) -> bool:
    return reveal_in_explorer(path)[0]


def _reveal_in_explorer_impl(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = Path(os.path.abspath(str(path)))

    if not _win_path_exists(resolved):
        return _open_nearest_existing_folder(resolved.parent)

    target = str(resolved)
    if len(target) >= 255:
        short = _to_win32_short_path(target)
        if short and len(short) < 255:
            target = short
        else:
            return _open_nearest_existing_folder(resolved.parent)

    if "," not in target:
        if _win32_shell_execute_explorer(target, select=True):
            return True
        try:
            subprocess.Popen(f'explorer.exe /select,"{target}"')  # noqa: S602
            return True
        except OSError:
            pass
        if _win32_select_in_explorer(Path(target)):
            return True
        return _open_nearest_existing_folder(resolved.parent)

    if _win32_select_in_explorer(Path(target)):
        return True
    if _win32_shell_execute_explorer(target, select=True):
        return True
    return _open_nearest_existing_folder(resolved.parent)


def _win32_select_in_explorer(path: Path) -> bool:
    ole32 = None
    did_init = False
    try:
        ole32 = ctypes.WinDLL("ole32")
        shell32 = ctypes.WinDLL("shell32")
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoUninitialize.argtypes = []
        ole32.CoUninitialize.restype = None

        hr_co = int(ole32.CoInitializeEx(None, 2))  # COINIT_APARTMENTTHREADED
        did_init = hr_co in (0, 1)

        shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
        shell32.ILCreateFromPathW.restype = ctypes.c_void_p
        shell32.SHOpenFolderAndSelectItems.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        shell32.SHOpenFolderAndSelectItems.restype = ctypes.c_long
        shell32.ILFree.argtypes = [ctypes.c_void_p]
        shell32.ILFree.restype = None

        pidl = shell32.ILCreateFromPathW(str(path))
        if not pidl:
            return False
        try:
            hr = int(shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0))
            return hr == 0
        finally:
            shell32.ILFree(pidl)
    except Exception:
        return False
    finally:
        if did_init and ole32 is not None:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass
