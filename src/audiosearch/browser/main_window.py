"""Standalone desktop audio browser window."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QIcon,
    QKeyEvent,
    QPainter,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from audiosearch.audioio.video import is_video_path
from audiosearch.audition.render import RENDER_DIR_NAME, prune_render_dir, render_range
from audiosearch.browser.audition_controller import AuditionController
from audiosearch.browser.collections_tree import (
    KIND_COLLECTION,
    KIND_COLLECTION_FOLDER,
    KIND_DUPLICATES,
    KIND_FAVORITES,
    KIND_RECENT,
    SidebarViewTarget,
)
from audiosearch.browser.drag_knob import DragValueControl
from audiosearch.browser.file_drag import start_file_drag
from audiosearch.browser.file_table import (
    AI_COL_KEYS,
    COLUMNS,
    DEFAULT_COLUMN_ORDER,
    MARK_COL_KEY,
    NON_SORTABLE_COL_KEYS,
    PAGE_SIZE,
    FileTableModel,
    FileTableView,
)
from audiosearch.browser.folder_tree import FolderTree, TreeLocation
from audiosearch.browser.inspector import InspectorPanel
from audiosearch.browser.metadata_dialogs import MetadataEditorDialog, WriteBackDialog
from audiosearch.browser.metadata_tools_dialogs import (
    AiEnrichDialog,
    BatchRenameDialog,
    CsvMetadataDialog,
)
from audiosearch.browser.reaper_dialog import ReaperSetupDialog
from audiosearch.browser.reaper_shortcuts import default_reaper_ini_path
from audiosearch.browser.reaper_shortcuts_dialog import ReaperShortcutsImportDialog
from audiosearch.browser.reaper_spot import ReaperSpotClient
from audiosearch.browser.runtime_paths import is_frozen
from audiosearch.browser.update_dialog import UpdateDialog
from audiosearch.browser.updater import (
    UpdateCheckResult,
    check_for_updates,
    effective_version,
    mark_patch_boot_healthy,
)
from audiosearch.browser.settings import (
    BrowserSettings,
    load_settings,
    save_settings,
    settings_path_for_db,
)
from audiosearch.browser.settings_dialog import SettingsDialog
from audiosearch.browser.theme import (
    apply_theme,
    mono_font,
    normalize_ui_scale,
    ui_scale_spec,
)
from audiosearch.browser.title_bar import (
    CustomTitleBar,
    apply_win32_frameless_style,
    handle_win32_native_event,
)
from audiosearch.browser.waveform_delegate import ThumbnailManager
from audiosearch.browser.waveform_widget import WaveformWidget
from audiosearch.library.duplicates import DuplicateProgress, DuplicateScanReport, DuplicateScanner
from audiosearch.library.peaks import (
    PeakData,
    compute_file_peaks,
    peaks_to_thumbnail,
    peaks_to_thumbnail_spectral,
)
from audiosearch.library.scanner import LibraryScanner, ScanProgress, ScanReport
from audiosearch.library.search_syntax import extract_absolute_path_query
from audiosearch.library.store import FileRecord, LibraryStore, QueryFilter
from audiosearch.library.ucs import parse_ucs_filename

OVERVIEW_POINTS = 2048


def _make_nav_icon(direction: str) -> QIcon:
    """Build crisp HiDPI vector triangle icon so back/forward/up match size identically."""
    logical = 20
    scale = 2
    px = logical * scale
    if direction == "left":
        pts = [QPointF(4.5, 10.0), QPointF(14.5, 4.0), QPointF(14.5, 16.0)]
    elif direction == "right":
        pts = [QPointF(15.5, 10.0), QPointF(5.5, 4.0), QPointF(5.5, 16.0)]
    else:
        pts = [QPointF(10.0, 4.5), QPointF(4.0, 14.5), QPointF(16.0, 14.5)]
    poly = QPolygonF(pts)

    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, QColor("#C8C3B9")),
        (QIcon.Mode.Active, QColor("#ECE9E2")),
        (QIcon.Mode.Disabled, QColor("#5C5851")),
    ):
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(float(scale))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(poly)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


def _make_transport_icon(kind: str) -> QIcon:
    """Build crisp HiDPI vector transport icons (no Unicode Emoji fallback)."""
    logical = 14
    scale = 2
    px = logical * scale
    if kind == "play":
        modes = (
            (QIcon.Mode.Normal, QColor("#ECE9E2")),
            (QIcon.Mode.Active, QColor("#FFFFFF")),
            (QIcon.Mode.Disabled, QColor("#5C5851")),
        )
    elif kind == "pause":
        modes = (
            (QIcon.Mode.Normal, QColor("#F0A387")),
            (QIcon.Mode.Active, QColor("#FFB89E")),
            (QIcon.Mode.Disabled, QColor("#5C5851")),
        )
    else:
        modes = (
            (QIcon.Mode.Normal, QColor("#C8C3B9")),
            (QIcon.Mode.Active, QColor("#ECE9E2")),
            (QIcon.Mode.Disabled, QColor("#5C5851")),
        )

    icon = QIcon()
    for mode, color in modes:
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(float(scale))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        if kind == "play":
            poly = QPolygonF([QPointF(3.5, 2.0), QPointF(12.0, 7.0), QPointF(3.5, 12.0)])
            p.drawPolygon(poly)
        elif kind == "pause":
            p.drawRoundedRect(QRectF(3.0, 2.2, 2.6, 9.6), 0.9, 0.9)
            p.drawRoundedRect(QRectF(8.4, 2.2, 2.6, 9.6), 0.9, 0.9)
        else:
            p.drawRoundedRect(QRectF(2.5, 2.5, 9.0, 9.0), 1.4, 1.4)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


class _QuerySignals(QObject):
    finished = Signal(
        int, int, list, list, tuple, object
    )  # (gen, total, first_page, folders, params, reveal_payload)
    failed = Signal(int, str)


class _QueryTask(QRunnable):
    def __init__(
        self,
        store: LibraryStore,
        generation: int,
        params: tuple[str, int | None, int | None, str, bool, str, bool, QueryFilter],
        signals: _QuerySignals,
        *,
        reveal_file_id: int | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._generation = generation
        self._params = params
        self._signals = signals
        self._reveal_file_id = reveal_file_id

    @Slot()
    def run(self) -> None:
        query, root_id, folder_id, rel_path, include_sub, sort_col, sort_asc, filters = self._params
        try:
            folders: list[tuple[Any, str, bool]] = []
            if not include_sub and root_id is not None:
                root_obj = self._store.get_root(root_id)
                root_path = root_obj.path if root_obj is not None else ""
                root_online = root_obj.is_online if root_obj is not None else False
                for f in self._store.list_folders(root_id, parent_id=folder_id):
                    full = str(Path(root_path) / Path(f.rel_path)) if root_path else f.rel_path
                    is_on = root_online and Path(full).exists()
                    if filters is not None and filters.online_only and not is_on:
                        continue
                    folders.append((f, full, is_on))
                if sort_col == "filename" and not sort_asc:
                    folders.reverse()
            total = self._store.query_count(
                query=query,
                root_id=root_id,
                folder_id=folder_id,
                folder_rel_path=rel_path,
                include_subfolders=include_sub,
                filters=filters,
            )
            first_page = self._store.query_page(
                offset=0,
                limit=PAGE_SIZE,
                query=query,
                root_id=root_id,
                folder_id=folder_id,
                folder_rel_path=rel_path,
                include_subfolders=include_sub,
                sort_column=sort_col,
                sort_asc=sort_asc,
                filters=filters,
                total_count=total,
            )
            reveal_payload: tuple[int, int, dict[int, list[FileRecord]]] | None = None
            if self._reveal_file_id is not None:
                target_id = int(self._reveal_file_id)
                file_pos: int | None = None
                for idx, rec in enumerate(first_page):
                    if rec.id == target_id:
                        file_pos = idx
                        break
                extra_pages: dict[int, list[FileRecord]] = {}
                if file_pos is None:
                    file_pos = self._store.query_file_position(
                        target_id,
                        query=query,
                        root_id=root_id,
                        folder_id=folder_id,
                        folder_rel_path=rel_path,
                        include_subfolders=include_sub,
                        sort_column=sort_col,
                        sort_asc=sort_asc,
                        filters=filters,
                    )
                    if file_pos is not None:
                        page_idx = file_pos // PAGE_SIZE
                        if page_idx > 0:
                            target_page = self._store.query_page(
                                offset=page_idx * PAGE_SIZE,
                                limit=PAGE_SIZE,
                                query=query,
                                root_id=root_id,
                                folder_id=folder_id,
                                folder_rel_path=rel_path,
                                include_subfolders=include_sub,
                                sort_column=sort_col,
                                sort_asc=sort_asc,
                                filters=filters,
                                total_count=total,
                            )
                            extra_pages[page_idx] = target_page
                if file_pos is not None:
                    reveal_payload = (target_id, file_pos, extra_pages)
        except Exception as exc:
            try:
                self._signals.failed.emit(self._generation, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.finished.emit(
                self._generation, total, first_page, folders, self._params, reveal_payload
            )
        except RuntimeError:
            pass


class _DupSignals(QObject):
    progress = Signal(object)  # DuplicateProgress
    done = Signal(object)  # DuplicateReport
    failed = Signal(str)


class _OnlineSignals(QObject):
    finished = Signal(bool)  # changed


class _OnlineTask(QRunnable):
    def __init__(
        self,
        store: LibraryStore,
        signals: _OnlineSignals,
        *,
        active_folder: tuple[int, str] | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._signals = signals
        self._active_folder = active_folder

    @Slot()
    def run(self) -> None:
        try:
            before = {r.id: r.is_online for r in self._store.list_roots()}
            after_roots = self._store.refresh_roots_online()
            after = {r.id: r.is_online for r in after_roots}
            changed = before != after
            if self._active_folder is not None:
                rid, rel = self._active_folder
                if self._store.refresh_folder_online(rid, rel):
                    changed = True
        except Exception:
            changed = False
        try:
            self._signals.finished.emit(changed)
        except RuntimeError:
            pass


class _PeakSignals(QObject):
    loaded = Signal(int, int, object)  # (gen, file_id, PeakData)
    failed = Signal(int, int, str)


class _PeakTask(QRunnable):
    def __init__(
        self,
        store: LibraryStore,
        generation: int,
        file_id: int,
        path: Path,
        mtime_ns: int,
        signals: _PeakSignals,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        require_spectral: bool = False,
    ) -> None:
        super().__init__()
        self._store = store
        self._generation = generation
        self._file_id = file_id
        self._path = path
        self._mtime_ns = mtime_ns
        self._signals = signals
        self._start_seconds = start_seconds
        self._end_seconds = end_seconds
        self._require_spectral = require_spectral

    @Slot()
    def run(self) -> None:
        is_overview = self._start_seconds <= 1e-4 and self._end_seconds is None
        try:
            if is_overview:
                cached = self._store.get_cached_peaks(
                    self._file_id, self._mtime_ns, OVERVIEW_POINTS
                )
                if cached is not None and (
                    not self._require_spectral or cached.spectral is not None
                ):
                    try:
                        self._signals.loaded.emit(self._generation, self._file_id, cached)
                    except RuntimeError:
                        pass
                    return

            data = compute_file_peaks(
                self._path,
                points=OVERVIEW_POINTS,
                mtime_ns=self._mtime_ns,
                start_seconds=self._start_seconds,
                end_seconds=self._end_seconds,
            )
            if is_overview:
                self._store.save_cached_peaks(
                    self._file_id,
                    self._mtime_ns,
                    data,
                    requested_points=OVERVIEW_POINTS,
                )
        except Exception as exc:
            try:
                self._signals.failed.emit(self._generation, self._file_id, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.loaded.emit(self._generation, self._file_id, data)
        except RuntimeError:
            pass


class _ScanSignals(QObject):
    progress = Signal(object)  # ScanProgress
    done = Signal(object)  # ScanReport
    failed = Signal(str)


class _AiSearchSignals(QObject):
    finished = Signal(int, object)  # (generation, AiSearchResponse)
    failed = Signal(int, str)
    customDone = Signal(object, object)  # (callback, result)
    customFailed = Signal(object, str)  # (callback, error_message)


class _AiSearchWorker:
    """Single persistent Python worker thread for PyTorch/CUDA AI searches & R5b enrichment.

    Windows PySide6 QThreadPool worker threads can deadlock inside CUDA/PyTorch
    transfers across separate threads; keeping all AI calls on one dedicated
    thread with latest-wins coalescing avoids both deadlocks and request pileups.
    """

    def __init__(self, signals: _AiSearchSignals) -> None:
        self._signals = signals
        self._cond = threading.Condition()
        self._pending: tuple[Any, int, str, dict[str, Any]] | None = None
        self._custom_queue: list[tuple[Any, Any, Any]] = []
        self._latest_gen = 0
        self._stop = False
        self._close_engine: Any | None = None
        self._thread: threading.Thread | None = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="AudioSearchAiWorker", daemon=True
            )
            self._thread.start()

    def submit(
        self,
        engine: Any,
        generation: int,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        with self._cond:
            if self._stop:
                return
            self._latest_gen = max(self._latest_gen, generation)
            self._pending = (engine, generation, kind, payload)
            self._ensure_thread_locked()
            self._cond.notify()

    def run_custom(
        self,
        fn: Any,
        on_done: Any,
        on_error: Any,
    ) -> None:
        with self._cond:
            if self._stop:
                return
            self._custom_queue.append((fn, on_done, on_error))
            self._ensure_thread_locked()
            self._cond.notify()

    def invalidate(self, generation: int) -> None:
        with self._cond:
            self._latest_gen = max(self._latest_gen, generation)
            self._pending = None

    def shutdown(self, engine_to_close: Any | None = None, timeout: float = 2.0) -> None:
        thread: threading.Thread | None
        with self._cond:
            thread = self._thread
            self._pending = None
            self._custom_queue.clear()
            self._close_engine = engine_to_close
            self._stop = True
            self._cond.notify()
        if thread is None:
            if engine_to_close is not None and hasattr(engine_to_close, "close"):
                try:
                    engine_to_close.close()
                except Exception:
                    pass
            return
        thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._custom_queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    close_engine = self._close_engine
                    self._close_engine = None
                    break
                if self._custom_queue:
                    custom_job = self._custom_queue.pop(0)
                    job = None
                else:
                    custom_job = None
                    job = self._pending
                    self._pending = None
                    latest_gen = self._latest_gen

            if custom_job is not None:
                fn, on_done, on_error = custom_job
                try:
                    result = fn()
                except Exception as exc:
                    try:
                        self._signals.customFailed.emit(on_error, str(exc))
                    except RuntimeError:
                        pass
                else:
                    try:
                        self._signals.customDone.emit(on_done, result)
                    except RuntimeError:
                        pass
                continue

            assert job is not None
            engine, generation, kind, payload = job
            if generation < latest_gen:
                continue

            try:
                if kind == "semantic":
                    resp = engine.search_semantic(**payload)
                elif kind == "similar_file":
                    resp = engine.find_similar_file(**payload)
                else:
                    resp = engine.find_similar_range(**payload)
            except Exception as exc:
                with self._cond:
                    stale = self._stop or generation < self._latest_gen
                if not stale:
                    try:
                        self._signals.failed.emit(generation, str(exc))
                    except RuntimeError:
                        pass
                continue

            with self._cond:
                stale = self._stop or generation < self._latest_gen
            if not stale:
                try:
                    self._signals.finished.emit(generation, resp)
                except RuntimeError:
                    pass

        if close_engine is not None and hasattr(close_engine, "close"):
            try:
                close_engine.close()
            except Exception:
                pass


class _AiIndexSignals(QObject):
    progress = Signal(object)  # IndexProgress
    done = Signal(object)  # IndexReport
    failed = Signal(str)
    coverageChanged = Signal()


class _SilentUpdateSignals(QObject):
    finished = Signal(object)  # UpdateCheckResult


class _SilentUpdateTask(QRunnable):
    def __init__(self, signals: _SilentUpdateSignals) -> None:
        super().__init__()
        self._signals = signals

    @Slot()
    def run(self) -> None:
        res = check_for_updates(timeout=5.0)
        try:
            self._signals.finished.emit(res)
        except RuntimeError:
            pass


class BrowserWindow(QMainWindow):
    """Main standalone audio browser modeled on REAPER Media Explorer."""

    def __init__(
        self,
        db_path: str | Path = ".local/library.sqlite3",
        *,
        audition: AuditionController | None = None,
        spot_client: ReaperSpotClient | None = None,
        ai_engine: Any | None = None,
        ai_index: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Whistle")
        app_inst = QApplication.instance()
        self._use_native_frameless = (
            sys.platform == "win32"
            and app_inst is not None
            and app_inst.platformName() != "offscreen"
        )
        self._win32_frame_applied = False
        if self._use_native_frameless:
            self.setWindowFlags(
                Qt.WindowType.Window
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowMinMaxButtonsHint
                | Qt.WindowType.WindowSystemMenuHint
            )

        self.store = LibraryStore(db_path)
        self.store.refresh_roots_online()
        self._settings_path = settings_path_for_db(self.store.db_path)
        self._settings = load_settings(self._settings_path)
        self._ui_scale = normalize_ui_scale(self._settings.ui_scale)
        self._output_device_name = str(self._settings.output_device_name or "").strip()
        self._index_video_files = bool(self._settings.index_video_files)
        if app_inst is not None:
            apply_theme(app_inst, self._ui_scale)
        self._settings_ready = False
        self._remembered_col_widths: dict[str, int] = {key: width for key, _, width in COLUMNS}
        self._active_col_layout_mode: str = "keyword"
        self._col_layouts: dict[str, dict[str, Any]] = {}
        self._active_ai_limit: int = max(50, min(5000, int(self._settings.ai_result_limit)))
        self.resize(self._settings.window_width, self._settings.window_height)
        if self._settings.window_maximized:
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        self._render_dir = self.store.db_path.parent / RENDER_DIR_NAME
        prune_render_dir(self._render_dir, max_files=50)

        self.audition = audition or AuditionController(
            device_name=self._output_device_name,
            parent=self,
        )
        self._spot_client = spot_client or ReaperSpotClient()
        self._ai_index = ai_index
        self._ai_engine = ai_engine
        self._ai_owns_index = ai_index is None
        self._ai_owns_engine = ai_engine is None
        self._ai_mode = False
        self._similar_request: tuple[int, tuple[float, float] | None] | None = None
        self._ai_search_gen = 0
        self._ai_search_signals = _AiSearchSignals(self)
        self._ai_search_signals.finished.connect(self._on_ai_search_finished)
        self._ai_search_signals.failed.connect(self._on_ai_search_failed)
        self._ai_search_signals.customDone.connect(self._on_custom_ai_done)
        self._ai_search_signals.customFailed.connect(self._on_custom_ai_failed)
        self._ai_worker = _AiSearchWorker(self._ai_search_signals)
        self._ai_indexer: Any | None = None
        self._ai_index_thread: threading.Thread | None = None
        self._ai_prune_running = False
        self._ai_auto_pruned = False
        self._ai_index_signals = _AiIndexSignals(self)
        self._ai_index_signals.progress.connect(self._on_ai_index_progress)
        self._ai_index_signals.done.connect(self._on_ai_index_done)
        self._ai_index_signals.failed.connect(self._on_ai_index_failed)
        self._ai_index_signals.coverageChanged.connect(self._on_ai_coverage_reconciled)

        self._scanner: LibraryScanner | None = None
        self._scan_thread: threading.Thread | None = None
        self._scan_started_at: float = 0.0
        self._scan_signals = _ScanSignals(self)
        self._scan_signals.progress.connect(self._on_scan_progress)
        self._scan_signals.done.connect(self._on_scan_done)
        self._scan_signals.failed.connect(self._on_scan_failed)

        self._pool = QThreadPool.globalInstance()
        self.thumbnail_manager = ThumbnailManager(self.store, parent=self)
        self._query_gen = 0
        self._query_signals = _QuerySignals(self)
        self._query_signals.finished.connect(self._on_query_finished)
        self._query_signals.failed.connect(self._on_query_failed)

        self._peak_gen = 0
        self._peak_signals = _PeakSignals(self)
        self._peak_signals.loaded.connect(self._on_peaks_loaded)
        self._peak_signals.failed.connect(self._on_peaks_failed)

        self._location = TreeLocation(None, None, "", "全部素材")
        self._history: list[TreeLocation] = [self._location]
        self._history_idx = 0
        self._current_file: FileRecord | None = None
        self._selection: tuple[float, float] | None = None
        self._pending_seek_sec: float = 0.0
        self._sort_column = self._settings.sort_column
        self._sort_asc = self._settings.sort_asc
        self._saved_sort_before_recent: tuple[str, bool] | None = None
        self._active_sidebar: SidebarViewTarget | None = None
        self._dup_mode = False
        self._dup_scan_mode = "name_duration"
        self._dup_scanner: DuplicateScanner | None = None
        self._dup_thread: threading.Thread | None = None
        self._dup_signals = _DupSignals(self)
        self._dup_signals.progress.connect(self._on_dup_scan_progress)
        self._dup_signals.done.connect(self._on_dup_scan_done)
        self._dup_signals.failed.connect(self._on_dup_scan_failed)
        self._online_signals = _OnlineSignals(self)
        self._online_signals.finished.connect(self._on_online_check_finished)
        self._online_checking = False
        self._latest_update_result: UpdateCheckResult | None = None
        self._update_signals = _SilentUpdateSignals(self)
        self._update_signals.finished.connect(self._on_silent_update_checked)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(100)
        self._search_timer.timeout.connect(self._dispatch_query)

        self._history_record_timer = QTimer(self)
        self._history_record_timer.setSingleShot(True)
        self._history_record_timer.setInterval(900)
        self._history_record_timer.timeout.connect(self._record_keyword_history)

        self._pending_zoom: tuple[float, float] | None = None
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(80)
        self._zoom_timer.timeout.connect(self._load_zoom_peaks)

        self._reaper_timer = QTimer(self)
        self._reaper_timer.setInterval(2000)
        self._reaper_timer.timeout.connect(self._refresh_reaper_status)

        self._online_timer = QTimer(self)
        self._online_timer.setInterval(8000)
        self._online_timer.timeout.connect(self._schedule_online_check)

        self._save_settings_timer = QTimer(self)
        self._save_settings_timer.setSingleShot(True)
        self._save_settings_timer.setInterval(400)
        self._save_settings_timer.timeout.connect(self._flush_saved_settings)

        self._silent_video_state: str = "empty"
        self._silent_video_pos: float = 0.0
        self._silent_video_anchor_mono: float = 0.0
        self._silent_video_anchor_pos: float = 0.0
        self._silent_video_start: float = 0.0
        self._silent_video_end: float | None = None
        self._silent_video_timer = QTimer(self)
        self._silent_video_timer.setInterval(33)
        self._silent_video_timer.timeout.connect(self._tick_silent_video)

        self._build_ui()
        self._apply_saved_settings(self._settings)
        self._wire_signals()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.folder_tree.reload(keep_location=self._location)
        self._update_nav_buttons()
        self._update_device_status()
        self._refresh_reaper_status()
        self._refresh_ai_coverage()
        self._refresh_dirty_status()
        self._reaper_timer.start()
        self._online_timer.start()
        self._update_transport_time(0.0)
        self._settings_ready = True
        mark_patch_boot_healthy()
        if is_frozen():
            QTimer.singleShot(3500, self._schedule_silent_update_check)
        self._dispatch_query()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        spec = ui_scale_spec(self._ui_scale)
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self, self)
        root_layout.addWidget(self.title_bar)

        top_bar = QFrame(self)
        top_bar.setObjectName("topBar")
        top_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout = QGridLayout(top_bar)
        top_layout.setContentsMargins(10, 8, 10, 8)
        top_layout.setHorizontalSpacing(6)
        top_layout.setVerticalSpacing(6)
        top_layout.setColumnStretch(0, 0)
        top_layout.setColumnStretch(1, 1)

        # Row 0: Folder navigation (col 0 fills mode_track width) + path & actions (col 1)
        nav_group = QWidget(top_bar)
        nav_group.setStyleSheet("background: transparent;")
        nav_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        nav_group_layout = QHBoxLayout(nav_group)
        nav_group_layout.setContentsMargins(0, 0, 0, 0)
        nav_group_layout.setSpacing(4)

        self.back_btn = QPushButton("")
        self.back_btn.setIcon(_make_nav_icon("left"))
        self.back_btn.setIconSize(QSize(16, 16))
        self.back_btn.setFixedHeight(spec.btn_h)
        self.back_btn.setMinimumWidth(0)
        self.back_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.back_btn.setProperty("variant", "ghost")
        self.back_btn.setProperty("iconBtn", True)
        self.back_btn.setToolTip("后退 (Alt+←)")

        self.forward_btn = QPushButton("")
        self.forward_btn.setIcon(_make_nav_icon("right"))
        self.forward_btn.setIconSize(QSize(16, 16))
        self.forward_btn.setFixedHeight(spec.btn_h)
        self.forward_btn.setMinimumWidth(0)
        self.forward_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.forward_btn.setProperty("variant", "ghost")
        self.forward_btn.setProperty("iconBtn", True)
        self.forward_btn.setToolTip("前进 (Alt+→)")

        self.up_btn = QPushButton("")
        self.up_btn.setIcon(_make_nav_icon("up"))
        self.up_btn.setIconSize(QSize(16, 16))
        self.up_btn.setFixedHeight(spec.btn_h)
        self.up_btn.setMinimumWidth(0)
        self.up_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.up_btn.setProperty("variant", "ghost")
        self.up_btn.setProperty("iconBtn", True)
        self.up_btn.setToolTip("上级目录")

        nav_group_layout.addWidget(self.back_btn, 1)
        nav_group_layout.addWidget(self.forward_btn, 1)
        nav_group_layout.addWidget(self.up_btn, 1)

        nav_right = QHBoxLayout()
        nav_right.setContentsMargins(0, 0, 0, 0)
        nav_right.setSpacing(6)
        self.path_edit = QLineEdit("全部素材")
        self.path_edit.setObjectName("pathEdit")
        self.path_edit.setReadOnly(True)

        self.add_folder_btn = QPushButton("+ 添加文件夹")
        self.add_folder_btn.setProperty("variant", "secondary")
        self.scope_btn = QPushButton("范围：当前目录")
        self.scope_btn.setCheckable(True)
        self.scope_btn.setProperty("variant", "chip")
        self.scope_btn.setToolTip(
            "仅在搜索或 AI 找同类时生效：开启后搜全库，关闭则限定当前目录；搜索框为空时点左侧目录始终直接浏览"
        )
        self.inspector_btn = QPushButton("检查器")
        self.inspector_btn.setCheckable(True)
        self.inspector_btn.setChecked(True)
        self.inspector_btn.setProperty("variant", "ghost")

        self.scan_label = QLabel("")
        self.scan_label.setProperty("role", "muted")
        self.scan_pause_btn = QPushButton("暂停扫描")
        self.scan_pause_btn.setProperty("variant", "ghost")
        self.scan_pause_btn.setProperty("compact", True)
        self.scan_pause_btn.setVisible(False)

        nav_right.addWidget(self.path_edit, 1)
        nav_right.addWidget(self.scope_btn)
        nav_right.addWidget(self.add_folder_btn)
        nav_right.addWidget(self.scan_label)
        nav_right.addWidget(self.scan_pause_btn)
        nav_right.addWidget(self.inspector_btn)

        top_layout.addWidget(nav_group, 0, 0)
        top_layout.addLayout(nav_right, 0, 1)

        # Row 1: Hero mode switch (col 0) + search box & actions (col 1)
        self.mode_track = QFrame(top_bar)
        self.mode_track.setObjectName("heroModeTrack")
        self.mode_track.setFixedHeight(spec.input_h)
        mode_box = QHBoxLayout(self.mode_track)
        mode_box.setContentsMargins(2, 2, 2, 2)
        mode_box.setSpacing(2)
        mode_btn_h = max(24, spec.input_h - 6)
        self.mode_keyword_btn = QPushButton("关键词")
        self.mode_keyword_btn.setCheckable(True)
        self.mode_keyword_btn.setChecked(True)
        self.mode_keyword_btn.setFixedHeight(mode_btn_h)
        self.mode_keyword_btn.setProperty("variant", "modeKw")
        self.mode_keyword_btn.setToolTip("即时关键词与 REAPER ME 语法检索（0 延迟）")

        self.mode_ai_btn = QPushButton("✦ AI 语义")
        self.mode_ai_btn.setCheckable(True)
        self.mode_ai_btn.setFixedHeight(mode_btn_h)
        self.mode_ai_btn.setProperty("variant", "modeAi")
        self.mode_ai_btn.setToolTip("中文自然语言语义 + 声波双塔融合检索（按 Enter 触发）")

        mode_box.addWidget(self.mode_keyword_btn)
        mode_box.addWidget(self.mode_ai_btn)

        search_right = QHBoxLayout()
        search_right.setContentsMargins(0, 0, 0, 0)
        search_right.setSpacing(6)
        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setFixedHeight(spec.input_h)
        self.search_input.setProperty("aiMode", False)
        self.search_input.setPlaceholderText(
            '搜索音效（即时过滤：空格AND / OR / NOT,-排除 / "精确短语" / *通配符）'
        )
        self.search_input.setClearButtonEnabled(True)

        self.ai_search_btn = QPushButton("检索")
        self.ai_search_btn.setFixedHeight(spec.input_h)
        self.ai_search_btn.setProperty("variant", "primary")
        self.ai_search_btn.setVisible(False)
        self.ai_search_btn.setToolTip("执行 AI 语义融合检索 (Enter)")

        self.history_btn = QPushButton("历史 ▾")
        self.history_btn.setProperty("variant", "ghost")
        self.history_btn.setToolTip("查看、置顶或复用最近搜索历史")

        self.filter_toggle_btn = QPushButton("筛选")
        self.filter_toggle_btn.setCheckable(True)
        self.filter_toggle_btn.setProperty("variant", "ghost")
        self.filter_toggle_btn.setToolTip("展开/收起格式、时长、收藏与离线快捷筛选栏")

        self.count_label = QLabel("0 条")
        self.count_label.setFont(mono_font(spec.mono_pt))
        self.count_label.setProperty("role", "muted")
        self.count_label.setStyleSheet("padding-right: 4px;")

        search_right.addWidget(self.search_input, 1)
        search_right.addWidget(self.ai_search_btn)
        search_right.addWidget(self.history_btn)
        search_right.addWidget(self.filter_toggle_btn)
        search_right.addWidget(self.count_label)

        top_layout.addWidget(self.mode_track, 1, 0)
        top_layout.addLayout(search_right, 1, 1)
        root_layout.addWidget(top_bar)

        # Collapsible quick-filter strip (docked inside center_pane to avoid global reflow jitter)
        self.filter_bar = QFrame(self)
        self.filter_bar.setObjectName("filterBar")
        self.filter_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_layout = QHBoxLayout(self.filter_bar)
        filter_layout.setContentsMargins(10, 5, 10, 5)
        filter_layout.setSpacing(10)

        self.filter_ext_combo = QComboBox()
        for label, val in (
            ("全部格式", ""),
            ("仅音频", "__audio__"),
            ("仅视频", "__video__"),
            ("WAV", "wav"),
            ("OGG", "ogg"),
            ("MP3", "mp3"),
            ("FLAC", "flac"),
            ("AIF", "aif"),
            ("MP4", "mp4"),
            ("MOV", "mov"),
            ("MKV", "mkv"),
        ):
            self.filter_ext_combo.addItem(label, val)

        self.filter_dur_combo = QComboBox()
        for label, val in (
            ("全部时长", ""),
            ("< 1s 瞬态", "lt1"),
            ("1 - 5s 短效", "1_5"),
            ("5 - 15s 中段", "5_15"),
            ("> 15s 环境/循环", "gt15"),
        ):
            self.filter_dur_combo.addItem(label, val)

        self.filter_fav_cb = QCheckBox("★ 仅看收藏")
        self.filter_online_cb = QCheckBox("隐藏离线")
        self.filter_reset_btn = QPushButton("重置")
        self.filter_reset_btn.setProperty("variant", "ghost")
        self.filter_reset_btn.setProperty("compact", True)

        ext_lbl = QLabel("格式:")
        ext_lbl.setProperty("role", "muted")
        dur_lbl = QLabel("时长:")
        dur_lbl.setProperty("role", "muted")
        filter_layout.addWidget(ext_lbl)
        filter_layout.addWidget(self.filter_ext_combo)
        filter_layout.addWidget(dur_lbl)
        filter_layout.addWidget(self.filter_dur_combo)
        filter_layout.addWidget(self.filter_fav_cb)
        filter_layout.addWidget(self.filter_online_cb)
        filter_layout.addWidget(self.filter_reset_btn)
        filter_layout.addStretch(1)
        self.filter_bar.setVisible(False)

        # Bottom-docked AI status / Find-Similar strip (kept inside center_pane to avoid page jump)
        self.ai_banner = QWidget(self)
        self.ai_banner.setObjectName("aiBanner")
        ai_banner_layout = QHBoxLayout(self.ai_banner)
        ai_banner_layout.setContentsMargins(10, 5, 10, 5)
        ai_banner_layout.setSpacing(8)
        self.ai_banner_label = QLabel("")
        self.ai_banner_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.ai_banner_label.setMinimumWidth(0)
        self.ai_banner_label.setProperty("role", "muted")
        self.ai_limit_combo = QComboBox()
        for opt_val in (200, 500, 1000, 2000, 5000):
            self.ai_limit_combo.addItem(f"上限 {opt_val}", opt_val)
        self.ai_limit_combo.setCurrentIndex(1)
        self.ai_limit_combo.setToolTip("设置 AI 语义 / 找同类单次返回的结果条数上限")
        self.ai_load_more_btn = QPushButton("加载更多")
        self.ai_load_more_btn.setProperty("variant", "chip")
        self.ai_load_more_btn.setProperty("compact", True)
        self.ai_load_more_btn.setVisible(False)
        self.ai_load_more_btn.setToolTip("放宽返回上限并重新检索更多相关结果")
        self.exclude_same_pkg_btn = QPushButton("排除同目录/同包")
        self.exclude_same_pkg_btn.setCheckable(True)
        self.exclude_same_pkg_btn.setProperty("variant", "chip")
        self.exclude_same_pkg_btn.setProperty("compact", True)
        self.exclude_same_pkg_btn.setVisible(False)
        self.ai_exit_similar_btn = QPushButton("返回列表")
        self.ai_exit_similar_btn.setProperty("variant", "ghost")
        self.ai_exit_similar_btn.setProperty("compact", True)
        self.ai_exit_similar_btn.setVisible(False)
        ai_banner_layout.addWidget(self.ai_banner_label, 1)
        ai_banner_layout.addWidget(self.ai_limit_combo)
        ai_banner_layout.addWidget(self.ai_load_more_btn)
        ai_banner_layout.addWidget(self.exclude_same_pkg_btn)
        ai_banner_layout.addWidget(self.ai_exit_similar_btn)
        self.ai_banner.setVisible(False)

        # Bottom-docked read-only duplicate finder bar (kept inside center_pane)
        self.dup_banner = QWidget(self)
        self.dup_banner.setObjectName("dupBanner")
        dup_layout = QHBoxLayout(self.dup_banner)
        dup_layout.setContentsMargins(10, 5, 10, 5)
        dup_layout.setSpacing(6)
        self.dup_banner_label = QLabel("安全查重（只读扫描，绝不删原文件）")
        self.dup_banner_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.dup_banner_label.setMinimumWidth(0)
        self.dup_banner_label.setProperty("role", "muted")

        dup_track = QFrame(self.dup_banner)
        dup_track.setObjectName("segTrack")
        dup_seg = QHBoxLayout(dup_track)
        dup_seg.setContentsMargins(2, 2, 2, 2)
        dup_seg.setSpacing(2)
        self.dup_mode_name_btn = QPushButton("同源多格式")
        self.dup_mode_name_btn.setCheckable(True)
        self.dup_mode_name_btn.setChecked(True)
        self.dup_mode_name_btn.setProperty("variant", "seg")
        self.dup_mode_name_btn.setProperty("compact", True)
        self.dup_mode_name_btn.setToolTip("按同名文件词干 + 时长(±0.15s) 识别 WAV/OGG/MP3 跨格式重复")
        self.dup_mode_exact_btn = QPushButton("完全字节重复")
        self.dup_mode_exact_btn.setCheckable(True)
        self.dup_mode_exact_btn.setProperty("variant", "seg")
        self.dup_mode_exact_btn.setProperty("compact", True)
        self.dup_mode_exact_btn.setToolTip("按字节大小 + SHA-256 哈希精确识别完全相同文件")
        dup_seg.addWidget(self.dup_mode_name_btn)
        dup_seg.addWidget(self.dup_mode_exact_btn)

        self.dup_scan_btn = QPushButton("开始查重")
        self.dup_scan_btn.setProperty("variant", "primary")
        self.dup_scan_btn.setProperty("compact", True)
        self.dup_hide_all_btn = QPushButton("隐藏非主版本")
        self.dup_hide_all_btn.setProperty("variant", "chip")
        self.dup_hide_all_btn.setProperty("compact", True)
        self.dup_hide_all_btn.setToolTip("仅在列表隐藏全部非主版本副本（磁盘原文件绝不改动）")
        self.dup_unhide_all_btn = QPushButton("恢复已隐藏")
        self.dup_unhide_all_btn.setProperty("variant", "chip")
        self.dup_unhide_all_btn.setProperty("compact", True)
        self.dup_exit_btn = QPushButton("返回列表")
        self.dup_exit_btn.setProperty("variant", "ghost")
        self.dup_exit_btn.setProperty("compact", True)

        dup_layout.addWidget(self.dup_banner_label, 1)
        dup_layout.addWidget(dup_track)
        dup_layout.addWidget(self.dup_scan_btn)
        dup_layout.addWidget(self.dup_hide_all_btn)
        dup_layout.addWidget(self.dup_unhide_all_btn)
        dup_layout.addWidget(self.dup_exit_btn)
        self.dup_banner.setVisible(False)

        # Main vertical splitter: top (sidebar | table+context bars | inspector) + bottom (transport + waveform)
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setHandleWidth(1)

        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_splitter.setHandleWidth(1)
        self.folder_tree = FolderTree(self.store)
        self.folder_tree.setMinimumWidth(160)

        self.table_model = FileTableModel(self.store, thumbnail_manager=self.thumbnail_manager)
        self.table_view = FileTableView(
            self.table_model, thumbnail_manager=self.thumbnail_manager
        )
        self.center_pane = QWidget(self)
        center_layout = QVBoxLayout(self.center_pane)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.filter_bar)
        center_layout.addWidget(self.table_view, 1)
        center_layout.addWidget(self.ai_banner)
        center_layout.addWidget(self.dup_banner)

        self.inspector = InspectorPanel()

        self.h_splitter.addWidget(self.folder_tree)
        self.h_splitter.addWidget(self.center_pane)
        self.h_splitter.addWidget(self.inspector)
        self.h_splitter.setStretchFactor(0, 0)
        self.h_splitter.setStretchFactor(1, 1)
        self.h_splitter.setStretchFactor(2, 0)

        bottom_pane = QFrame(self)
        bottom_pane.setObjectName("bottomDock")
        bottom_layout = QVBoxLayout(bottom_pane)
        bottom_layout.setContentsMargins(10, 6, 10, 8)
        bottom_layout.setSpacing(6)

        # Transport & audition controls bar
        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(5)

        self._icon_play = _make_transport_icon("play")
        self._icon_pause = _make_transport_icon("pause")
        self._icon_stop = _make_transport_icon("stop")

        self.play_btn = QPushButton("")
        self.play_btn.setIcon(self._icon_play)
        self.play_btn.setIconSize(QSize(14, 14))
        self.play_btn.setFixedSize(int(round(36 * spec.scale)), spec.btn_h)
        self.play_btn.setProperty("variant", "transportPlay")
        self.play_btn.setProperty("playing", False)
        self.play_btn.setProperty("iconBtn", True)
        self.play_btn.setToolTip("播放 / 暂停 (Space)")
        self.stop_btn = QPushButton("")
        self.stop_btn.setIcon(self._icon_stop)
        self.stop_btn.setIconSize(QSize(12, 12))
        self.stop_btn.setFixedSize(int(round(30 * spec.scale)), spec.btn_h)
        self.stop_btn.setProperty("variant", "ghost")
        self.stop_btn.setProperty("iconBtn", True)
        self.stop_btn.setToolTip("停止 (Esc)")

        self.autoplay_btn = QPushButton("自动播放")
        self.autoplay_btn.setCheckable(True)
        self.autoplay_btn.setProperty("variant", "chip")
        self.autoplay_btn.setToolTip("开启后单击切换行或上下键切换行时自动从头播放")

        self.loop_btn = QPushButton("循环")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setProperty("variant", "chip")
        self.loop_btn.setToolTip("循环播放整条音频或当前选区")

        self.volume_knob = DragValueControl(
            "音量",
            minimum=-48.0,
            maximum=6.0,
            default=0.0,
            value_format=lambda v: "静音" if v <= -48.0 else ("0.0 dB" if abs(v) < 0.05 else f"{v:+.1f} dB"),
            drag_range_px=150.0,
            wheel_step=1.0,
            decimals=1,
        )
        self.pitch_knob = DragValueControl(
            "变调",
            minimum=-24.0,
            maximum=24.0,
            default=0.0,
            value_format=lambda v: "0.0 st" if abs(v) < 0.05 else f"{v:+.1f} st",
            drag_range_px=160.0,
            wheel_step=1.0,
            decimals=1,
        )
        self.rate_knob = DragValueControl(
            "变速",
            minimum=0.25,
            maximum=4.0,
            default=1.0,
            value_format=lambda v: f"{v:.2f}x",
            drag_range_px=160.0,
            wheel_step=0.05,
            decimals=2,
        )

        self.preserve_pitch_btn = QPushButton("保留音高")
        self.preserve_pitch_btn.setCheckable(True)
        self.preserve_pitch_btn.setProperty("variant", "chip")
        self.preserve_pitch_btn.setToolTip("开启后变调与变速独立生效；关闭为传统 Varispeed 联动变调")

        self.reset_tone_btn = QPushButton("重置")
        self.reset_tone_btn.setProperty("variant", "ghost")
        self.reset_tone_btn.setToolTip("复位变调 0 st 与变速 1.00x")

        self.reset_switch_btn = QPushButton("切行重置")
        self.reset_switch_btn.setCheckable(True)
        self.reset_switch_btn.setChecked(True)
        self.reset_switch_btn.setProperty("variant", "chip")
        self.reset_switch_btn.setToolTip("切换文件试听时自动重置变调与变速")

        self.bake_drag_btn = QPushButton("应用调音")
        self.bake_drag_btn.setCheckable(True)
        self.bake_drag_btn.setChecked(True)
        self.bake_drag_btn.setProperty("variant", "chip")
        self.bake_drag_btn.setToolTip(
            "拖出到 DAW 或双击 / Enter 插入 REAPER 时应用当前变调 / 变速 / 保留音高"
        )

        self.spectral_peaks_btn = QPushButton("频谱波形")
        self.spectral_peaks_btn.setCheckable(True)
        self.spectral_peaks_btn.setProperty("variant", "chip")
        self.spectral_peaks_btn.setToolTip(
            "REAPER Spectral Peaks 风格频谱着色：\n"
            "红/橙=低频 · 黄/绿=中频 · 青/蓝/紫=高频 · 灰=宽频噪声/瞬态"
        )

        self.transport_time_label = QLabel("0:00.000 / 0:00.000")
        self.transport_time_label.setFont(mono_font(spec.mono_pt))
        self.transport_time_label.setProperty("role", "muted")

        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addWidget(self.autoplay_btn)
        transport.addWidget(self.loop_btn)
        transport.addSpacing(2)
        transport.addWidget(self.volume_knob)
        transport.addWidget(self.pitch_knob)
        transport.addWidget(self.rate_knob)
        transport.addWidget(self.preserve_pitch_btn)
        transport.addWidget(self.reset_tone_btn)
        transport.addWidget(self.reset_switch_btn)
        transport.addWidget(self.bake_drag_btn)
        transport.addSpacing(4)
        transport.addWidget(self.spectral_peaks_btn)
        transport.addStretch(1)
        transport.addWidget(self.transport_time_label)

        self.waveform = WaveformWidget()
        wf_frame = QFrame(bottom_pane)
        wf_frame.setObjectName("waveformFrame")
        wf_layout = QVBoxLayout(wf_frame)
        wf_layout.setContentsMargins(0, 0, 0, 0)
        wf_layout.addWidget(self.waveform, 1)

        bottom_layout.addLayout(transport)
        bottom_layout.addWidget(wf_frame, 1)

        self.v_splitter.addWidget(self.h_splitter)
        self.v_splitter.addWidget(bottom_pane)
        self.v_splitter.setStretchFactor(0, 1)
        self.v_splitter.setStretchFactor(1, 0)

        root_layout.addWidget(self.v_splitter, 1)

        # Status bar (segments + background AI indexing status)
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self.status_total = QLabel("总库: 0 条")
        self.status_view = QLabel("当前: 0 条")
        self.status_audio = QPushButton("声卡: 检测中")
        self.status_audio.setFlat(True)
        self.status_audio.setProperty("variant", "statusLink")
        self.status_audio.setProperty("statusState", "idle")
        self.status_audio.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_audio.setToolTip("点击切换或刷新音频输出设备")
        self.status_audio.clicked.connect(lambda _c=False: self._open_settings_dialog("audio"))

        self.status_mode = QPushButton("REAPER 未连接")
        self.status_mode.setFlat(True)
        self.status_mode.setProperty("variant", "statusLink")
        self.status_mode.setProperty("statusState", "idle")
        self.status_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_mode.clicked.connect(self._open_reaper_setup_dialog)
        self.status_ai = QLabel("AI 索引: 未建库")
        self.write_back_btn = QPushButton("待写回: 0")
        self.write_back_btn.setProperty("variant", "chip")
        self.write_back_btn.setProperty("compact", True)
        self.write_back_btn.setVisible(False)
        self.write_back_btn.setToolTip("预览待写回源文件的元数据修改、安全写盘或撤销历史批次")

        self.ai_index_btn = QPushButton("AI 建库")
        self.ai_index_btn.setProperty("variant", "chip")
        self.ai_index_btn.setProperty("compact", True)
        self.ai_index_btn.setToolTip("后台增量建立本地 Qwen3 + CLAP2 向量索引（不阻塞浏览与试听）")

        self.status_version = QPushButton(f"v{effective_version()}")
        self.status_version.setFlat(True)
        self.status_version.setProperty("variant", "statusLink")
        self.status_version.setProperty("statusState", "idle")
        self.status_version.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_version.setToolTip("当前版本 · 点击检查 Whistle 在线更新（增量热更新不改动本地曲库）")
        self.status_version.clicked.connect(self._open_update_dialog)

        for widget in (self.status_total, self.status_view):
            widget.setProperty("role", "muted")
            widget.setStyleSheet("padding: 0 10px;")
            bar.addWidget(widget)
        bar.addWidget(self.status_audio)
        bar.addWidget(self.status_mode)
        self.status_ai.setProperty("role", "muted")
        self.status_ai.setStyleSheet("padding: 0 10px;")
        bar.addWidget(self.status_ai)
        bar.addPermanentWidget(self.write_back_btn)
        bar.addPermanentWidget(self.ai_index_btn)
        bar.addPermanentWidget(self.status_version)

    def _apply_saved_settings(self, s: BrowserSettings) -> None:
        self.autoplay_btn.setChecked(s.autoplay)
        self.loop_btn.setChecked(s.loop)
        self.audition.set_loop(s.loop)

        vol_db = max(-48.0, min(6.0, float(s.volume_db)))
        self.volume_knob.setValue(vol_db)
        self._apply_volume_db(vol_db)

        self.preserve_pitch_btn.setChecked(s.preserve_pitch)
        self.audition.set_preserve_pitch(s.preserve_pitch)
        self.reset_switch_btn.setChecked(s.reset_on_switch)
        self.bake_drag_btn.setChecked(s.bake_on_drag)
        self.spectral_peaks_btn.setChecked(s.spectral_peaks)
        self.waveform.set_spectral_peaks(s.spectral_peaks)
        self.table_view.set_spectral_peaks(s.spectral_peaks)

        self.inspector_btn.setChecked(s.inspector_visible)
        self.inspector.setVisible(s.inspector_visible)

        self.filter_online_cb.setChecked(s.hide_offline)
        self.filter_toggle_btn.setChecked(s.filter_bar_visible)
        self.filter_bar.setVisible(s.filter_bar_visible)
        self.scope_btn.setChecked(s.global_scope)
        self.scope_btn.setText("范围：全局搜索" if s.global_scope else "范围：当前目录")
        self.folder_tree.set_local_roots_expanded(s.local_roots_expanded)
        self.folder_tree.set_collapsed_collection_ids(s.collapsed_collections)

        self.h_splitter.setSizes(s.h_splitter_sizes)
        self.v_splitter.setSizes(s.v_splitter_sizes)

        header = self.table_view.horizontalHeader()
        col_keys = [col[0] for col in COLUMNS]
        if s.sort_column in col_keys and s.sort_column not in NON_SORTABLE_COL_KEYS:
            sort_idx = col_keys.index(s.sort_column)
            header.setSortIndicator(
                sort_idx,
                Qt.SortOrder.AscendingOrder if s.sort_asc else Qt.SortOrder.DescendingOrder,
            )

        self._active_ai_limit = max(50, min(5000, int(s.ai_result_limit)))
        self._set_ai_limit_combo(self._active_ai_limit)
        self._init_column_layouts(s)
        self._apply_col_layout("keyword", snapshot_previous=False)
        self.exclude_same_pkg_btn.setChecked(s.exclude_same_package_similar)
        self.apply_ui_scale(s.ui_scale, persist=False)
        self._index_video_files = bool(s.index_video_files)

    @staticmethod
    def _merge_column_order(saved_order_raw: list[str]) -> list[str]:
        col_keys = [col[0] for col in COLUMNS]
        saved_order = [k for k in saved_order_raw if k in col_keys]
        seen_order: set[str] = set()
        merged_order: list[str] = []
        for k in saved_order:
            if k not in seen_order:
                seen_order.add(k)
                merged_order.append(k)
        if not merged_order:
            return list(DEFAULT_COLUMN_ORDER)
        for default_idx, key in enumerate(col_keys):
            if key not in seen_order:
                if key == MARK_COL_KEY and "filename" in merged_order:
                    insert_at = merged_order.index("filename")
                elif key == "waveform" and "filename" in merged_order:
                    insert_at = merged_order.index("filename") + 1
                else:
                    insert_at = min(default_idx, len(merged_order))
                merged_order.insert(insert_at, key)
                seen_order.add(key)
        return merged_order

    def _init_column_layouts(self, s: BrowserSettings) -> None:
        default_widths = {key: width for key, _, width in COLUMNS}
        kw_widths = {
            key: (
                s.column_widths[key]
                if s.column_widths.get(key, 0) >= 24
                else default_widths[key]
            )
            for key, _, _ in COLUMNS
        }
        kw_order = self._merge_column_order(s.column_order)
        kw_hidden = (set(s.hidden_columns) - {"filename"}) - set(AI_COL_KEYS)

        has_saved_ai = bool(s.ai_column_widths or s.ai_column_order or s.ai_hidden_columns)
        ai_widths = dict(kw_widths)
        for key in AI_COL_KEYS:
            ai_widths[key] = default_widths[key]
        if s.ai_column_widths:
            for key, default_w in default_widths.items():
                w = s.ai_column_widths.get(key, 0)
                if w >= 24:
                    ai_widths[key] = w
                elif key in AI_COL_KEYS:
                    ai_widths[key] = default_w
        ai_order = self._merge_column_order(s.ai_column_order or kw_order)
        ai_hidden = (set(s.ai_hidden_columns) - {"filename"}) if has_saved_ai else set(kw_hidden)

        self._col_layouts = {
            "keyword": {"widths": kw_widths, "order": kw_order, "hidden": kw_hidden},
            "ai": {"widths": ai_widths, "order": ai_order, "hidden": ai_hidden},
        }
        self._remembered_col_widths = dict(kw_widths)

    def _snapshot_current_col_layout(self, mode: str) -> None:
        if self._dup_mode or mode not in self._col_layouts:
            return
        header = self.table_view.horizontalHeader()
        col_keys = [col[0] for col in COLUMNS]
        default_widths = {key: width for key, _, width in COLUMNS}
        layout = self._col_layouts[mode]
        widths: dict[str, int] = dict(layout["widths"])
        for idx, key in enumerate(col_keys):
            w = self.table_view.columnWidth(idx)
            if not self.table_view.isColumnHidden(idx) and w >= 24:
                widths[key] = int(w)
            elif key not in widths or widths[key] < 24:
                widths[key] = default_widths[key]
        order = [col_keys[header.logicalIndex(vis)] for vis in range(len(col_keys))]
        hidden = {
            key
            for idx, key in enumerate(col_keys)
            if key != "filename"
            and self.table_view.isColumnHidden(idx)
            and (mode == "ai" or key not in AI_COL_KEYS)
        }
        layout["widths"] = widths
        layout["order"] = order
        layout["hidden"] = hidden
        self._remembered_col_widths = widths

    def _apply_col_layout(self, mode: str, *, snapshot_previous: bool = True) -> None:
        if mode not in self._col_layouts:
            return
        if snapshot_previous and self._settings_ready and not self._dup_mode:
            self._snapshot_current_col_layout(self._active_col_layout_mode)
        self._active_col_layout_mode = mode
        layout = self._col_layouts[mode]
        widths: dict[str, int] = layout["widths"]
        order: list[str] = self._merge_column_order(layout["order"])
        hidden_set: set[str] = set(layout["hidden"]) - {"filename"}
        if mode == "keyword":
            hidden_set |= set(AI_COL_KEYS)

        col_keys = [col[0] for col in COLUMNS]
        default_widths = {key: width for key, _, width in COLUMNS}
        header = self.table_view.horizontalHeader()
        self.table_view.setUpdatesEnabled(False)
        blocked = header.blockSignals(True)
        try:
            for target_visual, key in enumerate(order):
                logical = col_keys.index(key)
                current_visual = header.visualIndex(logical)
                if current_visual != target_visual:
                    header.moveSection(current_visual, target_visual)
            for logical_idx, key in enumerate(col_keys):
                target_w = int(widths.get(key, default_widths[key]))
                if target_w < 24:
                    target_w = default_widths[key]
                widths[key] = target_w
                self.table_view.setColumnHidden(logical_idx, key in hidden_set)
                self.table_view.setColumnWidth(logical_idx, target_w)
            self.table_view._refresh_first_visible_col()
            self._remembered_col_widths = widths
        finally:
            header.blockSignals(blocked)
            self.table_view.setUpdatesEnabled(True)
        self.table_view.viewport().update()

    @Slot(int, int, int)
    def _on_header_section_resized(self, logical_idx: int, _old_size: int, new_size: int) -> None:
        if (
            0 <= logical_idx < len(COLUMNS)
            and new_size >= 24
            and not self.table_view.isColumnHidden(logical_idx)
        ):
            key = COLUMNS[logical_idx][0]
            if not self._dup_mode and self._active_col_layout_mode in self._col_layouts:
                self._col_layouts[self._active_col_layout_mode]["widths"][key] = int(new_size)
            self._remembered_col_widths[key] = int(new_size)
            self._schedule_save_settings()

    @Slot()
    def _on_header_layout_mutated(self, *_args: object) -> None:
        if self._settings_ready and not self._dup_mode:
            self._snapshot_current_col_layout(self._active_col_layout_mode)
        self._schedule_save_settings()

    @Slot()
    def _schedule_save_settings(self) -> None:
        if self._settings_ready:
            self._save_settings_timer.start()

    @Slot()
    def _flush_saved_settings(self) -> None:
        if self._save_settings_timer.isActive():
            self._save_settings_timer.stop()
        self._settings = self._collect_settings()
        save_settings(self._settings_path, self._settings)

    def _collect_settings(self) -> BrowserSettings:
        if self._settings_ready and not self._dup_mode:
            self._snapshot_current_col_layout(self._active_col_layout_mode)
        kw = self._col_layouts.get("keyword", {})
        ai = self._col_layouts.get("ai", {})
        col_keys = [col[0] for col in COLUMNS]
        default_widths = {key: width for key, _, width in COLUMNS}
        kw_widths = {
            key: int(kw.get("widths", {}).get(key, default_widths[key]))
            for key in col_keys
        }
        ai_widths = {
            key: int(ai.get("widths", {}).get(key, default_widths[key]))
            for key in col_keys
        }
        kw_order = list(kw.get("order") or DEFAULT_COLUMN_ORDER)
        ai_order = list(ai.get("order") or kw_order)
        kw_hidden = sorted((set(kw.get("hidden", set())) - {"filename"}) - set(AI_COL_KEYS))
        ai_hidden = sorted(set(ai.get("hidden", set())) - {"filename"})
        saved_sort = (
            self._saved_sort_before_recent[0]
            if self._saved_sort_before_recent is not None
            else self._sort_column
        )
        saved_sort_asc = (
            self._saved_sort_before_recent[1]
            if self._saved_sort_before_recent is not None
            else self._sort_asc
        )
        if saved_sort in AI_COL_KEYS or saved_sort == "played_at":
            saved_sort = "filename"
        is_max = self.isMaximized()
        if is_max:
            ng = self.normalGeometry()
            win_w = ng.width() if ng.width() >= 640 else self._settings.window_width
            win_h = ng.height() if ng.height() >= 480 else self._settings.window_height
        else:
            win_w = max(640, self.width())
            win_h = max(480, self.height())
        return BrowserSettings(
            autoplay=self.autoplay_btn.isChecked(),
            loop=self.loop_btn.isChecked(),
            volume_db=float(self.volume_knob.value()),
            preserve_pitch=self.preserve_pitch_btn.isChecked(),
            reset_on_switch=self.reset_switch_btn.isChecked(),
            bake_on_drag=self.bake_drag_btn.isChecked(),
            inspector_visible=self.inspector_btn.isChecked(),
            sort_column=saved_sort,
            sort_asc=saved_sort_asc,
            window_width=win_w,
            window_height=win_h,
            window_maximized=is_max,
            h_splitter_sizes=self.h_splitter.sizes(),
            v_splitter_sizes=self.v_splitter.sizes(),
            column_widths=kw_widths,
            column_order=kw_order,
            hidden_columns=kw_hidden,
            ai_column_widths=ai_widths,
            ai_column_order=ai_order,
            ai_hidden_columns=ai_hidden,
            ai_result_limit=self._current_ai_limit(),
            ai_profile=self._settings.ai_profile,
            exclude_same_package_similar=self.exclude_same_pkg_btn.isChecked(),
            spectral_peaks=self.spectral_peaks_btn.isChecked(),
            hide_offline=self.filter_online_cb.isChecked(),
            filter_bar_visible=self.filter_toggle_btn.isChecked(),
            global_scope=self.scope_btn.isChecked(),
            local_roots_expanded=self.folder_tree.is_local_roots_expanded(),
            collapsed_collections=self.folder_tree.collapsed_collection_ids(),
            sidebar_splitter_sizes=self._settings.sidebar_splitter_sizes,
            ui_scale=self._ui_scale,
            output_device_name=self._output_device_name,
            index_video_files=self._index_video_files,
        )

    def _wire_signals(self) -> None:
        self.title_bar.settingsRequested.connect(
            lambda: self._open_settings_dialog("appearance")
        )
        self.back_btn.clicked.connect(self.go_back)
        self.forward_btn.clicked.connect(self.go_forward)
        self.up_btn.clicked.connect(self.go_up)
        self.add_folder_btn.clicked.connect(self._choose_and_add_root)
        self.scope_btn.toggled.connect(self._on_scope_toggled)
        self.inspector_btn.toggled.connect(self.inspector.setVisible)
        self.scan_pause_btn.clicked.connect(self._toggle_scan_pause)

        self.play_btn.clicked.connect(self._on_play_button)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.loop_btn.toggled.connect(self.audition.set_loop)
        self.volume_knob.valueChanged.connect(self._apply_volume_db)
        self.pitch_knob.valueChanged.connect(self._on_pitch_rate_changed)
        self.rate_knob.valueChanged.connect(self._on_pitch_rate_changed)
        self.preserve_pitch_btn.toggled.connect(self.audition.set_preserve_pitch)
        self.reset_tone_btn.clicked.connect(self._reset_tone)
        self.spectral_peaks_btn.toggled.connect(self._on_spectral_peaks_toggled)

        self.table_view.drag_path_resolver = self._resolve_table_drag_path
        self.table_view.collections_provider = (
            lambda: [(c.id, c.name) for c in self.store.list_collections()]
        )
        self.table_view.active_collection_id_provider = self._active_collection_id

        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._on_search_return_pressed)
        self.mode_keyword_btn.clicked.connect(lambda _c=False: self._set_search_mode(False))
        self.mode_ai_btn.clicked.connect(lambda _c=False: self._set_search_mode(True))
        self.ai_search_btn.clicked.connect(self._dispatch_ai_search)
        self.history_btn.clicked.connect(self._show_history_menu)
        self.filter_toggle_btn.toggled.connect(self._on_filter_bar_toggled)
        self.filter_ext_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.filter_dur_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.filter_fav_cb.toggled.connect(self._on_filter_changed)
        self.filter_online_cb.toggled.connect(self._on_hide_offline_toggled)
        self.filter_reset_btn.clicked.connect(self._reset_filters)

        self.ai_limit_combo.currentIndexChanged.connect(self._on_ai_limit_changed)
        self.ai_load_more_btn.clicked.connect(self._on_ai_load_more)
        self.exclude_same_pkg_btn.toggled.connect(self._on_exclude_same_pkg_toggled)
        self.ai_exit_similar_btn.clicked.connect(self._exit_similar_mode)
        self.write_back_btn.clicked.connect(lambda _c=False: self._open_write_back_dialog(None))
        self.ai_index_btn.clicked.connect(self._toggle_ai_indexing)

        self.dup_mode_name_btn.clicked.connect(
            lambda _c=False: self._set_duplicate_mode("name_duration")
        )
        self.dup_mode_exact_btn.clicked.connect(lambda _c=False: self._set_duplicate_mode("exact"))
        self.dup_scan_btn.clicked.connect(self._toggle_duplicate_scan)
        self.dup_hide_all_btn.clicked.connect(self._on_hide_all_non_primary)
        self.dup_unhide_all_btn.clicked.connect(self._on_unhide_all_duplicates)
        self.dup_exit_btn.clicked.connect(self._exit_duplicate_mode)

        self.folder_tree.viewSelected.connect(self._on_sidebar_view_selected)
        self.folder_tree.createCollectionRequested.connect(self._on_create_collection)
        self.folder_tree.renameCollectionRequested.connect(self._on_rename_collection)
        self.folder_tree.deleteCollectionRequested.connect(self._on_delete_collection)
        self.folder_tree.removeCollectionFolderRequested.connect(
            self._on_remove_collection_folder
        )
        self.folder_tree.filesDroppedOnCollection.connect(self._on_files_dropped_on_collection)
        self.folder_tree.urlsDroppedOnCollection.connect(self._on_urls_dropped_on_collection)
        self.folder_tree.rootsReordered.connect(self._on_roots_reordered)
        self.folder_tree.collectionsReordered.connect(self._on_collections_reordered)
        self.folder_tree.collectionFolderMoved.connect(self._on_collection_folder_moved)
        self.folder_tree.importReaperShortcutsRequested.connect(
            lambda: self._on_import_reaper_shortcuts()
        )

        self.folder_tree.locationSelected.connect(self._on_tree_location_selected)
        self.folder_tree.rescanRootRequested.connect(self.start_scan_root)
        self.folder_tree.removeRootRequested.connect(self._remove_root)
        self.folder_tree.relocateRootRequested.connect(self._on_relocate_root)
        self.folder_tree.relocateFolderRequested.connect(self._on_relocate_folder)
        self.folder_tree.refreshOnlineRequested.connect(self._refresh_online_now)
        self.folder_tree.addFolderToCollectionRequested.connect(
            self._on_add_folder_to_collection
        )

        self.table_model.sortChanged.connect(self._on_sort_changed)
        self.table_view.auditionRowRequested.connect(self._on_audition_row)
        self.table_view.waveformSeekRequested.connect(self._on_waveform_cell_seek)
        self.table_view.spotRowRequested.connect(self._on_spot_row)
        self.table_view.spotFilesRequested.connect(self._on_spot_files)
        self.table_view.findSimilarRowRequested.connect(self._on_find_similar_row)
        self.table_view.toggleFavoriteRequested.connect(self._on_toggle_favorite)
        self.table_view.batchFavoriteRequested.connect(self._on_batch_favorite)
        self.table_view.setPlayedMarkRequested.connect(self._on_set_played_marks)
        self.table_view.clearPlayedMarksRequested.connect(self._on_clear_played_marks)
        self.table_view.addFilesToCollectionRequested.connect(self._on_files_dropped_on_collection)
        self.table_view.removeFilesFromCollectionRequested.connect(
            self._on_remove_files_from_active_collection
        )
        self.table_view.hideDuplicatesRequested.connect(self._on_hide_duplicates)
        self.table_view.setDuplicatePrimaryRequested.connect(self._on_set_duplicate_primary)
        self.table_view.folderActivated.connect(self._on_table_folder_activated)
        self.table_view.navigateUpRequested.connect(self.go_up)
        self.table_view.folderRelocateRequested.connect(self._on_relocate_folder)
        self.table_view.folderAddToCollectionRequested.connect(self._on_add_folder_to_collection)
        self.table_view.navigateToFileFolderRequested.connect(self._on_navigate_to_file_folder)
        self.table_model.inlineMetadataEdited.connect(self._on_inline_metadata_edited)
        self.table_view.editMetadataRequested.connect(self._on_edit_metadata_requested)
        self.table_view.parseUcsFilenameRequested.connect(self._on_parse_ucs_requested)
        self.table_view.aiEnrichMetadataRequested.connect(self._on_ai_enrich_requested)
        self.table_view.batchRenameRequested.connect(self._on_batch_rename_requested)
        self.table_view.csvMetadataRequested.connect(self._on_csv_metadata_requested)
        self.table_view.writeBackMetadataRequested.connect(self._on_write_back_requested)
        self.table_view.statusMessageRequested.connect(
            lambda msg: self.statusBar().showMessage(msg, 6000)
        )
        self.thumbnail_manager.thumbnailsReady.connect(self._on_thumbnails_ready)

        self.inspector.favoriteToggled.connect(self._on_inspector_favorite_toggled)
        self.inspector.noteChanged.connect(self._on_inspector_note_changed)
        self.inspector.editMetadataRequested.connect(
            lambda fid: self._on_edit_metadata_requested([fid])
        )
        self.inspector.navigateToFolderRequested.connect(self._on_navigate_to_file_folder)
        self.inspector.statusMessageRequested.connect(
            lambda msg: self.statusBar().showMessage(msg, 6000)
        )
        self.inspector.togglePlayRequested.connect(self._on_play_button)

        self.waveform.seekRequested.connect(self._on_waveform_seek)
        self.waveform.selectionChanged.connect(self._on_selection_changed)
        self.waveform.selectionCommitted.connect(self._on_selection_committed)
        self.waveform.selectionDragRequested.connect(self._on_selection_drag)
        self.waveform.zoomRangeRequested.connect(self._on_zoom_range_requested)
        self.waveform.findSimilarRangeRequested.connect(self._on_find_similar_waveform)
        self.waveform.spectralPeaksToggled.connect(self.spectral_peaks_btn.setChecked)

        self.audition.positionChanged.connect(self._on_audition_position)
        self.audition.stateChanged.connect(self._on_audition_state)
        self.audition.failed.connect(self._on_audition_failed)

        # Debounced auto-save for layout, column widths/order, sidebar state, and toggles
        header = self.table_view.horizontalHeader()
        header.sectionResized.connect(self._on_header_section_resized)
        header.sectionMoved.connect(self._on_header_layout_mutated)
        self.table_view.columnVisibilityChanged.connect(self._on_header_layout_mutated)
        self.table_model.sortChanged.connect(lambda *_args: self._schedule_save_settings())
        self.h_splitter.splitterMoved.connect(lambda *_args: self._schedule_save_settings())
        self.v_splitter.splitterMoved.connect(lambda *_args: self._schedule_save_settings())
        self.folder_tree.layoutStateChanged.connect(self._schedule_save_settings)
        for toggle_btn in (
            self.autoplay_btn,
            self.loop_btn,
            self.preserve_pitch_btn,
            self.reset_switch_btn,
            self.bake_drag_btn,
            self.spectral_peaks_btn,
            self.inspector_btn,
            self.scope_btn,
            self.filter_online_cb,
            self.filter_toggle_btn,
            self.exclude_same_pkg_btn,
        ):
            toggle_btn.toggled.connect(lambda _checked=False: self._schedule_save_settings())
        self.volume_knob.valueChanged.connect(lambda _v=0.0: self._schedule_save_settings())

    # -- Keyboard & Focus (global event filter) ----------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if isinstance(watched, QWidget) and (watched is self or self.isAncestorOf(watched)):
                if self._handle_key_press(event):
                    return True
        return super().eventFilter(watched, event)

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        key = event.key()
        mods = event.modifiers()
        focus = self.focusWidget()
        in_text_input = isinstance(focus, (QLineEdit, QAbstractSpinBox, QPlainTextEdit))

        if key == Qt.Key.Key_F and mods == Qt.KeyboardModifier.ControlModifier:
            self.search_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.search_input.selectAll()
            return True

        if key == Qt.Key.Key_Comma and mods == Qt.KeyboardModifier.ControlModifier:
            self._open_settings_dialog("appearance")
            return True

        if key == Qt.Key.Key_Left and mods == Qt.KeyboardModifier.AltModifier:
            self.go_back()
            return True

        if key == Qt.Key.Key_Right and mods == Qt.KeyboardModifier.AltModifier:
            self.go_forward()
            return True

        if key == Qt.Key.Key_Escape:
            if self._selection is not None:
                self._clear_selection()
            else:
                self._stop_playback()
            return True

        if key == Qt.Key.Key_Space and not in_text_input:
            if not event.isAutoRepeat():
                self._on_play_button()
            return True

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not in_text_input:
            if not event.isAutoRepeat():
                row = self.table_view.currentIndex().row()
                if self.focusWidget() is self.table_view and row >= 0:
                    folder_entry = self.table_model.folder_at(row)
                    if folder_entry is not None:
                        self._on_table_folder_activated(folder_entry[0])
                        return True
                    selected_recs = self.table_view.selected_records()
                    if len(selected_recs) > 1:
                        self._spot_records(selected_recs)
                        return True
                rec = self._current_file
                if rec is None and row >= 0:
                    rec = self.table_model.record_at(row)
                if rec is not None:
                    self._spot_record(rec)
            return True

        return False

    # -- Navigation --------------------------------------------------------
    def _clear_search_for_directory_browse(self) -> bool:
        if not self.scope_btn.isChecked():
            return False
        has_kw = (not self._ai_mode) and bool(self.search_input.text().strip())
        has_sim = self._similar_request is not None
        if not has_kw and not has_sim:
            return False
        self._search_timer.stop()
        self._history_record_timer.stop()
        if has_kw:
            blocked = self.search_input.blockSignals(True)
            try:
                self.search_input.clear()
            finally:
                self.search_input.blockSignals(blocked)
        if has_sim:
            self._similar_request = None
            self.exclude_same_pkg_btn.setVisible(False)
            self.ai_exit_similar_btn.setVisible(False)
            if not self._ai_mode:
                self._apply_col_layout("keyword")
                self.ai_banner.setVisible(False)
                self.ai_load_more_btn.setVisible(False)
        return True

    def _set_location(
        self,
        loc: TreeLocation,
        *,
        record_history: bool,
        reveal_file_id: int | None = None,
    ) -> None:
        if loc == self._location:
            if reveal_file_id is not None:
                self.table_model.clear_played_marks()
                self.path_edit.setText(loc.display_path)
                self._update_nav_buttons()
                self._refresh_active_view(from_navigation=True, reveal_file_id=reveal_file_id)
            return
        self.table_model.clear_played_marks()
        self._location = loc
        self.path_edit.setText(loc.display_path)
        if record_history:
            del self._history[self._history_idx + 1 :]
            self._history.append(loc)
            self._history_idx = len(self._history) - 1
        self._update_nav_buttons()
        self._refresh_active_view(from_navigation=True, reveal_file_id=reveal_file_id)

    def _refresh_active_view(
        self, *, from_navigation: bool = False, reveal_file_id: int | None = None
    ) -> None:
        if self._similar_request is not None:
            file_id, seed_range = self._similar_request
            self._start_find_similar(file_id, seed_range)
        elif self._ai_mode:
            has_query = bool(self.search_input.text().strip())
            if not from_navigation and self.table_model.is_ai_mode and has_query:
                self._dispatch_ai_search()
                return
            if has_query:
                msg = (
                    "已切换视图（保留搜索词）；按 Enter 或点击「检索」在当前范围执行 AI 语义搜索"
                    if from_navigation
                    else "已更新筛选条件；按 Enter 或点击「检索」在当前范围执行 AI 语义搜索"
                )
                self.ai_banner_label.setText(msg)
            else:
                self._update_ai_banner_idle()
            self._dispatch_query(reveal_file_id=reveal_file_id)
        else:
            self._dispatch_query(reveal_file_id=reveal_file_id)

    def _update_nav_buttons(self) -> None:
        self.back_btn.setEnabled(self._history_idx > 0)
        self.forward_btn.setEnabled(self._history_idx < len(self._history) - 1)
        self.up_btn.setEnabled(self._location.root_id is not None)

    @Slot()
    def go_back(self) -> None:
        if self._history_idx <= 0:
            return
        self._history_idx -= 1
        loc = self._history[self._history_idx]
        was_sidebar = self._active_sidebar is not None
        self._active_sidebar = None
        self.folder_tree.select_location(loc.root_id, loc.folder_id)
        if loc == self._location and was_sidebar:
            self.table_model.clear_played_marks()
            self.path_edit.setText(loc.display_path)
            self._update_nav_buttons()
            self._refresh_active_view(from_navigation=True)
            return
        self._set_location(loc, record_history=False)

    @Slot()
    def go_forward(self) -> None:
        if self._history_idx >= len(self._history) - 1:
            return
        self._history_idx += 1
        loc = self._history[self._history_idx]
        was_sidebar = self._active_sidebar is not None
        self._active_sidebar = None
        self.folder_tree.select_location(loc.root_id, loc.folder_id)
        if loc == self._location and was_sidebar:
            self.table_model.clear_played_marks()
            self.path_edit.setText(loc.display_path)
            self._update_nav_buttons()
            self._refresh_active_view(from_navigation=True)
            return
        self._set_location(loc, record_history=False)

    @Slot()
    def go_up(self) -> None:
        loc = self._location
        if loc.root_id is None:
            return
        if loc.folder_id is None:
            target = TreeLocation(None, None, "", "全部素材")
        else:
            folder = self.store.get_folder(loc.folder_id)
            root = self.store.get_root(loc.root_id)
            if folder is None or folder.parent_id is None:
                display = root.path if root is not None else "根目录"
                target = TreeLocation(loc.root_id, None, "", display)
            else:
                parent = self.store.get_folder(folder.parent_id)
                if parent is None:
                    display = root.path if root is not None else "根目录"
                    target = TreeLocation(loc.root_id, None, "", display)
                else:
                    display = str(Path(root.path) / Path(parent.rel_path)) if root else parent.rel_path
                    target = TreeLocation(loc.root_id, parent.id, parent.rel_path, display)
        was_sidebar = self._active_sidebar is not None
        self._active_sidebar = None
        self.folder_tree.select_location(target.root_id, target.folder_id)
        if target == self._location and was_sidebar:
            self.table_model.clear_played_marks()
            self.path_edit.setText(target.display_path)
            self._update_nav_buttons()
            self._refresh_active_view(from_navigation=True)
            return
        self._set_location(target, record_history=True)

    @Slot(object)
    def _on_tree_location_selected(self, loc: TreeLocation) -> None:
        self._leave_recent_sort_if_needed()
        was_cleared = self._clear_search_for_directory_browse() if loc.root_id is not None else False
        was_sidebar = self._active_sidebar is not None
        self._active_sidebar = None
        was_dup = self._dup_mode
        if self._dup_mode:
            self._dup_mode = False
            self.dup_banner.setVisible(False)
            self._apply_col_layout(
                "ai" if (self._ai_mode or self._similar_request is not None) else "keyword",
                snapshot_previous=False,
            )
        if loc == self._location:
            if was_sidebar or was_dup or was_cleared:
                self.table_model.clear_played_marks()
            self.path_edit.setText(loc.display_path)
            self._update_nav_buttons()
            self._refresh_active_view(from_navigation=True)
            return
        self._set_location(loc, record_history=True)

    @Slot(bool)
    def _on_scope_toggled(self, global_mode: bool) -> None:
        self.scope_btn.setText("范围：全局搜索" if global_mode else "范围：当前目录")
        self._refresh_active_view()

    # -- Search & Table ----------------------------------------------------
    @Slot(str)
    def _on_search_text_changed(self, text: str) -> None:
        if self._ai_mode:
            if not text.strip() and self._similar_request is None:
                self._update_ai_banner_idle()
                self._dispatch_query()
            return
        if self._similar_request is not None or self._dup_mode:
            return
        self._search_timer.start()
        if text.strip():
            self._history_record_timer.start()
        else:
            self._history_record_timer.stop()

    @Slot()
    def _on_search_return_pressed(self) -> None:
        if self._ai_mode:
            self._similar_request = None
            self.exclude_same_pkg_btn.setVisible(False)
            self.ai_exit_similar_btn.setVisible(False)
            self._dispatch_ai_search()
        elif not self._dup_mode:
            self._history_record_timer.stop()
            self._record_keyword_history()

    def _set_search_mode(self, ai_enabled: bool, *, dispatch: bool = True) -> None:
        self.mode_keyword_btn.setChecked(not ai_enabled)
        self.mode_ai_btn.setChecked(ai_enabled)
        self._ai_mode = ai_enabled
        self.search_input.setProperty("aiMode", bool(ai_enabled))
        st = self.search_input.style()
        if st is not None:
            st.unpolish(self.search_input)
            st.polish(self.search_input)
        self.search_input.update()
        self._similar_request = None
        self._ai_search_gen += 1
        self._ai_worker.invalidate(self._ai_search_gen)
        if self._dup_mode:
            self._dup_mode = False
            self.dup_banner.setVisible(False)
        self.ai_search_btn.setVisible(ai_enabled)
        self.exclude_same_pkg_btn.setVisible(False)
        self.ai_exit_similar_btn.setVisible(False)
        if ai_enabled:
            self.search_input.setPlaceholderText(
                "输入中文自然语言描述搜索（如：清脆金币奖励 / 怪物低吼 / 火焰呼啸 -爆炸），按 Enter 检索"
            )
            self._apply_col_layout("ai")
            self.ai_banner.setVisible(True)
            if self.search_input.text().strip():
                self.ai_banner_label.setText(
                    "已切到 AI 语义模式（保留搜索词）；按 Enter 或点击「检索」开始搜索"
                )
                self.ai_load_more_btn.setVisible(False)
                if dispatch:
                    self._dispatch_query()
            else:
                self._update_ai_banner_idle()
        else:
            self.search_input.setPlaceholderText(
                '搜索音效（即时过滤：空格AND / OR / NOT,-排除 / "精确短语" / *通配符）'
            )
            self._apply_col_layout("keyword")
            self.ai_banner.setVisible(False)
            self.ai_load_more_btn.setVisible(False)
            self._sync_sort_indicator(self._sort_column, self._sort_asc)
            if dispatch:
                self._dispatch_query()

    @Slot(str, bool)
    def _on_sort_changed(self, sort_column: str, ascending: bool) -> None:
        if self.table_model.is_ai_mode or self.table_model.is_duplicate_mode:
            self.table_model.sort_ai_records(sort_column, ascending)
            return
        self._saved_sort_before_recent = None
        self._sort_column = sort_column
        self._sort_asc = ascending
        self._dispatch_query()

    def _current_filters(self, *, ignore_sidebar_scope: bool = False) -> QueryFilter:
        ext = str(self.filter_ext_combo.currentData() or "") or None
        dur_key = str(self.filter_dur_combo.currentData() or "")
        min_dur: float | None = None
        max_dur: float | None = None
        if dur_key == "lt1":
            max_dur = 1.0
        elif dur_key == "1_5":
            min_dur, max_dur = 1.0, 5.0
        elif dur_key == "5_15":
            min_dur, max_dur = 5.0, 15.0
        elif dur_key == "gt15":
            min_dur = 15.0
        fav_only = self.filter_fav_cb.isChecked()
        online_only = self.filter_online_cb.isChecked()
        recent_only = False
        coll_id: int | None = None
        if not ignore_sidebar_scope and self._active_sidebar is not None:
            if self._active_sidebar.kind == KIND_FAVORITES:
                fav_only = True
            elif self._active_sidebar.kind == KIND_RECENT:
                recent_only = True
            elif self._active_sidebar.kind == KIND_COLLECTION:
                coll_id = self._active_sidebar.collection_id
        return QueryFilter(
            ext=ext,
            min_duration=min_dur,
            max_duration=max_dur,
            favorites_only=fav_only,
            online_only=online_only,
            recent_only=recent_only,
            collection_id=coll_id,
            include_hidden=False,
        )

    def _sync_sort_indicator(self, sort_column: str, ascending: bool) -> None:
        col_keys = [c[0] for c in COLUMNS]
        if sort_column not in col_keys or sort_column in NON_SORTABLE_COL_KEYS:
            return
        sort_idx = col_keys.index(sort_column)
        header = self.table_view.horizontalHeader()
        blocked = header.blockSignals(True)
        try:
            header.setSortIndicator(
                sort_idx,
                Qt.SortOrder.AscendingOrder if ascending else Qt.SortOrder.DescendingOrder,
            )
        finally:
            header.blockSignals(blocked)

    def _is_directory_browse_mode(self, scope: dict[str, Any], filters: QueryFilter) -> bool:
        if self._similar_request is not None or self._dup_mode:
            return False
        if (not self._ai_mode) and self.search_input.text().strip():
            return False
        if scope["root_id"] is None:
            return False
        if self._active_sidebar is not None and self._active_sidebar.kind != KIND_COLLECTION_FOLDER:
            return False
        return not (
            filters.ext
            or filters.min_duration is not None
            or filters.max_duration is not None
            or filters.favorites_only
            or filters.recent_only
            or filters.collection_id is not None
        )

    def _dispatch_query(self, *, reveal_file_id: int | None = None) -> None:
        if self._dup_mode:
            self._load_duplicate_table()
            return
        self._ai_search_gen += 1
        self._ai_worker.invalidate(self._ai_search_gen)
        self._query_gen += 1
        query_text = "" if self._ai_mode else self.search_input.text()
        is_abs_path = (not self._ai_mode) and (extract_absolute_path_query(query_text) is not None)
        is_global = self._is_global_search_active(for_search=False) or is_abs_path
        scope = self._current_scope_args(global_override=is_global)
        filters = self._current_filters(ignore_sidebar_scope=is_global)
        include_sub = not self._is_directory_browse_mode(scope, filters)
        params = (
            query_text,
            scope["root_id"],
            scope["folder_id"],
            scope["folder_rel_path"],
            include_sub,
            self._sort_column,
            self._sort_asc,
            filters,
        )
        self._pool.start(
            _QueryTask(
                self.store,
                self._query_gen,
                params,
                self._query_signals,
                reveal_file_id=reveal_file_id,
            )
        )

    @Slot(int, int, list, list, tuple, object)
    def _on_query_finished(
        self,
        generation: int,
        total_count: int,
        first_page: list[FileRecord],
        folders: list[tuple[Any, str, bool]],
        params: tuple,
        reveal_payload: object = None,
    ) -> None:
        if generation != self._query_gen or self._dup_mode:
            return
        query, root_id, folder_id, rel_path, include_sub, sort_col, sort_asc, filters = params
        extra_pages = (
            reveal_payload[2]
            if isinstance(reveal_payload, tuple) and len(reveal_payload) == 3
            else None
        )
        self.table_model.apply_snapshot(
            total_count=total_count,
            first_page=first_page,
            query=query,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=rel_path,
            include_subfolders=include_sub,
            sort_column=sort_col,
            sort_asc=sort_asc,
            filters=filters,
            folders=folders,
            extra_pages=extra_pages,
        )
        self._sync_sort_indicator(sort_col, sort_asc)
        if isinstance(reveal_payload, tuple) and len(reveal_payload) == 3:
            _target_id, file_pos, _extra = reveal_payload
            row = len(folders) + int(file_pos)
            if 0 <= row < self.table_model.rowCount():
                idx = self.table_model.index(row, 0)
                self.table_view.setCurrentIndex(idx)
                self.table_view.selectRow(row)
                self.table_view.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
        total_lib = self.store.total_file_count()
        if folders:
            parts = [f"{len(folders):,} 个文件夹"]
            if total_count > 0:
                parts.append(f"{total_count:,} 条")
            view_str = " · ".join(parts)
        else:
            view_str = f"{total_count:,} 条"
        self.count_label.setText(view_str)
        self.status_total.setText(f"总库: {total_lib:,} 条")
        self.status_view.setText(f"当前: {view_str}")

    @Slot(int, str)
    def _on_query_failed(self, generation: int, message: str) -> None:
        if generation != self._query_gen:
            return
        self.status_view.setText(f"查询异常: {message}")

    # -- Transport, Selection, Peaks & Audition ----------------------------
    @Slot(float)
    def _apply_volume_db(self, value: float) -> None:
        self.audition.set_volume_db(float(value))

    @Slot(float)
    def _on_pitch_rate_changed(self, _value: float = 0.0) -> None:
        self.audition.set_pitch_rate(self.pitch_knob.value(), self.rate_knob.value())

    @Slot()
    def _reset_tone(self) -> None:
        self.pitch_knob.reset()
        self.rate_knob.reset()
        self.audition.set_pitch_rate(0.0, 1.0)

    def _should_bake_drag(self) -> bool:
        return self.bake_drag_btn.isChecked() and (
            abs(self.pitch_knob.value()) > 1e-4 or abs(self.rate_knob.value() - 1.0) > 1e-4
        )

    def _resolve_table_drag_path(self, rec: FileRecord) -> Path:
        if not self._should_bake_drag():
            return rec.full_path
        try:
            rendered = render_range(
                rec.full_path,
                start_seconds=0.0,
                end_seconds=None,
                out_dir=self._render_dir,
                pitch_semitones=self.pitch_knob.value(),
                rate=self.rate_knob.value(),
                preserve_pitch=self.preserve_pitch_btn.isChecked(),
            )
            prune_render_dir(self._render_dir, max_files=50)
            return rendered
        except Exception as exc:
            self.status_audio.setText(f"拖出烘焙失败: {exc}")
            return rec.full_path

    @staticmethod
    def _is_silent_video(rec: FileRecord | None) -> bool:
        return rec is not None and is_video_path(rec.full_path) and rec.sample_rate <= 0

    @Slot()
    def _on_stop_clicked(self) -> None:
        self._stop_playback()

    def _stop_playback(self) -> None:
        self.audition.stop()
        self._stop_silent_video(reset_pos=True)

    def _start_silent_video(
        self,
        rec: FileRecord,
        start_seconds: float = 0.0,
        *,
        end_seconds: float | None = None,
    ) -> None:
        self.audition.stop()
        self._silent_video_timer.stop()
        dur = max(0.0, float(rec.duration_sec))
        start = max(0.0, min(float(start_seconds), max(0.0, dur - 0.02))) if dur > 0.0 else 0.0
        end = None if end_seconds is None else max(start + 0.02, min(float(end_seconds), dur))
        self._silent_video_start = start
        self._silent_video_end = end
        self._silent_video_pos = start
        self._silent_video_anchor_pos = start
        self._silent_video_anchor_mono = time.monotonic()
        self._silent_video_state = "playing"
        self.table_model.set_playing_file(rec.id)
        ratio = max(0.0, min(1.0, start / dur)) if dur > 0.0 else 0.0
        self.table_view.set_playing_progress(rec.id, ratio)
        self.waveform.set_playback_head(start)
        self._update_transport_time(start)
        self._update_play_btn_visual(True)
        self.inspector.set_video_playing(True)
        self.inspector.sync_video_time(start)
        self._silent_video_timer.start()

    def _pause_silent_video(self) -> None:
        if self._silent_video_state != "playing":
            return
        self._silent_video_timer.stop()
        rate = max(0.25, float(self.rate_knob.value()))
        elapsed = max(0.0, time.monotonic() - self._silent_video_anchor_mono) * rate
        self._silent_video_pos = self._silent_video_anchor_pos + elapsed
        self._silent_video_state = "paused"
        self.table_model.set_playing_file(None)
        self._update_play_btn_visual(False)
        self.inspector.set_video_playing(False)

    def _resume_silent_video(self) -> None:
        rec = self._current_file
        if not self._is_silent_video(rec) or self._silent_video_state != "paused":
            return
        assert rec is not None
        self._silent_video_anchor_pos = self._silent_video_pos
        self._silent_video_anchor_mono = time.monotonic()
        self._silent_video_state = "playing"
        self.table_model.set_playing_file(rec.id)
        self._update_play_btn_visual(True)
        self.inspector.set_video_playing(True)
        self._silent_video_timer.start()

    def _stop_silent_video(self, *, reset_pos: bool = True, clear_state: bool = False) -> None:
        was_active = self._silent_video_state != "empty" or self._silent_video_timer.isActive()
        self._silent_video_timer.stop()
        self._silent_video_state = "empty" if clear_state else "paused"
        if reset_pos:
            self._silent_video_pos = 0.0
        if was_active:
            self.waveform.set_playback_head(None)
            self.table_model.set_playing_file(None)
            self.table_view.set_playing_progress(None, None)
            self._update_transport_time(self._silent_video_pos)
            self._update_play_btn_visual(False)
            self.inspector.set_video_playing(False)
            if reset_pos and not clear_state and self._current_file is not None and is_video_path(self._current_file.full_path):
                self.inspector.sync_video_time(0.0)

    @Slot()
    def _tick_silent_video(self) -> None:
        rec = self._current_file
        if self._silent_video_state != "playing" or not self._is_silent_video(rec):
            self._silent_video_timer.stop()
            return
        assert rec is not None
        dur = max(0.0, float(rec.duration_sec))
        bound_end = self._silent_video_end if self._silent_video_end is not None else dur
        rate = max(0.25, float(self.rate_knob.value()))
        pos = self._silent_video_anchor_pos + max(0.0, time.monotonic() - self._silent_video_anchor_mono) * rate
        if bound_end > 0.0 and pos >= bound_end:
            if self.loop_btn.isChecked():
                pos = self._silent_video_start
                self._silent_video_anchor_pos = pos
                self._silent_video_anchor_mono = time.monotonic()
            else:
                self._stop_silent_video(reset_pos=True, clear_state=True)
                return
        self._silent_video_pos = pos
        ratio = max(0.0, min(1.0, pos / dur)) if dur > 0.0 else 0.0
        self.waveform.set_playback_head(pos)
        self.table_view.set_playing_progress(rec.id, ratio)
        self._update_transport_time(pos)
        self.inspector.sync_video_time(pos)

    def _play_file(
        self,
        rec: FileRecord,
        start_seconds: float = 0.0,
        *,
        end_seconds: float | None = None,
    ) -> None:
        if self._is_silent_video(rec):
            self._start_silent_video(rec, start_seconds, end_seconds=end_seconds)
        else:
            self._stop_silent_video(reset_pos=True, clear_state=True)
            self.audition.play_file(rec.id, rec.full_path, start_seconds, end_seconds=end_seconds)
        self.table_model.mark_played(rec.id)
        self.store.record_recent_play(rec.id)
        self.folder_tree.refresh_quick_counts()

    @Slot()
    def _on_play_button(self) -> None:
        rec = self._current_file
        if rec is None:
            row = self.table_view.currentIndex().row()
            if row >= 0:
                rec = self.table_model.record_at(row)
        if rec is None:
            return
        if self._is_silent_video(rec):
            if not rec.is_available or not rec.full_path.exists():
                return
            if self._silent_video_state == "playing":
                self._pause_silent_video()
                return
            if self._silent_video_state == "paused" and self._silent_video_pos > 1e-4:
                self._resume_silent_video()
                return
            start = self._selection[0] if self._selection is not None else self._pending_seek_sec
            end = self._selection[1] if self._selection is not None else None
            self._play_file(rec, start, end_seconds=end)
            return
        if self.audition.current_file_id == rec.id and self.audition.state in (
            "playing",
            "paused",
            "finished",
        ):
            if self.audition.state == "playing" or getattr(
                self.audition, "can_resume_buffered", True
            ):
                self.audition.toggle_pause_resume()
                return
        if not rec.is_available or not rec.full_path.exists():
            return
        start = self._selection[0] if self._selection is not None else self._pending_seek_sec
        end = self._selection[1] if self._selection is not None else None
        self._play_file(rec, start, end_seconds=end)

    def _prepare_file_selection(self, rec: FileRecord, *, ai_hit: Any | None = None) -> bool:
        self._stop_silent_video(reset_pos=True, clear_state=True)
        self._current_file = rec
        self._selection = None
        self._pending_seek_sec = 0.0
        self._pending_zoom = None
        self._zoom_timer.stop()
        self.audition.set_selection(None)
        fields = self.store.get_metadata_fields(rec.id)
        self.inspector.show_file(rec, fields, ai_hit=ai_hit)
        self._update_transport_time(0.0)

        if not rec.is_available or not rec.full_path.exists():
            if rec.is_available:
                rel_dir = Path(rec.rel_path).parent.as_posix()
                changed = self.store.mark_file_available(rec.id, False)
                if rel_dir and rel_dir != ".":
                    if self.store.refresh_folder_online(rec.root_id, rel_dir):
                        changed = True
                else:
                    if self.store.refresh_roots_online():
                        changed = True
                fresh = self.store.get_file(rec.id)
                if fresh is not None:
                    self._current_file = fresh
                    self.inspector.show_file(fresh, fields, ai_hit=ai_hit)
                if changed:
                    self.folder_tree.reload(
                        keep_location=self._location, keep_target=self._active_sidebar
                    )
                    if self.filter_online_cb.isChecked():
                        self._refresh_active_view()
                    else:
                        self.table_model.refresh_file_meta({rec.id})
                        self.table_view.repaint_file_waveform(rec.id)
            self.waveform.clear("文件离线或不存在")
            self.audition.stop()
            self.table_model.set_playing_file(None)
            self.table_view.set_playing_progress(None, None)
            self.statusBar().showMessage(f"文件离线或已移动：{rec.filename}（可右键目录重定位路径）", 4000)
            return False

        if self._is_silent_video(rec):
            self.waveform.set_loading(rec.duration_sec, "视频无音轨 · 可在右侧检查器预览画面")
            return True

        self.waveform.set_loading(rec.duration_sec, "正在读取波形…")
        self._peak_gen += 1
        self._pool.start(
            _PeakTask(
                self.store,
                self._peak_gen,
                rec.id,
                rec.full_path,
                rec.mtime_ns,
                self._peak_signals,
                require_spectral=self.spectral_peaks_btn.isChecked(),
            )
        )
        return True

    @Slot(object)
    def _on_table_folder_activated(self, folder: Any) -> None:
        root = self.store.get_root(folder.root_id)
        disp = str(Path(root.path) / Path(folder.rel_path)) if root is not None else folder.rel_path
        self._leave_recent_sort_if_needed()
        self._clear_search_for_directory_browse()
        self._active_sidebar = None
        self.folder_tree.select_location(folder.root_id, folder.id)
        self._set_location(
            TreeLocation(folder.root_id, folder.id, folder.rel_path, disp),
            record_history=True,
        )

    @Slot(int)
    def _on_navigate_to_file_folder(self, file_id: int) -> None:
        rec = self.store.get_file(int(file_id))
        if rec is None:
            return
        root = self.store.get_root(rec.root_id)
        if root is None:
            return
        folder_id = rec.folder_id
        folder_rel = ""
        if folder_id is not None:
            folder = self.store.get_folder(folder_id)
            if folder is not None:
                folder_rel = folder.rel_path
            else:
                folder_id = None
        if not folder_rel:
            rel_parent = Path(rec.rel_path).parent.as_posix()
            if rel_parent and rel_parent != ".":
                folder_rel = rel_parent
        display_path = (
            str(Path(root.path) / Path(folder_rel)) if folder_rel else str(root.path)
        )
        target_loc = TreeLocation(root.id, folder_id, folder_rel, display_path)

        self._search_timer.stop()
        self._history_record_timer.stop()
        self._leave_recent_sort_if_needed()
        self._active_sidebar = None
        if self._dup_mode:
            self._dup_mode = False
            self.dup_banner.setVisible(False)
        self._similar_request = None
        self._ai_search_gen += 1
        self._ai_worker.invalidate(self._ai_search_gen)
        if self._ai_mode:
            self._set_search_mode(False, dispatch=False)
        else:
            self._apply_col_layout("keyword")
            self.ai_banner.setVisible(False)
            self.ai_load_more_btn.setVisible(False)
            self.exclude_same_pkg_btn.setVisible(False)
            self.ai_exit_similar_btn.setVisible(False)

        blocked_search = self.search_input.blockSignals(True)
        try:
            self.search_input.clear()
        finally:
            self.search_input.blockSignals(blocked_search)

        filter_widgets = (
            self.filter_ext_combo,
            self.filter_dur_combo,
            self.filter_fav_cb,
            self.filter_online_cb,
        )
        for w in filter_widgets:
            w.blockSignals(True)
        try:
            self.filter_ext_combo.setCurrentIndex(0)
            self.filter_dur_combo.setCurrentIndex(0)
            self.filter_fav_cb.setChecked(False)
            if not rec.is_available and self.filter_online_cb.isChecked():
                self.filter_online_cb.setChecked(False)
        finally:
            for w in filter_widgets:
                w.blockSignals(False)

        self.folder_tree.select_location(
            target_loc.root_id,
            target_loc.folder_id,
            expand=False,
            prefer_shortcut=True,
            scroll=True,
        )
        self._set_location(target_loc, record_history=True, reveal_file_id=rec.id)
        if rec.is_hidden:
            self.statusBar().showMessage(
                f"已跳转到所在目录（该副本当前已在库中隐藏）：{target_loc.display_path}",
                4500,
            )
        else:
            self.statusBar().showMessage(f"已跳转到所在目录：{target_loc.display_path}", 3500)

    @Slot(int)
    def _on_audition_row(self, row: int) -> None:
        if self.table_model.is_folder_row(row):
            self.audition.stop()
            self.table_model.set_playing_file(None)
            self.table_view.set_playing_progress(None, None)
            return
        rec = self.table_model.record_at(row)
        if rec is None:
            return
        switching = self._current_file is not None and rec.full_path != self._current_file.full_path
        if switching and self.reset_switch_btn.isChecked():
            self._reset_tone()
        ai_hit = self.table_model.ai_hit_for(rec.id)
        if not self._prepare_file_selection(rec, ai_hit=ai_hit):
            return
        start_sec = 0.0
        if (
            ai_hit is not None
            and ai_hit.hit_start_sec is not None
            and ai_hit.hit_end_sec is not None
            and rec.duration_sec > 10.0
        ):
            h0 = float(ai_hit.hit_start_sec)
            h1 = float(ai_hit.hit_end_sec)
            start_sec = max(0.0, min(h0, max(0.0, rec.duration_sec - 0.05)))
            self._pending_seek_sec = start_sec
            self.waveform.set_highlight_range(
                (h0, h1), f"AI 命中 {_fmt_short_clock(h0)}-{_fmt_short_clock(h1)}"
            )
            self.waveform.set_cursor(start_sec)
            self._update_transport_time(start_sec)
        if self.autoplay_btn.isChecked():
            self._play_file(rec, start_sec)
        else:
            self._stop_playback()

    @Slot(int, float)
    def _on_waveform_cell_seek(self, row: int, ratio: float) -> None:
        rec = self.table_model.record_at(row)
        if rec is None:
            return
        had_prev = self._current_file is not None
        switching = self._current_file is None or self._current_file.id != rec.id
        if switching:
            if had_prev and self.reset_switch_btn.isChecked():
                self._reset_tone()
            if not self._prepare_file_selection(rec):
                return
        elif not rec.is_available or not rec.full_path.exists():
            self.waveform.clear("文件离线或不存在")
            self.audition.stop()
            self.table_model.set_playing_file(None)
            self.table_view.set_playing_progress(None, None)
            return

        dur = max(0.0, float(rec.duration_sec))
        seek_sec = max(0.0, min(ratio * dur, max(0.0, dur - 0.005))) if dur > 0.0 else 0.0
        if not switching and self._selection is not None:
            if not (self._selection[0] <= seek_sec < self._selection[1]):
                self._clear_selection()
        end = (
            self._selection[1]
            if self._selection is not None and self._selection[0] <= seek_sec < self._selection[1]
            else None
        )
        self.waveform.set_cursor(seek_sec)
        self.table_view.set_playing_progress(rec.id, ratio)
        self._play_file(rec, seek_sec, end_seconds=end)

    @Slot(int)
    def _on_spot_row(self, row: int) -> None:
        rec = self.table_model.record_at(row)
        if rec is None:
            return
        self._spot_record(rec)

    @Slot(list)
    def _on_spot_files(self, file_ids: list[int]) -> None:
        if not file_ids:
            return
        by_id = self.store.get_files_by_ids([int(fid) for fid in file_ids])
        ordered = [by_id[int(fid)] for fid in file_ids if int(fid) in by_id]
        if ordered:
            self._spot_records(ordered)

    def _spot_record(self, rec: FileRecord) -> None:
        if not rec.is_available or not rec.full_path.exists():
            self.statusBar().showMessage("文件离线或不存在，无法插入 REAPER", 3500)
            return
        st = self._spot_client.status()
        if not st.connected:
            self._refresh_reaper_status()
            self.statusBar().showMessage(
                "REAPER 未连接：可点击底部「REAPER 未连接」一键安装脚本，或直接拖拽音频到工程中",
                5000,
            )
            return

        sel = (
            self._selection
            if (self._current_file is not None and self._current_file.id == rec.id)
            else None
        )
        start_sec = sel[0] if sel is not None else 0.0
        end_sec = sel[1] if sel is not None else None
        result = self._spot_client.spot(
            rec.full_path,
            start_sec=start_sec,
            end_sec=end_sec,
            pitch_semitones=self.pitch_knob.value(),
            rate=self.rate_knob.value(),
            preserve_pitch=self.preserve_pitch_btn.isChecked(),
            apply_tone=self.bake_drag_btn.isChecked(),
            move_cursor=True,
            take_name=rec.filename,
        )
        if result.ok:
            self.statusBar().showMessage(
                f"已插入 REAPER：{result.track_name or '轨道'} @ {result.position:.2f}s（{rec.filename}）",
                4000,
            )
            self._refresh_reaper_status()
        else:
            self.statusBar().showMessage(f"插入 REAPER 失败：{result.message}", 4500)

    def _spot_records(self, records: list[FileRecord]) -> None:
        if not records:
            return
        if len(records) == 1:
            self._spot_record(records[0])
            return
        st = self._spot_client.status()
        if not st.connected:
            self._refresh_reaper_status()
            self.statusBar().showMessage(
                "REAPER 未连接：可点击底部「REAPER 未连接」一键安装脚本，或直接拖拽多条音频到工程中",
                5000,
            )
            return

        pitch = self.pitch_knob.value()
        rate = self.rate_knob.value()
        preserve = self.preserve_pitch_btn.isChecked()
        apply_tone = self.bake_drag_btn.isChecked()

        inserted = 0
        skipped_offline = 0
        first_pos: float | None = None
        last_track = ""
        failed_msg = ""

        for rec in records:
            if not rec.is_available or not rec.full_path.exists():
                skipped_offline += 1
                continue
            result = self._spot_client.spot(
                rec.full_path,
                start_sec=0.0,
                end_sec=None,
                pitch_semitones=pitch,
                rate=rate,
                preserve_pitch=preserve,
                apply_tone=apply_tone,
                move_cursor=True,
                take_name=rec.filename,
            )
            if not result.ok:
                failed_msg = result.message
                break
            if first_pos is None:
                first_pos = result.position
            last_track = result.track_name or last_track
            inserted += 1

        self._refresh_reaper_status()
        if inserted > 0:
            pos_str = f"起始 {first_pos:.2f}s" if first_pos is not None else ""
            extra: list[str] = []
            if pos_str:
                extra.append(pos_str)
            if last_track:
                extra.append(f"轨道 {last_track}")
            if skipped_offline:
                extra.append(f"跳过离线 {skipped_offline} 条")
            if failed_msg:
                extra.append(f"中断：{failed_msg}")
            detail = " · ".join(extra)
            self.statusBar().showMessage(
                f"已连续插入 REAPER：{inserted} 条音频（{detail}）" if detail else f"已连续插入 REAPER：{inserted} 条音频",
                4500,
            )
        elif failed_msg:
            self.statusBar().showMessage(f"连续插入 REAPER 失败：{failed_msg}", 4500)
        else:
            self.statusBar().showMessage("所选文件均离线或不存在，无法插入 REAPER", 3500)

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        st = widget.style()
        if st is not None:
            st.unpolish(widget)
            st.polish(widget)
        widget.update()

    @Slot(str)
    def apply_ui_scale(self, scale_key: str, *, persist: bool = True) -> None:
        resolved = normalize_ui_scale(scale_key)
        self._ui_scale = resolved
        spec = ui_scale_spec(resolved)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, resolved)

        mode_btn_h = max(24, spec.input_h - 6)
        for btn in (self.back_btn, self.forward_btn, self.up_btn):
            btn.setFixedHeight(spec.btn_h)
        self.mode_track.setFixedHeight(spec.input_h)
        self.mode_keyword_btn.setFixedHeight(mode_btn_h)
        self.mode_ai_btn.setFixedHeight(mode_btn_h)
        self.search_input.setFixedHeight(spec.input_h)
        self.ai_search_btn.setFixedHeight(spec.input_h)
        self.play_btn.setFixedSize(int(round(36 * spec.scale)), spec.btn_h)
        self.stop_btn.setFixedSize(int(round(30 * spec.scale)), spec.btn_h)

        self.count_label.setFont(mono_font(spec.mono_pt))
        self.transport_time_label.setFont(mono_font(spec.mono_pt))

        self.folder_tree.apply_ui_scale(resolved)
        self.table_view.apply_ui_scale(resolved)
        self.inspector.apply_ui_scale(resolved)
        self.volume_knob.apply_ui_scale(resolved)
        self.pitch_knob.apply_ui_scale(resolved)
        self.rate_knob.apply_ui_scale(resolved)

        self._repolish(self.status_audio)
        self._repolish(self.status_mode)
        if persist:
            self._schedule_save_settings()

    def set_output_device(
        self,
        device_name: str,
        *,
        refresh_portaudio: bool = False,
        persist: bool = True,
    ) -> tuple[bool, str]:
        requested = str(device_name or "").strip()
        self._output_device_name = requested
        matched = True
        active_name = ""
        setter = getattr(self.audition, "set_output_device", None)
        if callable(setter):
            matched, active_name = setter(requested, refresh_portaudio=refresh_portaudio)
        self._update_device_status()
        if persist:
            self._schedule_save_settings()
        return matched, active_name

    def set_index_video_files(self, enabled: bool, *, persist: bool = True) -> None:
        self._index_video_files = bool(enabled)
        if persist:
            self._schedule_save_settings()

    @Slot()
    def _open_settings_dialog(self, initial_tab: str = "appearance") -> None:
        dlg = SettingsDialog(self, initial_tab=initial_tab, parent=self)
        dlg.exec()

    @Slot()
    def _schedule_silent_update_check(self) -> None:
        self._pool.start(_SilentUpdateTask(self._update_signals))

    @Slot(object)
    def _on_silent_update_checked(self, result: UpdateCheckResult) -> None:
        self._latest_update_result = result
        if result.status in ("patch_available", "full_update_required") and result.manifest is not None:
            self.status_version.setText(f"↑ v{result.manifest.version}")
            self.status_version.setProperty("statusState", "ok")
            self._repolish(self.status_version)
            self.status_version.setToolTip(f"{result.message}\n点击打开更新窗口")

    @Slot()
    def _open_update_dialog(self) -> None:
        dlg = UpdateDialog(initial_result=self._latest_update_result, parent=self)
        dlg.restartRequested.connect(self._restart_application)
        dlg.cudaUpgradeRestartRequested.connect(self._apply_cuda_upgrade_and_restart)
        dlg.exec()

    @Slot(str)
    def _apply_cuda_upgrade_and_restart(self, script_path: str) -> None:
        from audiosearch.browser.cuda_updater import launch_apply_cuda_script

        self._flush_saved_settings()
        if self._ai_indexer is not None:
            self._ai_indexer.cancel()
        try:
            self.audition.stop()
        except Exception:
            pass
        launch_apply_cuda_script(Path(script_path))
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    @Slot()
    def _restart_application(self) -> None:
        self._flush_saved_settings()
        if is_frozen():
            cmd = [sys.executable, *sys.argv[1:]]
        else:
            cmd = [sys.executable, "-m", "audiosearch.browser", *sys.argv[1:]]
        subprocess.Popen(cmd)  # noqa: S603
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    @Slot()
    def _open_reaper_setup_dialog(self) -> None:
        dlg = ReaperSetupDialog(
            self._spot_client,
            on_import_shortcuts=lambda reaper_dir: self._on_import_reaper_shortcuts(
                reaper_dir=reaper_dir
            ),
            parent=self,
        )
        dlg.exec()
        self._refresh_reaper_status()

    def _refresh_reaper_status(self) -> None:
        st = self._spot_client.status()
        if st.connected:
            track = st.track_name or "(新建轨道)"
            self.status_mode.setText(f"REAPER 已连接 · {track}")
            self.status_mode.setProperty("statusState", "ok")
            self._repolish(self.status_mode)
            self.status_mode.setToolTip(
                f"工程：{st.project_name or '未命名'}\n"
                f"目标轨：{track}\n"
                f"光标：{st.cursor_sec:.2f}s\n"
                "点击查看 REAPER 桥接状态或重新安装脚本"
            )
        else:
            self.status_mode.setText("REAPER 未连接")
            self.status_mode.setProperty("statusState", "idle")
            self._repolish(self.status_mode)
            self.status_mode.setToolTip(
                "双击 / Enter 无破坏插入需要 REAPER 启动并运行 audiosearch_spot_bridge.lua\n"
                "点击此处可一键安装 / 配置 REAPER 桥接脚本（平时也可直接拖拽音频导入）"
            )

    @Slot()
    def _on_thumbnails_ready(self) -> None:
        ids = self.thumbnail_manager.drain_ready_ids()
        if ids:
            self.table_view.repaint_file_waveforms(ids)

    @Slot(float)
    def _on_waveform_seek(self, seconds: float) -> None:
        rec = self._current_file
        if rec is None or not rec.is_available or not rec.full_path.exists():
            return
        end = (
            self._selection[1]
            if self._selection is not None and self._selection[0] <= seconds < self._selection[1]
            else None
        )
        self._play_file(rec, seconds, end_seconds=end)

    @Slot(object)
    def _on_selection_changed(self, selection: tuple[float, float] | None) -> None:
        self._selection = selection
        self.audition.set_selection(selection)
        self._update_transport_time(self.audition.position_seconds)

    @Slot(float, float)
    def _on_selection_committed(self, start: float, end: float) -> None:
        rec = self._current_file
        if rec is None or not rec.is_available or not rec.full_path.exists():
            return
        self._selection = (start, end)
        self.audition.set_selection(self._selection)
        self._update_transport_time(start)
        if self.autoplay_btn.isChecked() or self.audition.state == "playing":
            self._play_file(rec, start, end_seconds=end)

    def _clear_selection(self) -> None:
        self._selection = None
        self.waveform.set_selection(None)
        self.audition.set_selection(None)
        self._update_transport_time(self.audition.position_seconds)

    @Slot(float, float, bool)
    def _on_selection_drag(self, start: float, end: float, fade_5ms: bool) -> None:
        rec = self._current_file
        if rec is None or not rec.is_available or not rec.full_path.exists():
            return
        bake = self._should_bake_drag()
        try:
            rendered = render_range(
                rec.full_path,
                start_seconds=start,
                end_seconds=end,
                out_dir=self._render_dir,
                fade_ms=5.0 if fade_5ms else 0.0,
                pitch_semitones=self.pitch_knob.value() if bake else 0.0,
                rate=self.rate_knob.value() if bake else 1.0,
                preserve_pitch=self.preserve_pitch_btn.isChecked() if bake else False,
            )
            prune_render_dir(self._render_dir, max_files=50)
        except Exception as exc:
            self.status_audio.setText(f"选区渲染失败: {exc}")
            return
        start_file_drag(self.waveform, rendered)

    @Slot(bool)
    def _on_spectral_peaks_toggled(self, enabled: bool) -> None:
        self.waveform.set_spectral_peaks(enabled)
        self.table_view.set_spectral_peaks(enabled)
        rec = self._current_file
        if (
            not enabled
            or rec is None
            or not rec.is_available
            or not rec.full_path.exists()
            or not self.waveform.needs_spectral_data()
        ):
            return
        overview = self.waveform._overview_peaks
        if overview is not None and overview.spectral is None:
            self._peak_gen += 1
            self._pool.start(
                _PeakTask(
                    self.store,
                    self._peak_gen,
                    rec.id,
                    rec.full_path,
                    rec.mtime_ns,
                    self._peak_signals,
                    require_spectral=True,
                )
            )
            return
        v0, v1 = self.waveform.view
        if self.waveform.duration > 0 and (v1 - v0) < self.waveform.duration * 0.25:
            self._on_zoom_range_requested(v0, v1)

    @Slot(float, float)
    def _on_zoom_range_requested(self, start_sec: float, end_sec: float) -> None:
        self._pending_zoom = (start_sec, end_sec)
        self._zoom_timer.start()

    @Slot()
    def _load_zoom_peaks(self) -> None:
        rec = self._current_file
        if rec is None or self._pending_zoom is None or not rec.is_available:
            return
        s0, s1 = self._pending_zoom
        self._peak_gen += 1
        self._pool.start(
            _PeakTask(
                self.store,
                self._peak_gen,
                rec.id,
                rec.full_path,
                rec.mtime_ns,
                self._peak_signals,
                start_seconds=s0,
                end_seconds=s1,
                require_spectral=self.spectral_peaks_btn.isChecked(),
            )
        )

    @Slot(int, int, object)
    def _on_peaks_loaded(self, generation: int, file_id: int, peak_data: PeakData) -> None:
        if generation != self._peak_gen:
            return
        if self._current_file is None or self._current_file.id != file_id:
            return
        self.waveform.set_peaks(peak_data)
        is_full = peak_data.start_seconds <= 1e-4 and (
            peak_data.duration_seconds <= 0
            or abs(peak_data.end_seconds - peak_data.duration_seconds) <= 0.05
        )
        if is_full:
            thumb = peaks_to_thumbnail(peak_data)
            spec_thumb = peaks_to_thumbnail_spectral(peak_data) or None
            if thumb:
                self.thumbnail_manager.inject(
                    file_id, self._current_file.mtime_ns, thumb, spectral_blob=spec_thumb
                )
            if self.waveform.duration > 0:
                v0, v1 = self.waveform.view
                if (v1 - v0) < self.waveform.duration * 0.25:
                    self._on_zoom_range_requested(v0, v1)

    @Slot(int, int, str)
    def _on_peaks_failed(self, generation: int, file_id: int, message: str) -> None:
        if generation != self._peak_gen:
            return
        if self._current_file is None or self._current_file.id != file_id:
            return
        self.waveform.set_peaks(None, message=f"波形不可用：{message}")

    @Slot(int, float)
    def _on_audition_position(self, file_id: int, seconds: float) -> None:
        if self._current_file is None or self._current_file.id != file_id:
            return
        dur = self._current_file.duration_sec
        ratio = max(0.0, min(1.0, seconds / dur)) if dur > 0.0 else 0.0
        if self.audition.state == "playing":
            self.waveform.set_playback_head(seconds)
            self.table_view.set_playing_progress(file_id, ratio)
            self._update_transport_time(seconds)
            if is_video_path(self._current_file.full_path):
                self.inspector.sync_video_time(seconds)
        elif self.audition.state == "finished":
            self.waveform.set_playback_head(None)
            self.table_view.set_playing_progress(None, None)
            self._update_transport_time(0.0)
            if is_video_path(self._current_file.full_path):
                self.inspector.set_video_playing(False)
                self.inspector.sync_video_time(
                    self._selection[0] if self._selection is not None else 0.0
                )
        elif self.audition.state == "paused":
            if seconds > 1e-4:
                self.table_view.set_playing_progress(file_id, ratio)
            else:
                self.table_view.set_playing_progress(None, None)
            self._update_transport_time(seconds)
            if is_video_path(self._current_file.full_path):
                self.inspector.set_video_playing(False)
                self.inspector.sync_video_time(seconds)

    def _update_play_btn_visual(self, playing: bool) -> None:
        is_playing = bool(playing)
        self.play_btn.setIcon(self._icon_pause if is_playing else self._icon_play)
        if bool(self.play_btn.property("playing")) != is_playing:
            self.play_btn.setProperty("playing", is_playing)
            st = self.play_btn.style()
            if st is not None:
                st.unpolish(self.play_btn)
                st.polish(self.play_btn)
            self.play_btn.update()

    @Slot(str)
    def _on_audition_state(self, state: str) -> None:
        if state == "playing" and self._current_file is not None:
            self.table_model.set_playing_file(self._current_file.id)
            self._update_play_btn_visual(True)
            if is_video_path(self._current_file.full_path):
                self.inspector.set_video_playing(True)
        elif state in ("finished", "empty"):
            self.waveform.set_playback_head(None)
            self.table_model.set_playing_file(None)
            self.table_view.set_playing_progress(None, None)
            self._update_play_btn_visual(False)
            self.inspector.set_video_playing(False)
        elif state == "paused":
            self.table_model.set_playing_file(None)
            self._update_play_btn_visual(False)
            self.inspector.set_video_playing(False)
        self._update_device_status()

    @Slot(str)
    def _on_audition_failed(self, message: str) -> None:
        self.status_audio.setText(f"声卡异常: {message}")

    def _update_transport_time(self, pos_sec: float) -> None:
        dur = self._current_file.duration_sec if self._current_file is not None else 0.0
        text = f"{_fmt_clock(pos_sec)} / {_fmt_clock(dur)}"
        if self._selection is not None:
            s0, s1 = self._selection
            text += f"  [{_fmt_clock(s0)}-{_fmt_clock(s1)}]"
        self.transport_time_label.setText(text)

    def _update_device_status(self) -> None:
        err = self.audition.device_error
        rate = int(getattr(self.audition, "output_rate_hz", 48000) or 48000)
        rate_str = f"{rate // 1000}kHz" if rate % 1000 == 0 else f"{rate / 1000:.1f}kHz"
        active_name = str(getattr(self.audition, "active_device_name", "") or "").strip()
        fallback = bool(getattr(self.audition, "device_fallback", False))
        if err:
            self.status_audio.setText(f"声卡异常: {err}")
            self.status_audio.setProperty("statusState", "error")
            self._repolish(self.status_audio)
            self.status_audio.setToolTip(f"{err}\n点击打开「试听与音频」设置切换输出设备")
            return

        if self._output_device_name and active_name and not fallback:
            short = active_name if len(active_name) <= 18 else f"{active_name[:16]}…"
            self.status_audio.setText(f"声卡就绪 · {short} ({rate_str})")
        else:
            self.status_audio.setText(f"声卡就绪 ({rate_str})")
        self.status_audio.setProperty("statusState", "idle")
        self._repolish(self.status_audio)
        dev_line = active_name or "系统默认输出设备"
        fb_line = f"\n注意：原选设备「{self._output_device_name}」未连接，已自动回退" if fallback else ""
        self.status_audio.setToolTip(
            f"输出设备：{dev_line}\n采样率：{rate_str}（WASAPI 共享模式）{fb_line}\n点击切换音频输出设备"
        )

    # -- Roots & Scanning --------------------------------------------------
    @Slot()
    def _choose_and_add_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择音效库文件夹")
        if not chosen:
            return
        self.add_and_scan_root(Path(chosen))

    def add_and_scan_root(self, path: Path) -> None:
        root = self.store.add_root(path)
        self.folder_tree.set_local_roots_expanded(True)
        self.folder_tree.reload(keep_location=self._location)
        self._schedule_save_settings()
        self.start_scan_root(root.id)

    @Slot(int)
    def start_scan_root(self, root_id: int) -> None:
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self.scan_label.setText("已有扫描在进行中")
            return
        scanner = LibraryScanner(self.store.db_path, include_videos=self._index_video_files)
        self._scanner = scanner
        self._scan_started_at = time.monotonic()
        self.scan_pause_btn.setText("暂停扫描")
        self.scan_pause_btn.setVisible(True)
        self.scan_label.setText("扫描启动中…")

        def run_scan() -> None:
            try:
                report = scanner.scan_root(
                    root_id,
                    on_progress=lambda p: self._scan_signals.progress.emit(p),
                )
                self._scan_signals.done.emit(report)
            except Exception as exc:
                self._scan_signals.failed.emit(str(exc))

        self._scan_thread = threading.Thread(target=run_scan, daemon=True)
        self._scan_thread.start()

    @Slot()
    def _toggle_scan_pause(self) -> None:
        if self._scanner is None:
            return
        if self._scanner.is_paused:
            self._scanner.resume()
            self.scan_pause_btn.setText("暂停扫描")
        else:
            self._scanner.pause()
            self.scan_pause_btn.setText("继续扫描")

    @Slot(int)
    def _remove_root(self, root_id: int) -> None:
        removed_ids = self.store.file_ids_for_root(root_id)
        self.store.remove_root(root_id)
        if removed_ids:
            idx = self._ensure_ai_index()
            if idx is not None and hasattr(idx, "deactivate_file_ids"):
                idx.deactivate_file_ids(removed_ids)
        if self._location.root_id == root_id:
            self._location = TreeLocation(None, None, "", "全部素材")
            self.path_edit.setText("全部素材")
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_ai_coverage()
        if self._ai_mode and self._similar_request is None and not self.search_input.text().strip():
            self._update_ai_banner_idle()
        self._refresh_active_view()

    @Slot(object)
    def _on_scan_progress(self, p: ScanProgress) -> None:
        state_zh = {
            "scanning": "扫描中",
            "paused": "已暂停",
            "offline": "离线",
            "finalizing": "汇总目录中",
        }.get(p.status, p.status)
        elapsed = time.monotonic() - self._scan_started_at if self._scan_started_at > 0.0 else 0.0
        rate_str = (
            f" · {int(p.scanned / elapsed):,}/s"
            if p.status == "scanning" and elapsed >= 1.0 and p.scanned > 0
            else ""
        )
        err_str = f" · 异常 {p.errors:,}" if p.errors > 0 else ""
        self.scan_label.setText(
            f"{state_zh}：已扫 {p.scanned:,} · 新增 {p.added:,} · 跳过 {p.skipped:,}{rate_str}{err_str}"
        )

    @Slot(object)
    def _on_scan_done(self, report: ScanReport) -> None:
        self.scan_pause_btn.setVisible(False)
        err_str = f" · 异常 {report.errors:,}" if report.errors > 0 else ""
        self.scan_label.setText(
            f"扫描完成：新增 {report.added:,} · 更新 {report.updated:,} · 跳过 {report.skipped:,}{err_str}"
        )
        self.store.bump_index_version()
        self.table_model.invalidate_id_cache()
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        if report.removed > 0:
            self._schedule_ai_orphan_prune()
        self._refresh_ai_coverage()
        self._refresh_active_view()

    @Slot(str)
    def _on_scan_failed(self, message: str) -> None:
        self.scan_pause_btn.setVisible(False)
        self.scan_label.setText(f"扫描失败：{message}")

    # -- AI Semantic Search, Find Similar & Background Indexing -------------
    def _ensure_ai_index(self) -> Any | None:
        if self._ai_index is not None:
            return self._ai_index
        try:
            from audiosearch.ai.vector_store import LibraryVectorIndex, vector_dir_for_db

            vec_dir = vector_dir_for_db(self.store.db_path, self._settings.ai_profile)
            self._ai_index = LibraryVectorIndex(vec_dir, profile=self._settings.ai_profile)
            return self._ai_index
        except Exception:
            return None

    def _ensure_ai_engine(self) -> Any | None:
        if self._ai_engine is not None:
            return self._ai_engine
        idx = self._ensure_ai_index()
        if idx is None:
            return None
        from audiosearch.ai.search_engine import AiSearchEngine

        self._ai_engine = AiSearchEngine(self.store, idx)
        return self._ai_engine

    def _refresh_ai_coverage(self) -> None:
        idx = self._ensure_ai_index()
        if idx is None:
            self.status_ai.setText("AI 索引: 依赖未安装")
            self.status_ai.setToolTip("")
            self.ai_index_btn.setEnabled(False)
            return
        self.ai_index_btn.setEnabled(True)
        total = self.store.total_file_count()
        cov = idx.coverage(total)
        audio_windows = int(getattr(cov, "audio_windows", 0))
        if cov.text_files == 0 and cov.audio_files == 0:
            self.status_ai.setText("AI 索引: 未建库")
            self.status_ai.setToolTip(f"当前素材库共 {total:,} 条，尚未构建 AI 向量索引。")
        else:
            self.status_ai.setText(
                f"AI 索引: {cov.overall_percent:.0f}% (文本 {cov.text_files:,} · 声波 {cov.audio_files:,})"
            )
            tip_lines = [
                f"当前素材库共 {total:,} 条",
                f"文本索引：{cov.text_files:,} 条",
                f"声波索引：{cov.audio_files:,} 条（{audio_windows:,} 个声波窗口）",
            ]
            if cov.text_files > cov.audio_files:
                diff = cov.text_files - cov.audio_files
                tip_lines.append(
                    f"差值 {diff:,} 条：通常为 0 秒空文件或无法解码的损坏音频，跳过声波建库"
                )
            self.status_ai.setToolTip("\n".join(tip_lines))
            if (cov.text_files > total or cov.audio_files > total) and not self._ai_auto_pruned:
                self._ai_auto_pruned = True
                self._schedule_ai_orphan_prune()

    @Slot()
    def _on_ai_coverage_reconciled(self) -> None:
        self._ai_prune_running = False
        self._refresh_ai_coverage()
        if self._ai_mode and self._similar_request is None and not self.search_input.text().strip():
            self._update_ai_banner_idle()

    def _schedule_ai_orphan_prune(self) -> None:
        if self._ai_prune_running:
            return
        idx = self._ensure_ai_index()
        if idx is None or not hasattr(idx, "prune_missing_file_ids"):
            return
        self._ai_prune_running = True
        db_path = self.store.db_path

        def _prune() -> None:
            worker_store: LibraryStore | None = None
            try:
                worker_store = LibraryStore(db_path)
                valid_ids = worker_store.all_file_ids()
                idx.prune_missing_file_ids(valid_ids)
            except Exception:
                pass
            finally:
                if worker_store is not None:
                    try:
                        worker_store.close()
                    except Exception:
                        pass
                try:
                    self._ai_index_signals.coverageChanged.emit()
                except RuntimeError:
                    pass

        threading.Thread(target=_prune, daemon=True, name="ai-orphan-prune").start()

    def _current_ai_limit(self) -> int:
        if hasattr(self, "ai_limit_combo"):
            val = self.ai_limit_combo.currentData()
            if val is not None:
                try:
                    return max(50, min(5000, int(val)))
                except (TypeError, ValueError):
                    pass
        return max(50, min(5000, int(self._active_ai_limit)))

    def _set_ai_limit_combo(self, limit: int) -> None:
        if not hasattr(self, "ai_limit_combo"):
            return
        target = max(50, min(5000, int(limit)))
        self._active_ai_limit = target
        blocked = self.ai_limit_combo.blockSignals(True)
        try:
            best_idx = 1
            best_diff = 10**9
            for i in range(self.ai_limit_combo.count()):
                item_val = int(self.ai_limit_combo.itemData(i) or 500)
                diff = abs(item_val - target)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            self.ai_limit_combo.setCurrentIndex(best_idx)
        finally:
            self.ai_limit_combo.blockSignals(blocked)

    @Slot()
    def _on_ai_limit_changed(self) -> None:
        self._active_ai_limit = self._current_ai_limit()
        self._schedule_save_settings()
        if not self._settings_ready or self._dup_mode:
            return
        if self._similar_request is not None:
            file_id, seed_range = self._similar_request
            self._start_find_similar(file_id, seed_range)
        elif self._ai_mode and self.table_model.is_ai_mode and self.search_input.text().strip():
            self._dispatch_ai_search()

    @Slot()
    def _on_ai_load_more(self) -> None:
        cur = self._current_ai_limit()
        steps = (200, 500, 1000, 2000, 5000)
        next_limit = next((s for s in steps if s > cur), 5000)
        if next_limit == cur and cur >= 5000:
            return
        self._set_ai_limit_combo(next_limit)
        self._schedule_save_settings()
        if self._similar_request is not None:
            file_id, seed_range = self._similar_request
            self._start_find_similar(file_id, seed_range)
        elif self._ai_mode and self.search_input.text().strip():
            self._dispatch_ai_search()

    def _update_ai_banner_idle(self) -> None:
        self.ai_load_more_btn.setVisible(False)
        idx = self._ensure_ai_index()
        if idx is None:
            self.ai_banner_label.setText("当前环境未启用 AI 检索依赖（ai extra）")
            return
        cov = idx.coverage(self.store.total_file_count())
        if cov.text_files == 0 and cov.audio_files == 0:
            self.ai_banner_label.setText(
                "AI 语义就绪：请先点击右下角「AI 建库」构建本地向量库，之后可按中文自然语言搜氛围/质感"
            )
        else:
            self.ai_banner_label.setText(
                f"AI 语义就绪：已索引文本 {cov.text_files:,} 条 · 声波 {cov.audio_files:,} 条（输入描述按 Enter 检索）"
            )

    def _is_global_search_active(self, *, for_search: bool = False) -> bool:
        if not self.scope_btn.isChecked():
            return False
        if for_search or self._similar_request is not None:
            return True
        return (not self._ai_mode) and bool(self.search_input.text().strip())

    def _current_scope_args(
        self,
        *,
        for_search: bool = False,
        global_override: bool | None = None,
    ) -> dict[str, Any]:
        global_mode = (
            self._is_global_search_active(for_search=for_search)
            if global_override is None
            else bool(global_override)
        )
        if global_mode:
            return {"root_id": None, "folder_id": None, "folder_rel_path": ""}
        if self._active_sidebar is not None:
            if self._active_sidebar.kind == KIND_COLLECTION_FOLDER:
                return {
                    "root_id": self._active_sidebar.root_id,
                    "folder_id": self._active_sidebar.folder_id,
                    "folder_rel_path": self._active_sidebar.folder_rel_path,
                }
            if self._active_sidebar.kind in (KIND_FAVORITES, KIND_RECENT, KIND_COLLECTION):
                return {"root_id": None, "folder_id": None, "folder_rel_path": ""}
        return {
            "root_id": self._location.root_id,
            "folder_id": self._location.folder_id,
            "folder_rel_path": self._location.folder_rel_path,
        }

    @Slot()
    def _dispatch_ai_search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            self._update_ai_banner_idle()
            return
        engine = self._ensure_ai_engine()
        if engine is None:
            self.ai_banner.setVisible(True)
            self.ai_banner_label.setText("AI 检索引擎不可用，请确认已安装 ai 依赖组")
            return
        self._query_gen += 1
        self._ai_search_gen += 1
        self.ai_banner.setVisible(True)
        self.ai_load_more_btn.setVisible(False)
        self.ai_banner_label.setText("正在执行 AI 语义 + 声波融合检索…")
        is_global = self._is_global_search_active(for_search=True)
        payload = {
            "query": query,
            "limit": self._current_ai_limit(),
            **self._current_scope_args(global_override=is_global),
            "filters": self._current_filters(ignore_sidebar_scope=is_global),
        }
        self._ai_worker.submit(engine, self._ai_search_gen, "semantic", payload)

    @Slot(int)
    def _on_find_similar_row(self, row: int) -> None:
        rec = self.table_model.record_at(row)
        if rec is None:
            return
        self._start_find_similar(rec.id, None)

    @Slot(object)
    def _on_find_similar_waveform(self, range_sec: object) -> None:
        rec = self._current_file
        if rec is None:
            return
        parsed = (
            (float(range_sec[0]), float(range_sec[1]))
            if isinstance(range_sec, tuple) and len(range_sec) == 2
            else None
        )
        self._start_find_similar(rec.id, parsed)

    def _start_find_similar(
        self, file_id: int, seed_range: tuple[float, float] | None
    ) -> None:
        engine = self._ensure_ai_engine()
        if engine is None:
            self.statusBar().showMessage("AI 找同类不可用，请确认已安装 ai 依赖组", 3500)
            return
        self._similar_request = (file_id, seed_range)
        if self._dup_mode:
            self._dup_mode = False
            self.dup_banner.setVisible(False)
        self._apply_col_layout("ai")
        self.ai_banner.setVisible(True)
        self.ai_load_more_btn.setVisible(False)
        self.exclude_same_pkg_btn.setVisible(True)
        self.ai_exit_similar_btn.setVisible(True)
        self.ai_banner_label.setText("正在检索音色相似素材…")
        self._query_gen += 1
        self._ai_search_gen += 1
        is_global = self._is_global_search_active(for_search=True)
        payload: dict[str, Any] = {
            "file_id": file_id,
            "limit": self._current_ai_limit(),
            "exclude_same_folder_or_package": self.exclude_same_pkg_btn.isChecked(),
            **self._current_scope_args(global_override=is_global),
            "filters": self._current_filters(ignore_sidebar_scope=is_global),
        }
        if seed_range is not None:
            kind = "similar_range"
            payload["start_sec"] = seed_range[0]
            payload["end_sec"] = seed_range[1]
        else:
            kind = "similar_file"
        self._ai_worker.submit(engine, self._ai_search_gen, kind, payload)

    @Slot(bool)
    def _on_exclude_same_pkg_toggled(self, _checked: bool) -> None:
        if self._similar_request is not None:
            file_id, seed_range = self._similar_request
            self._start_find_similar(file_id, seed_range)

    @Slot()
    def _exit_similar_mode(self) -> None:
        self._similar_request = None
        self.exclude_same_pkg_btn.setVisible(False)
        self.ai_exit_similar_btn.setVisible(False)
        if self._ai_mode:
            if self.search_input.text().strip():
                self._dispatch_ai_search()
            else:
                self._update_ai_banner_idle()
                self._dispatch_query()
        else:
            self._apply_col_layout("keyword")
            self.ai_banner.setVisible(False)
            self.ai_load_more_btn.setVisible(False)
            self._dispatch_query()

    @Slot(int, object)
    def _on_ai_search_finished(self, generation: int, resp: Any) -> None:
        if generation != self._ai_search_gen:
            return
        hits_map = {h.file_id: h for h in resp.hits}
        self.table_model.apply_ai_results(
            list(resp.records), hits_map, sort_column="relevance", sort_asc=False
        )
        self._sync_sort_indicator("relevance", False)

        count = len(resp.records)
        if resp.mode != "similar" and self.search_input.text().strip():
            self.store.record_search_history(
                self.search_input.text().strip(), mode="ai", result_count=count
            )
        has_more = bool(getattr(resp, "has_more", False)) and count < 5000
        self.ai_load_more_btn.setVisible(has_more)
        self.count_label.setText(f"{count:,}+ 条" if has_more else f"{count:,} 条")
        self.status_view.setText(f"AI 结果: {count:,} 条 ({resp.elapsed_ms:.0f}ms)")
        prefix = "音色找同类" if resp.mode == "similar" else "AI 语义"
        if has_more:
            summary = f"{prefix}显示前 {count:,} 条（有更多命中 · {resp.elapsed_ms:.0f}ms）"
        else:
            summary = f"{prefix}命中 {count:,} 条 ({resp.elapsed_ms:.0f}ms)"
        if resp.note:
            summary += f" · {resp.note}"
        self.ai_banner_label.setText(summary)

    # -- R4: Favorites, Notes, Filters, History, Collections, Duplicates & Drives
    def _leave_recent_sort_if_needed(self) -> None:
        if self._saved_sort_before_recent is not None:
            self._sort_column, self._sort_asc = self._saved_sort_before_recent
            self._saved_sort_before_recent = None

    def _active_collection_id(self) -> int | None:
        if self._active_sidebar is not None and self._active_sidebar.kind in (
            KIND_COLLECTION,
            KIND_COLLECTION_FOLDER,
        ):
            return self._active_sidebar.collection_id
        return None

    @Slot(bool)
    def _on_filter_bar_toggled(self, visible: bool) -> None:
        self.center_pane.setUpdatesEnabled(False)
        try:
            self.filter_bar.setVisible(bool(visible))
            lay = self.center_pane.layout()
            if lay is not None:
                lay.activate()
        finally:
            self.center_pane.setUpdatesEnabled(True)
        self.center_pane.update()

    @Slot()
    def _on_filter_changed(self) -> None:
        self._refresh_active_view()

    @Slot(bool)
    def _on_hide_offline_toggled(self, checked: bool) -> None:
        if checked:
            if self._location.root_id is not None:
                changed = self.store.refresh_folder_online(
                    self._location.root_id, self._location.folder_rel_path
                )
            else:
                changed = self.store.refresh_roots_online()
            if changed:
                self.folder_tree.reload(
                    keep_location=self._location, keep_target=self._active_sidebar
                )
        self._refresh_active_view()

    @Slot()
    def _reset_filters(self) -> None:
        for w in (
            self.filter_ext_combo,
            self.filter_dur_combo,
            self.filter_fav_cb,
            self.filter_online_cb,
        ):
            w.blockSignals(True)
        try:
            self.filter_ext_combo.setCurrentIndex(0)
            self.filter_dur_combo.setCurrentIndex(0)
            self.filter_fav_cb.setChecked(False)
            self.filter_online_cb.setChecked(False)
        finally:
            for w in (
                self.filter_ext_combo,
                self.filter_dur_combo,
                self.filter_fav_cb,
                self.filter_online_cb,
            ):
                w.blockSignals(False)
        self._refresh_active_view()

    @Slot()
    def _record_keyword_history(self) -> None:
        if self._ai_mode or self._dup_mode:
            return
        q = self.search_input.text().strip()
        if not q:
            return
        self.store.record_search_history(q, mode="keyword", result_count=self.table_model.rowCount())

    @Slot()
    def _show_history_menu(self) -> None:
        menu = QMenu(self)
        entries = self.store.list_search_history(limit=25)
        if not entries:
            empty = QAction("暂无搜索历史", menu)
            empty.setEnabled(False)
            menu.addAction(empty)
        else:
            for entry in entries:
                pin_tag = "📌 " if entry.pinned else ""
                mode_tag = "[AI]" if entry.mode == "ai" else "[词]"
                label = f"{pin_tag}{mode_tag} {entry.query}  ({entry.result_count:,} 条)"
                sub = menu.addMenu(label)
                use_act = QAction("立即搜索", sub)
                use_act.triggered.connect(lambda _c=False, e=entry: self._apply_history_entry(e))
                sub.addAction(use_act)

                pin_act = QAction("取消置顶" if entry.pinned else "📌 置顶常用", sub)
                pin_act.triggered.connect(
                    lambda _c=False, eid=entry.id, p=not entry.pinned: self.store.set_search_history_pinned(
                        eid, p
                    )
                )
                sub.addAction(pin_act)

                del_act = QAction("删除此条", sub)
                del_act.triggered.connect(
                    lambda _c=False, eid=entry.id: self.store.delete_search_history(eid)
                )
                sub.addAction(del_act)

            menu.addSeparator()
            clear_act = QAction("清空非置顶历史", menu)
            clear_act.triggered.connect(lambda _c=False: self.store.clear_search_history(keep_pinned=True))
            menu.addAction(clear_act)
        menu.exec(self.history_btn.mapToGlobal(self.history_btn.rect().bottomLeft()))

    def _apply_history_entry(self, entry: Any) -> None:
        self._similar_request = None
        if entry.mode == "ai" and not self._ai_mode:
            self._set_search_mode(True)
        elif entry.mode == "keyword" and self._ai_mode:
            self._set_search_mode(False)
        self.search_input.setText(entry.query)
        if entry.mode == "ai":
            self._dispatch_ai_search()
        else:
            self._search_timer.stop()
            self._dispatch_query()

    @Slot(int)
    def _on_toggle_favorite(self, file_id: int) -> None:
        fav = self.store.toggle_favorite(file_id)
        self.table_model.refresh_file_meta([file_id])
        if self._current_file is not None and self._current_file.id == file_id:
            fresh = self.store.get_file(file_id)
            if fresh is not None:
                self._current_file = fresh
            self.inspector.set_favorite_state(fav)
        self.folder_tree.refresh_quick_counts()

    @Slot(list, bool)
    def _on_batch_favorite(self, file_ids: list[int], favorite: bool) -> None:
        if not file_ids:
            return
        self.store.set_favorites(file_ids, favorite)
        self.table_model.refresh_file_meta(file_ids)
        if self._current_file is not None and self._current_file.id in file_ids:
            fresh = self.store.get_file(self._current_file.id)
            if fresh is not None:
                self._current_file = fresh
            self.inspector.set_favorite_state(favorite)
        self.folder_tree.refresh_quick_counts()
        self.statusBar().showMessage(
            f"已{'加入收藏' if favorite else '取消收藏'} {len(file_ids)} 条素材", 2500
        )

    @Slot(list, bool)
    def _on_set_played_marks(self, file_ids: list[int], marked: bool) -> None:
        if not file_ids:
            return
        self.table_model.mark_played_many([int(x) for x in file_ids], marked)

    @Slot()
    def _on_clear_played_marks(self) -> None:
        n = self.table_model.clear_played_marks()
        if n > 0:
            self.statusBar().showMessage(f"已清空 {n:,} 条已听标记", 2500)

    @Slot(bool)
    def _on_inspector_favorite_toggled(self, favorite: bool) -> None:
        if self._current_file is None:
            return
        fid = self._current_file.id
        self.store.set_favorites([fid], favorite)
        fresh = self.store.get_file(fid)
        if fresh is not None:
            self._current_file = fresh
        self.table_model.refresh_file_meta([fid])
        self.folder_tree.refresh_quick_counts()

    @Slot(int, str)
    def _on_inspector_note_changed(self, file_id: int, note: str) -> None:
        self.store.set_note(file_id, note)
        self.table_model.refresh_file_meta([file_id])
        if self._current_file is not None and self._current_file.id == file_id:
            fresh = self.store.get_file(file_id)
            if fresh is not None:
                self._current_file = fresh

    @Slot(object)
    def _on_sidebar_view_selected(self, target: SidebarViewTarget) -> None:
        self._clear_search_for_directory_browse()
        if target != self._active_sidebar:
            self.table_model.clear_played_marks()
        self._active_sidebar = target
        self.path_edit.setText(target.display_path or "我的合集")

        if target.kind == KIND_COLLECTION_FOLDER and target.root_id is not None:
            loc = TreeLocation(
                target.root_id,
                target.folder_id,
                target.folder_rel_path,
                target.display_path or "",
            )
            if loc != self._location:
                self._location = loc
                del self._history[self._history_idx + 1 :]
                self._history.append(loc)
                self._history_idx = len(self._history) - 1
            self._update_nav_buttons()

        if target.kind == KIND_RECENT:
            if self._saved_sort_before_recent is None:
                self._saved_sort_before_recent = (self._sort_column, self._sort_asc)
            self._sort_column = "played_at"
            self._sort_asc = False
        else:
            self._leave_recent_sort_if_needed()

        if target.kind == KIND_DUPLICATES:
            self._enter_duplicate_mode()
            return
        if self._dup_mode:
            self._dup_mode = False
            self.dup_banner.setVisible(False)
            self._apply_col_layout(
                "ai" if (self._ai_mode or self._similar_request is not None) else "keyword",
                snapshot_previous=False,
            )
        self._refresh_active_view(from_navigation=True)

    @Slot()
    def _on_import_reaper_shortcuts(self, *, reaper_dir: Path | None = None) -> None:
        ini_path = default_reaper_ini_path(reaper_dir)
        dlg = ReaperShortcutsImportDialog(self.store, ini_path=ini_path, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.import_report is None:
            return
        report = dlg.import_report
        if dlg.collapse_after_import() and report.imported_collection_ids:
            collapsed = set(self.folder_tree.collapsed_collection_ids())
            collapsed.update(int(cid) for cid in report.imported_collection_ids)
            self.folder_tree.set_collapsed_collection_ids(sorted(collapsed))
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._flush_saved_settings()
        total_groups = report.collections_created + report.collections_reused
        self.statusBar().showMessage(
            f"已从 REAPER 导入 {total_groups} 个分组（新增 {report.folders_added} / 更新 {report.folders_updated} 个目录）",
            4500,
        )

    @Slot()
    def _on_create_collection(self) -> None:
        name, ok = QInputDialog.getText(self, "新建合集", "合集名称：")
        if not ok or not name.strip():
            return
        coll = self.store.create_collection(name.strip())
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self.statusBar().showMessage(f"已创建合集：{coll.name}", 2500)

    @Slot(int)
    def _on_rename_collection(self, collection_id: int) -> None:
        colls = {c.id: c for c in self.store.list_collections()}
        old = colls.get(collection_id)
        if old is None:
            return
        name, ok = QInputDialog.getText(self, "重命名合集", "新的合集名称：", text=old.name)
        if not ok or not name.strip():
            return
        self.store.rename_collection(collection_id, name.strip())
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)

    @Slot(int)
    def _on_delete_collection(self, collection_id: int) -> None:
        self.store.delete_collection(collection_id)
        if self._active_sidebar is not None and self._active_sidebar.collection_id == collection_id:
            self._active_sidebar = None
            self.path_edit.setText(self._location.display_path)
            self._refresh_active_view()
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)

    @Slot(int)
    def _on_remove_collection_folder(self, entry_id: int) -> None:
        self.store.remove_folder_from_collection(entry_id)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_active_view()

    @Slot(list)
    def _on_roots_reordered(self, root_ids: list) -> None:
        self.store.reorder_roots([int(rid) for rid in root_ids])
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)

    @Slot(list)
    def _on_collections_reordered(self, collection_ids: list) -> None:
        self.store.reorder_collections([int(cid) for cid in collection_ids])
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)

    @Slot(int, int, int)
    def _on_collection_folder_moved(
        self, entry_id: int, target_collection_id: int, target_index: int
    ) -> None:
        moved = self.store.move_or_reorder_collection_folder(
            int(entry_id), int(target_collection_id), int(target_index)
        )
        if not moved:
            return
        collapsed = set(self.folder_tree.collapsed_collection_ids())
        layout_changed = int(target_collection_id) in collapsed
        if layout_changed:
            collapsed.discard(int(target_collection_id))
            self.folder_tree.set_collapsed_collection_ids(collapsed)

        if (
            self._active_sidebar is not None
            and self._active_sidebar.kind == KIND_COLLECTION_FOLDER
        ):
            target_cfs = self.store.list_collection_folders(int(target_collection_id))
            matched = next(
                (
                    cf
                    for cf in target_cfs
                    if cf.id == int(entry_id)
                    or (
                        cf.root_id == self._active_sidebar.root_id
                        and cf.folder_rel_path == self._active_sidebar.folder_rel_path
                    )
                ),
                None,
            )
            if matched is not None:
                self._active_sidebar = SidebarViewTarget(
                    kind=KIND_COLLECTION_FOLDER,
                    collection_id=matched.collection_id,
                    root_id=matched.root_id,
                    folder_id=matched.folder_id,
                    folder_rel_path=matched.folder_rel_path,
                    display_path=self._active_sidebar.display_path,
                )

        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        if layout_changed:
            self._schedule_save_settings()
        if self._active_sidebar is not None and self._active_sidebar.kind == KIND_COLLECTION:
            self._refresh_active_view()

    @Slot(int, int, object, str, str)
    def _on_add_folder_to_collection(
        self,
        collection_id: int,
        root_id: int,
        folder_id: object,
        rel_path: str,
        label: str,
    ) -> None:
        coll_id = int(collection_id)
        coll_name = ""
        if coll_id <= 0:
            name, ok = QInputDialog.getText(self, "新建并加入合集", "请输入新合集名称：")
            if not ok or not name.strip():
                return
            try:
                coll = self.store.create_collection(name.strip())
            except ValueError as exc:
                QMessageBox.warning(self, "无法创建合集", str(exc))
                return
            coll_id = coll.id
            coll_name = coll.name
        else:
            existing = next((c for c in self.store.list_collections() if c.id == coll_id), None)
            if existing is None:
                return
            coll_name = existing.name
        root = self.store.get_root(root_id)
        clean_label = label.strip() or (Path(rel_path).name if rel_path else (root.label if root else "根目录"))
        fid_int = None if folder_id is None else int(folder_id)
        self.store.add_folder_to_collection(coll_id, root_id, fid_int, rel_path, label=clean_label)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        if self._active_sidebar and self._active_sidebar.collection_id == coll_id:
            self._refresh_active_view()
        self.statusBar().showMessage(f"已将目录「{clean_label}」加入合集「{coll_name}」", 3000)

    @Slot(int, list)
    @Slot(list, int)
    def _on_files_dropped_on_collection(self, a: int | list[int], b: list[int] | int) -> None:
        if isinstance(a, int):
            coll_id, file_ids = a, [int(x) for x in b]  # type: ignore[union-attr]
        else:
            file_ids, coll_id = [int(x) for x in a], int(b)  # type: ignore[arg-type]
        if not file_ids:
            return
        self.store.add_files_to_collection(coll_id, file_ids)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        if self._active_sidebar and self._active_sidebar.collection_id == coll_id:
            self._refresh_active_view()
        self.statusBar().showMessage(f"已将 {len(file_ids)} 个文件加入合集", 2500)

    @Slot(int, list)
    def _on_urls_dropped_on_collection(self, coll_id: int, paths: list[str]) -> None:
        resolved_targets = {str(Path(p).resolve()).lower() for p in paths}
        by_path = {
            str(r.full_path.resolve()).lower(): r.id for r in self.table_view.selected_records()
        }
        matched_ids = [by_path[k] for k in resolved_targets if k in by_path]
        if len(matched_ids) < len(resolved_targets):
            for idx in range(self.table_model.rowCount()):
                rec = self.table_model.peek_record_at(idx)
                if rec is not None:
                    key = str(rec.full_path.resolve()).lower()
                    if key in resolved_targets and rec.id not in matched_ids:
                        matched_ids.append(rec.id)
        if matched_ids:
            self._on_files_dropped_on_collection(coll_id, matched_ids)

    @Slot(list)
    def _on_remove_files_from_active_collection(self, file_ids: list[int]) -> None:
        cid = self._active_collection_id()
        if cid is None or not file_ids:
            return
        self.store.remove_files_from_collection(cid, file_ids)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_active_view()

    # -- Duplicate Finder (Read-Only Safe) ---------------------------------
    def _enter_duplicate_mode(self) -> None:
        if not self._dup_mode and self._settings_ready:
            self._snapshot_current_col_layout(self._active_col_layout_mode)
        self._dup_mode = True
        self._similar_request = None
        self._ai_search_gen += 1
        self._ai_worker.invalidate(self._ai_search_gen)
        self.ai_banner.setVisible(False)
        self.dup_banner.setVisible(True)
        self.table_view.set_ai_columns_visible(True)
        self.dup_mode_name_btn.setChecked(self._dup_scan_mode == "name_duration")
        self.dup_mode_exact_btn.setChecked(self._dup_scan_mode == "exact")
        self._load_duplicate_table()

    @Slot()
    def _exit_duplicate_mode(self) -> None:
        self._dup_mode = False
        self.dup_banner.setVisible(False)
        if self._active_sidebar is not None and self._active_sidebar.kind == KIND_DUPLICATES:
            self._active_sidebar = None
            self.folder_tree.select_location(self._location.root_id, self._location.folder_id)
            self.path_edit.setText(self._location.display_path)
        self._apply_col_layout(
            "ai" if (self._ai_mode or self._similar_request is not None) else "keyword",
            snapshot_previous=False,
        )
        self._refresh_active_view()

    def _set_duplicate_mode(self, mode: str) -> None:
        changed = mode != self._dup_scan_mode
        self._dup_scan_mode = mode
        self.dup_mode_name_btn.setChecked(mode == "name_duration")
        self.dup_mode_exact_btn.setChecked(mode == "exact")
        if self._dup_mode:
            if changed:
                self.table_model.clear_played_marks()
            self._load_duplicate_table()

    def _load_duplicate_table(self) -> None:
        rows = self.store.list_duplicate_rows(mode=self._dup_scan_mode, include_hidden=True)
        self.table_model.apply_duplicate_results(rows)
        groups, _tot, redundant, wasted = self.store.duplicate_summary(mode=self._dup_scan_mode)
        self.count_label.setText(f"{len(rows):,} 条")
        self.status_view.setText(f"重复项: {groups:,} 组 / {len(rows):,} 条")
        mode_zh = "同源多格式" if self._dup_scan_mode == "name_duration" else "完全字节重复"
        if groups == 0:
            self.dup_banner_label.setText(
                f"当前「{mode_zh}」暂无扫描结果，点击右侧「开始查重」（只读分析，绝不删文件）"
            )
        else:
            mb = wasted / (1024.0 * 1024.0)
            self.dup_banner_label.setText(
                f"「{mode_zh}」共 {groups:,} 组 · {redundant:,} 个可隐藏冗余副本（约 {mb:.1f} MB）"
            )

    @Slot()
    def _toggle_duplicate_scan(self) -> None:
        if self._dup_thread is not None and self._dup_thread.is_alive():
            if self._dup_scanner is not None:
                if self._dup_scanner.is_paused:
                    self._dup_scanner.resume()
                    self.dup_scan_btn.setText("暂停查重")
                else:
                    self._dup_scanner.pause()
                    self.dup_scan_btn.setText("继续查重")
            return

        scanner = DuplicateScanner(self.store.db_path)
        self._dup_scanner = scanner
        mode = self._dup_scan_mode
        self.dup_scan_btn.setText("暂停查重")
        self.dup_banner_label.setText("正在后台执行只读查重扫描…")

        def run_dup() -> None:
            try:
                rep = scanner.scan(mode=mode, on_progress=lambda p: self._dup_signals.progress.emit(p))
                self._dup_signals.done.emit(rep)
            except Exception as exc:
                self._dup_signals.failed.emit(str(exc))

        self._dup_thread = threading.Thread(target=run_dup, daemon=True)
        self._dup_thread.start()

    @Slot(object)
    def _on_dup_scan_progress(self, p: DuplicateProgress) -> None:
        if not self._dup_mode:
            return
        if p.stage == "paused":
            self.dup_banner_label.setText(p.message or "查重已暂停")
            return
        self.dup_banner_label.setText(
            f"查重中：{p.scanned:,}/{p.total:,} · 发现 {p.groups_found:,} 组"
        )

    @Slot(object)
    def _on_dup_scan_done(self, report: DuplicateScanReport) -> None:
        self.dup_scan_btn.setText("开始查重")
        self.folder_tree.refresh_quick_counts()
        if self._dup_mode:
            self._load_duplicate_table()
        mb = report.reclaimable_bytes / (1024.0 * 1024.0)
        self.statusBar().showMessage(
            f"查重完成：发现 {report.groups_found:,} 组重复 · 可隐藏副本 {report.redundant_files:,} 条（约 {mb:.1f} MB）",
            4500,
        )

    @Slot(str)
    def _on_dup_scan_failed(self, message: str) -> None:
        self.dup_scan_btn.setText("开始查重")
        self.dup_banner_label.setText(f"查重失败：{message}")

    @Slot(int, int)
    def _on_set_duplicate_primary(self, group_id: int, file_id: int) -> None:
        self.store.set_duplicate_primary(group_id, file_id)
        if self._dup_mode:
            self._load_duplicate_table()

    @Slot(list)
    def _on_hide_duplicates(self, file_ids: list[int]) -> None:
        n = self.store.hide_files(file_ids, reason="duplicate")
        if self._dup_mode:
            self._load_duplicate_table()
        self.statusBar().showMessage(f"已在库中隐藏 {n} 个副本（磁盘原文件未动）", 3000)

    @Slot()
    def _on_hide_all_non_primary(self) -> None:
        n = self.store.hide_all_non_primary_duplicates(self._dup_scan_mode)
        if self._dup_mode:
            self._load_duplicate_table()
        self.statusBar().showMessage(f"已隐藏全部 {n:,} 个非主版本副本（磁盘原文件完好）", 3500)

    @Slot()
    def _on_unhide_all_duplicates(self) -> None:
        n = self.store.unhide_files()
        if self._dup_mode:
            self._load_duplicate_table()
        else:
            self._refresh_active_view()
        self.statusBar().showMessage(f"已恢复显示 {n:,} 个已隐藏副本", 3000)

    # -- Offline Drives & Root Relocation ----------------------------------
    @Slot()
    @Slot(object)
    def _refresh_online_now(self, root_id: object = None) -> None:
        rid = None if root_id is None else int(root_id)
        self.store.refresh_root_folders_online(rid)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_active_view()
        self.statusBar().showMessage("已刷新目录与盘符在线状态", 2500)

    @Slot()
    def _schedule_online_check(self) -> None:
        if self._online_checking:
            return
        self._online_checking = True
        active_folder = (
            (self._location.root_id, self._location.folder_rel_path)
            if self._location.root_id is not None and self._location.folder_rel_path
            else None
        )
        self._pool.start(_OnlineTask(self.store, self._online_signals, active_folder=active_folder))

    @Slot(bool)
    def _on_online_check_finished(self, changed: bool) -> None:
        self._online_checking = False
        if changed:
            self.folder_tree.reload(
                keep_location=self._location, keep_target=self._active_sidebar
            )
            self._refresh_active_view()

    @Slot(int)
    def _on_relocate_root(self, root_id: int) -> None:
        root = self.store.get_root(root_id)
        if root is None:
            return
        chosen = QFileDialog.getExistingDirectory(
            self,
            f"重定位根目录：{root.path}",
            str(root.path) if Path(root.path).exists() else "",
        )
        if not chosen:
            return
        try:
            updated = self.store.relocate_root(root_id, Path(chosen))
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "重定位失败", str(exc))
            return
        if self._location.root_id == root_id:
            disp = (
                str(Path(updated.path) / Path(self._location.folder_rel_path))
                if self._location.folder_rel_path
                else str(updated.path)
            )
            self._location = TreeLocation(
                root_id, self._location.folder_id, self._location.folder_rel_path, disp
            )
            self.path_edit.setText(disp)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_active_view()
        self.statusBar().showMessage(f"已将根目录重定位到：{updated.path}", 4000)

    @Slot(int, int)
    def _on_relocate_folder(self, root_id: int, folder_id: int) -> None:
        root = self.store.get_root(root_id)
        folder = self.store.get_folder(folder_id)
        if root is None or folder is None:
            return
        old_full = Path(root.path) / Path(folder.rel_path)
        start_dir = str(old_full.parent if old_full.parent.exists() else (Path(root.path) if Path(root.path).exists() else Path.home()))
        chosen = QFileDialog.getExistingDirectory(
            self,
            f"重定位目录：{old_full}",
            start_dir,
        )
        if not chosen:
            return
        try:
            new_rid, new_fid, new_rel, new_disp = self.store.relocate_folder(
                root_id, folder_id, Path(chosen)
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "重定位失败", str(exc))
            return
        if self._location.root_id == root_id and self._location.folder_id == folder_id:
            self._location = TreeLocation(new_rid, new_fid, new_rel, new_disp)
            self.path_edit.setText(new_disp)
        self.folder_tree.reload(keep_location=self._location, keep_target=self._active_sidebar)
        self._refresh_dirty_status()
        self._refresh_active_view()
        self.statusBar().showMessage(f"已将目录重定位到：{new_disp}", 4000)

    @Slot(int, str)
    def _on_ai_search_failed(self, generation: int, message: str) -> None:
        if generation != self._ai_search_gen:
            return
        self.ai_banner.setVisible(True)
        self.ai_banner_label.setText(f"AI 检索失败：{message}")
        self.status_view.setText("AI 检索异常")

    @Slot()
    def _toggle_ai_indexing(self) -> None:
        if self._ai_index_thread is not None and self._ai_index_thread.is_alive():
            if self._ai_indexer is not None:
                if self._ai_indexer.is_paused:
                    self._ai_indexer.resume()
                    self.ai_index_btn.setText("暂停建库")
                else:
                    self._ai_indexer.pause()
                    self.ai_index_btn.setText("继续建库")
            return

        from audiosearch.ai.indexer import VectorIndexer

        indexer = VectorIndexer(self.store.db_path, profile=self._settings.ai_profile)
        self._ai_indexer = indexer
        self.ai_index_btn.setText("暂停建库")
        self.status_ai.setText("AI 建库启动中…")

        def run_index() -> None:
            try:
                report = indexer.run(on_progress=lambda p: self._ai_index_signals.progress.emit(p))
                self._ai_index_signals.done.emit(report)
            except Exception as exc:
                try:
                    import traceback
                    from datetime import datetime

                    log_path = self.store.db_path.parent / "logs" / "browser-crash.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().isoformat(timespec="seconds")
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"\n[{ts}] AI indexing failed\n")
                        traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
                except OSError:
                    pass
                self._ai_index_signals.failed.emit(str(exc))

        self._ai_index_thread = threading.Thread(target=run_index, daemon=True)
        self._ai_index_thread.start()

    @Slot(object)
    def _on_ai_index_progress(self, p: Any) -> None:
        total = max(1, 2 * int(p.total_files))
        pct = min(100, int(round(((p.text_done + p.audio_done) / total) * 100.0)))
        if p.status == "paused":
            self.status_ai.setText(f"AI 建库已暂停 {pct}%")
            return
        rate = f" · {p.items_per_sec:.0f}/s" if p.items_per_sec >= 1.0 else ""
        self.status_ai.setText(
            f"AI 建库中 {pct}% (文本 {p.text_done:,}/{p.total_files:,} · 声波 {p.audio_done:,}/{p.total_files:,}{rate})"
        )

    @Slot(object)
    def _on_ai_index_done(self, report: Any) -> None:
        self.ai_index_btn.setText("AI 建库")
        self._refresh_ai_coverage()
        if self._ai_mode and self._similar_request is None and not self.search_input.text().strip():
            self._update_ai_banner_idle()
        self.statusBar().showMessage(
            f"AI 建库结束（新增文本 {report.encoded_text:,}、声波 {report.encoded_audio:,}，耗时 {report.elapsed_seconds:.1f}s）",
            4500,
        )

    @Slot(str)
    def _on_ai_index_failed(self, message: str) -> None:
        self.ai_index_btn.setText("AI 建库")
        lines = [line.strip() for line in str(message).splitlines() if line.strip()]
        summary = lines[0] if lines else "未知错误"
        if len(summary) > 96:
            summary = summary[:95] + "…"
        self.status_ai.setText(f"AI 建库失败: {summary}")
        self.status_ai.setToolTip(
            f"AI 建库失败：\n{message}\n\n完整堆栈已写入 logs/browser-crash.log"
        )

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._use_native_frameless and not self._win32_frame_applied:
            self._win32_frame_applied = True
            apply_win32_frameless_style(self)
        if hasattr(self, "title_bar"):
            self.title_bar.set_maximized(self.isMaximized())

    def nativeEvent(self, event_type, message):  # noqa: N802
        if self._use_native_frameless and hasattr(self, "title_bar"):
            res = handle_win32_native_event(self, self.title_bar, event_type, message)
            if res is not None:
                return res
        return super().nativeEvent(event_type, message)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if getattr(self, "_settings_ready", False):
            self._schedule_save_settings()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, "title_bar"):
                self.title_bar.set_maximized(self.isMaximized())
            if getattr(self, "_settings_ready", False):
                self._schedule_save_settings()

    # -- Shutdown ----------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._search_timer.stop()
        self._history_record_timer.stop()
        self._zoom_timer.stop()
        self._reaper_timer.stop()
        self._online_timer.stop()
        self._save_settings_timer.stop()
        self._silent_video_timer.stop()
        self.inspector.flush_pending_note()
        self._flush_saved_settings()
        if self._scanner is not None:
            self._scanner.cancel()
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=1.5)
        if self._dup_scanner is not None:
            self._dup_scanner.cancel()
        if self._dup_thread is not None and self._dup_thread.is_alive():
            self._dup_thread.join(timeout=1.5)
        if self._ai_indexer is not None:
            self._ai_indexer.cancel()
        if self._ai_index_thread is not None and self._ai_index_thread.is_alive():
            self._ai_index_thread.join(timeout=2.0)
        engine_to_close = self._ai_engine if self._ai_owns_engine else None
        self._ai_worker.shutdown(engine_to_close=engine_to_close, timeout=2.0)
        if self._ai_owns_index and self._ai_index is not None and hasattr(self._ai_index, "close"):
            self._ai_index.close()
        self.inspector.shutdown()
        self.audition.close()
        self.table_model.shutdown()
        self.thumbnail_manager.shutdown()
        self._pool.waitForDone(1500)
        self.store.close()
        super().closeEvent(event)

    # -- R5a: Metadata Editing, UCS & Safe Write-Back ----------------------
    def _refresh_dirty_status(self) -> None:
        count = self.store.dirty_file_count()
        self.write_back_btn.setText(f"待写回: {count:,}")
        self.write_back_btn.setVisible(count > 0)

    def _refresh_after_metadata_change(self, file_ids: list[int] | None = None) -> None:
        self._refresh_dirty_status()
        if file_ids:
            self.table_model.refresh_file_meta(set(int(x) for x in file_ids))
            if self._current_file is not None and self._current_file.id in file_ids:
                fresh = self.store.get_file(self._current_file.id)
                if fresh is not None:
                    self._current_file = fresh
                    self.inspector.show_file(
                        fresh,
                        self.store.get_metadata_fields(fresh.id),
                        ai_hit=self.table_model.ai_hit_for(fresh.id),
                    )
            return
        self._refresh_active_view()
        if self._current_file is not None:
            fresh = self.store.get_file(self._current_file.id)
            if fresh is not None:
                self._current_file = fresh
                self.inspector.show_file(
                    fresh,
                    self.store.get_metadata_fields(fresh.id),
                    ai_hit=self.table_model.ai_hit_for(fresh.id),
                )

    @Slot(int, str, str)
    def _on_inline_metadata_edited(self, file_id: int, field_key: str, new_value: str) -> None:
        self.store.update_file_metadata(file_id, {field_key: new_value}, source="user")
        self._refresh_after_metadata_change([file_id])
        self.statusBar().showMessage("已保存到工具库并标记「待写回」（源文件尚未改动）", 3000)

    @Slot(list)
    def _on_edit_metadata_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        if not ids:
            return
        dlg = MetadataEditorDialog(self.store, ids, parent=self)
        dlg.metadataSaved.connect(lambda changed: self._refresh_after_metadata_change(changed))
        dlg.exec()

    @Slot(list)
    def _on_parse_ucs_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        if not ids:
            return
        recs = self.store.get_files_by_ids(ids)
        updated: list[int] = []
        for fid in ids:
            rec = recs.get(fid)
            if rec is None:
                continue
            parsed = parse_ucs_filename(rec.filename)
            if parsed.matched and parsed.fields:
                self.store.update_file_metadata(fid, parsed.fields, source="user")
                updated.append(fid)
        if updated:
            self._refresh_after_metadata_change(updated)
            self.statusBar().showMessage(
                f"已从文件名解析并更新 {len(updated):,} 条 UCS 元数据（仅存工具库）", 3500
            )
        else:
            self.statusBar().showMessage("选中文件名中未检测到可识别的 UCS 规范前缀", 3500)

    def _prepare_files_for_disk_write(self, file_ids: list[int]) -> None:
        if self._current_file is not None and (not file_ids or self._current_file.id in file_ids):
            self.audition.stop()

    @Slot(list)
    def _on_write_back_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        dirty_set = set(self.store.dirty_file_ids())
        filtered = [fid for fid in ids if fid in dirty_set]
        self._open_write_back_dialog(filtered if filtered else None)

    def _open_write_back_dialog(self, file_ids: list[int] | None) -> None:
        dlg = WriteBackDialog(
            self.store,
            file_ids,
            before_disk_write=self._prepare_files_for_disk_write,
            parent=self,
        )
        dlg.libraryChanged.connect(lambda: self._refresh_after_metadata_change(None))
        dlg.exec()

    # -- R5b: AI Enrichment, Batch Rename & CSV Tools ----------------------
    @Slot(object, object)
    def _on_custom_ai_done(self, callback: Any, result: Any) -> None:
        if callable(callback):
            callback(result)

    @Slot(object, str)
    def _on_custom_ai_failed(self, callback: Any, message: str) -> None:
        if callable(callback):
            callback(message)

    @Slot(list)
    def _on_ai_enrich_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        if not ids:
            return
        idx = self._ensure_ai_index()
        dlg = AiEnrichDialog(
            self.store,
            ids,
            run_ai_job=self._ai_worker.run_custom,
            ai_index=idx,
            ai_engine_provider=self._ensure_ai_engine,
            parent=self,
        )
        dlg.metadataSaved.connect(lambda changed: self._refresh_after_metadata_change(changed))
        dlg.exec()

    @Slot(list)
    def _on_batch_rename_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        if not ids:
            return
        dlg = BatchRenameDialog(
            self.store,
            ids,
            before_disk_rename=self._prepare_files_for_disk_write,
            parent=self,
        )
        dlg.libraryChanged.connect(lambda: self._refresh_after_metadata_change(None))
        dlg.exec()

    @Slot(list)
    def _on_csv_metadata_requested(self, file_ids: list[int]) -> None:
        ids = [int(x) for x in file_ids]
        dlg = CsvMetadataDialog(self.store, ids, parent=self)
        dlg.libraryChanged.connect(lambda: self._refresh_after_metadata_change(None))
        dlg.exec()


def _fmt_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins, rem = divmod(seconds, 60.0)
    return f"{int(mins)}:{rem:06.3f}"


def _fmt_short_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins, rem = divmod(seconds, 60.0)
    return f"{int(mins)}:{int(rem):02d}"


def create_app(argv: list[str] | None = None) -> QApplication:
    from audiosearch.browser.runtime_paths import resolve_app_icon_path

    app = QApplication.instance()
    if app is None:
        app = QApplication(argv or [])
    app.setApplicationName("Whistle")
    app.setApplicationDisplayName("Whistle")
    apply_theme(app)
    icon_path = resolve_app_icon_path()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            app.setWindowIcon(icon)
    return app
