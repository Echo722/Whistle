"""Waveform table cell delegate (QPixmap cached) and viewport-driven thumbnail loader."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QModelIndex, QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTableView, QWidget

from audiosearch.browser import theme
from audiosearch.browser.theme import spectral_color_luts
from audiosearch.library.peaks import (
    compute_thumbnail_with_spectral,
    decode_thumbnail_spectral,
    resample_spectral_columns,
    shape_thumbnail_envelope,
    thumbnail_width_fraction,
)
from audiosearch.library.store import LibraryStore

PAD_X = 6
PAD_Y = 3

ZERO_LINE = QColor("#34322E")
ZERO_LINE_SELECTED = QColor("#594238")
END_CAP = QColor("#45423B")
PEAK_NORMAL = QColor("#878177")
PEAK_SELECTED = QColor("#D5CEC2")
PEAK_OFFLINE = QColor("#4F4B45")
PEAK_PLAYED = QColor("#D97757")
PLAYHEAD = QColor("#E38566")
HOVER_LINE = QColor(217, 119, 87, 165)


def _first_visible_visual_col(table: QTableView) -> int:
    header = table.horizontalHeader()
    count = header.count()
    first = count
    for logical in range(count):
        if not table.isColumnHidden(logical):
            vis = header.visualIndex(logical)
            if 0 <= vis < first:
                first = vis
    return first


def paint_selection_chrome(
    painter: QPainter, opt: QStyleOptionViewItem, index: QModelIndex
) -> None:
    """Paint top/bottom row borders and 3px left accent bar on selected cells."""
    if not (opt.state & QStyle.StateFlag.State_Selected):
        return
    is_active = bool(opt.state & QStyle.StateFlag.State_Active)
    border_color = theme.SELECT_BORDER if is_active else theme.SELECT_BORDER_INACTIVE
    mark_color = theme.SELECT_MARK if is_active else theme.SELECT_MARK_INACTIVE
    r = opt.rect
    if not r.isValid():
        return

    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(border_color, 1))
        painter.drawLine(r.left(), r.top(), r.right(), r.top())
        painter.drawLine(r.left(), r.bottom(), r.right(), r.bottom())

        if isinstance(opt.widget, QTableView):
            header = opt.widget.horizontalHeader()
            cached = getattr(opt.widget, "_first_visible_visual_col_cached", None)
            first_vis = int(cached) if cached is not None else _first_visible_visual_col(opt.widget)
            if header.visualIndex(index.column()) == first_vis:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(mark_color)
                painter.drawRect(QRectF(float(r.left()), float(r.top()), 3.0, float(r.height())))
    finally:
        painter.restore()


class WaveformItemDelegate(QStyledItemDelegate):
    """Renders mono waveform thumbnails, live playhead, and hover seek guide."""

    def __init__(self, parent: QWidget | None = None, *, max_pixmaps: int = 512) -> None:
        super().__init__(parent)
        self._max_pixmaps = max_pixmaps
        self._pixmap_cache: OrderedDict[tuple[int, int, int, int, int, int, int, int, int], QPixmap] = (
            OrderedDict()
        )
        self._playing_file_id: int | None = None
        self._playing_ratio: float | None = None
        self._hover_row: int = -1
        self._hover_ratio: float | None = None
        self._spectral_peaks: bool = False

    def set_spectral_peaks(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._spectral_peaks == enabled:
            return
        self._spectral_peaks = enabled
        self.clear_cache()

    @staticmethod
    def waveform_rect(cell_rect: QRect, duration_seconds: float) -> QRect:
        inner = cell_rect.adjusted(PAD_X, PAD_Y, -PAD_X, -PAD_Y)
        if inner.width() <= 2 or inner.height() <= 2:
            return QRect()
        draw_w = min(
            inner.width(),
            max(4, int(round(inner.width() * thumbnail_width_fraction(duration_seconds)))),
        )
        return QRect(inner.left(), inner.top(), draw_w, inner.height())

    @staticmethod
    def seek_ratio_for_x(
        cell_rect: QRect, x: float, duration_seconds: float = 0.0
    ) -> float | None:
        wrect = WaveformItemDelegate.waveform_rect(cell_rect, duration_seconds)
        if not wrect.isValid() or wrect.width() <= 0:
            return None
        if x > wrect.right() + 2:
            return None
        return max(0.0, min(1.0, float((x - wrect.left()) / max(1, wrect.width()))))

    def set_playing_state(self, file_id: int | None, ratio: float | None) -> None:
        self._playing_file_id = file_id
        self._playing_ratio = None if ratio is None else max(0.0, min(1.0, float(ratio)))

    def set_hover_state(self, row: int, ratio: float | None) -> None:
        self._hover_row = row
        self._hover_ratio = None if ratio is None else max(0.0, min(1.0, float(ratio)))

    def invalidate_file(self, file_id: int) -> None:
        stale = [k for k in self._pixmap_cache if k[0] == file_id]
        for k in stale:
            self._pixmap_cache.pop(k, None)

    def clear_cache(self) -> None:
        self._pixmap_cache.clear()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        opt.text = ""
        opt.icon = QIcon()
        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        inner = opt.rect.adjusted(PAD_X, PAD_Y, -PAD_X, -PAD_Y)
        if inner.width() <= 2 or inner.height() <= 2:
            paint_selection_chrome(painter, opt, index)
            return

        model = index.model()
        if hasattr(model, "peek_record_at"):
            rec = model.peek_record_at(index.row())
        elif hasattr(model, "record_at"):
            rec = model.record_at(index.row())
        else:
            rec = None
        if rec is None:
            paint_selection_chrome(painter, opt, index)
            return

        wrect = self.waveform_rect(opt.rect, rec.duration_sec)
        if not wrect.isValid():
            paint_selection_chrome(painter, opt, index)
            return
        draw_w = wrect.width()
        draw_h = wrect.height()
        has_end_cap = draw_w < inner.width()

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            is_selected = bool(opt.state & QStyle.StateFlag.State_Selected)
            thumb_blob = model.thumbnail_for(rec) if hasattr(model, "thumbnail_for") else rec.thumb_blob
            spectral_blob = (
                model.spectral_thumbnail_for(rec)
                if hasattr(model, "spectral_thumbnail_for")
                else rec.spectral_thumb_blob
            )
            is_playing = (
                self._playing_file_id is not None
                and rec.id == self._playing_file_id
                and self._playing_ratio is not None
            )

            if thumb_blob:
                if is_playing:
                    self._paint_playing_cell(
                        painter,
                        wrect,
                        thumb_blob,
                        self._playing_ratio or 0.0,
                        is_selected=is_selected,
                        is_available=rec.is_available,
                        has_end_cap=has_end_cap,
                        spectral_blob=spectral_blob,
                    )
                else:
                    dpr = float(opt.widget.devicePixelRatioF()) if opt.widget is not None else 1.0
                    dpr = max(1.0, dpr)
                    dpr_int = max(100, int(round(dpr * 100)))
                    state_key = 2 if not rec.is_available else (1 if is_selected else 0)
                    has_spec_key = int(bool(spectral_blob)) if self._spectral_peaks else 0
                    cache_key = (
                        rec.id,
                        rec.mtime_ns,
                        draw_w,
                        draw_h,
                        state_key,
                        dpr_int,
                        int(has_end_cap),
                        int(self._spectral_peaks),
                        has_spec_key,
                    )
                    pixmap = self._pixmap_cache.get(cache_key)
                    if pixmap is None:
                        pixmap = self._render_static_pixmap(
                            draw_w,
                            draw_h,
                            thumb_blob,
                            dpr=dpr,
                            is_selected=is_selected,
                            is_available=rec.is_available,
                            has_end_cap=has_end_cap,
                            spectral_blob=spectral_blob,
                        )
                        self._pixmap_cache[cache_key] = pixmap
                        while len(self._pixmap_cache) > self._max_pixmaps:
                            self._pixmap_cache.popitem(last=False)
                    else:
                        self._pixmap_cache.move_to_end(cache_key)
                    painter.drawPixmap(wrect.topLeft(), pixmap)
            else:
                mid_y = wrect.top() + draw_h / 2.0
                zero_color = ZERO_LINE_SELECTED if is_selected else ZERO_LINE
                painter.setPen(QPen(zero_color, 1))
                painter.drawLine(QPointF(wrect.left(), mid_y), QPointF(wrect.right(), mid_y))
                if has_end_cap:
                    cap_x = float(wrect.right())
                    painter.setPen(QPen(END_CAP, 1))
                    painter.drawLine(
                        QPointF(cap_x, wrect.top() + 2.0),
                        QPointF(cap_x, max(wrect.top() + 2.0, float(wrect.bottom()) - 2.0)),
                    )
                if is_playing:
                    play_x = min(
                        float(wrect.right()),
                        wrect.left() + (self._playing_ratio or 0.0) * draw_w,
                    )
                    painter.setPen(QPen(PLAYHEAD, 1.5))
                    painter.drawLine(QPointF(play_x, wrect.top()), QPointF(play_x, wrect.bottom()))

            if index.row() == self._hover_row and self._hover_ratio is not None:
                hx = min(float(wrect.right()), wrect.left() + self._hover_ratio * draw_w)
                painter.setPen(QPen(HOVER_LINE, 1))
                painter.drawLine(QPointF(hx, wrect.top()), QPointF(hx, wrect.bottom()))
        finally:
            painter.restore()
        paint_selection_chrome(painter, opt, index)

    def _render_static_pixmap(
        self,
        width: int,
        height: int,
        thumb_blob: bytes,
        *,
        dpr: float,
        is_selected: bool,
        is_available: bool,
        has_end_cap: bool,
        spectral_blob: bytes | None = None,
    ) -> QPixmap:
        px_w = max(1, int(round(width * dpr)))
        px_h = max(1, int(round(height * dpr)))
        pixmap = QPixmap(px_w, px_h)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)

        p = QPainter()
        if p.begin(pixmap):
            try:
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen_w = 1.0 / dpr
                mid_y = height / 2.0
                p.setPen(QPen(ZERO_LINE_SELECTED if is_selected else ZERO_LINE, pen_w))
                p.drawLine(QPointF(0.0, mid_y), QPointF(float(max(0, width - 1)), mid_y))

                envelope = shape_thumbnail_envelope(thumb_blob)
                spec_arr = (
                    decode_thumbnail_spectral(spectral_blob)
                    if (self._spectral_peaks and is_available and spectral_blob)
                    else np.zeros((0, 2), dtype=np.uint8)
                )
                use_spectral = envelope.shape[0] > 0 and spec_arr.shape[0] == envelope.shape[0]
                if use_spectral:
                    samples, col_spec = resample_spectral_columns(envelope, spec_arr, px_w)
                    if samples.size > 0:
                        amp = max(height / 2.0 - 1.5, 2.0)
                        normal_lut, _ = spectral_color_luts()
                        last_key: tuple[int, int] | None = None
                        for col in range(px_w):
                            half = float(samples[col]) * amp
                            if half >= 0.35:
                                key = (int(col_spec[col, 0]) >> 2, int(col_spec[col, 1]) >> 5)
                                if key != last_key:
                                    last_key = key
                                    p.setPen(QPen(normal_lut[key[0]][key[1]], pen_w))
                                xf = (col + 0.5) / dpr
                                p.drawLine(QPointF(xf, mid_y - half), QPointF(xf, mid_y + half))
                else:
                    samples = _sample_columns(envelope, px_w)
                    if samples.size > 0:
                        amp = max(height / 2.0 - 1.5, 2.0)
                        color = (
                            PEAK_OFFLINE
                            if not is_available
                            else (PEAK_SELECTED if is_selected else PEAK_NORMAL)
                        )
                        p.setPen(QPen(color, pen_w))
                        for col in range(px_w):
                            half = float(samples[col]) * amp
                            if half >= 0.35:
                                xf = (col + 0.5) / dpr
                                p.drawLine(QPointF(xf, mid_y - half), QPointF(xf, mid_y + half))

                if has_end_cap and width >= 4:
                    cap_x = float(width) - 0.5 / dpr
                    p.setPen(QPen(END_CAP, pen_w))
                    p.drawLine(
                        QPointF(cap_x, 2.0),
                        QPointF(cap_x, max(2.0, float(height) - 2.0)),
                    )
            finally:
                p.end()
        return pixmap

    def _paint_playing_cell(
        self,
        painter: QPainter,
        wrect: QRect,
        thumb_blob: bytes,
        ratio: float,
        *,
        is_selected: bool,
        is_available: bool,
        has_end_cap: bool,
        spectral_blob: bytes | None = None,
    ) -> None:
        width = wrect.width()
        height = wrect.height()
        dev = painter.device()
        dpr = max(1.0, float(dev.devicePixelRatioF())) if dev is not None else 1.0
        px_w = max(1, int(round(width * dpr)))
        pen_w = 1.0 / dpr

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            mid_y = wrect.top() + height / 2.0
            painter.setPen(QPen(ZERO_LINE_SELECTED if is_selected else ZERO_LINE, pen_w))
            painter.drawLine(QPointF(wrect.left(), mid_y), QPointF(wrect.right(), mid_y))

            play_x = min(
                float(wrect.right()),
                wrect.left() + max(0.0, min(1.0, ratio)) * width,
            )
            envelope = shape_thumbnail_envelope(thumb_blob)
            spec_arr = (
                decode_thumbnail_spectral(spectral_blob)
                if (self._spectral_peaks and is_available and spectral_blob)
                else np.zeros((0, 2), dtype=np.uint8)
            )
            use_spectral = envelope.shape[0] > 0 and spec_arr.shape[0] == envelope.shape[0]
            if use_spectral:
                samples, col_spec = resample_spectral_columns(envelope, spec_arr, px_w)
                if samples.size > 0:
                    amp = max(height / 2.0 - 1.5, 2.0)
                    normal_lut, played_lut = spectral_color_luts()
                    last_spec_key: tuple[int, int, bool] | None = None
                    for col in range(px_w):
                        x = float(wrect.left()) + (col + 0.5) / dpr
                        is_played = x <= play_x
                        half = float(samples[col]) * amp
                        if half >= 0.35:
                            key = (int(col_spec[col, 0]) >> 2, int(col_spec[col, 1]) >> 5, is_played)
                            if key != last_spec_key:
                                last_spec_key = key
                                color = (played_lut if is_played else normal_lut)[key[0]][key[1]]
                                painter.setPen(QPen(color, pen_w))
                            painter.drawLine(QPointF(x, mid_y - half), QPointF(x, mid_y + half))
            else:
                samples = _sample_columns(envelope, px_w)
                if samples.size > 0:
                    amp = max(height / 2.0 - 1.5, 2.0)
                    unplayed_color = (
                        PEAK_OFFLINE
                        if not is_available
                        else (PEAK_SELECTED if is_selected else PEAK_NORMAL)
                    )
                    last_played: bool | None = None
                    for col in range(px_w):
                        x = float(wrect.left()) + (col + 0.5) / dpr
                        is_played = x <= play_x
                        if is_played != last_played:
                            last_played = is_played
                            painter.setPen(QPen(PEAK_PLAYED if is_played else unplayed_color, pen_w))
                        half = float(samples[col]) * amp
                        if half >= 0.35:
                            painter.drawLine(QPointF(x, mid_y - half), QPointF(x, mid_y + half))

            if has_end_cap and width >= 4:
                cap_x = float(wrect.right()) - 0.5 / dpr
                painter.setPen(QPen(END_CAP, pen_w))
                painter.drawLine(
                    QPointF(cap_x, wrect.top() + 2.0),
                    QPointF(cap_x, max(wrect.top() + 2.0, float(wrect.bottom()) - 2.0)),
                )

            painter.setPen(QPen(PLAYHEAD, 1.5))
            painter.drawLine(QPointF(play_x, wrect.top()), QPointF(play_x, wrect.bottom()))
        finally:
            painter.restore()


def _sample_columns(envelope: np.ndarray, cols: int) -> np.ndarray:
    """Resample 1D peak envelope into ``cols`` physical pixel columns (interp up, max down)."""
    n_pts = int(envelope.shape[0])
    if cols <= 0 or n_pts <= 0:
        return np.zeros(0, dtype=np.float32)
    if cols == n_pts:
        return envelope.astype(np.float32, copy=False)
    if cols > n_pts:
        xp = np.arange(n_pts, dtype=np.float32)
        x = np.linspace(0.0, float(n_pts - 1), cols, dtype=np.float32)
        return np.interp(x, xp, envelope).astype(np.float32, copy=False)
    out = np.empty(cols, dtype=np.float32)
    for col in range(cols):
        i0 = (col * n_pts) // cols
        i1 = ((col + 1) * n_pts) // cols
        if i1 <= i0:
            i1 = min(n_pts, i0 + 1)
        out[col] = float(envelope[i0:i1].max())
    return out


class ThumbnailManager(QObject):
    """Viewport-prioritized background waveform thumbnail decoder and batch writer."""

    thumbnailsReady = Signal()

    def __init__(
        self,
        store: LibraryStore,
        parent: QObject | None = None,
        *,
        max_threads: int = 4,
        mem_cache_size: int = 4096,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._max_threads = max(1, max_threads)
        self._mem_cache_size = max(64, mem_cache_size)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._db_write_lock = threading.Lock()
        self._closed = False
        self._pending: deque[tuple[int, int, Path]] = deque()
        self._pending_keys: set[tuple[int, int]] = set()
        self._in_flight: set[tuple[int, int]] = set()
        self._ready_ids: set[int] = set()
        self._threads: list[threading.Thread] = []
        self._mem_cache: OrderedDict[tuple[int, int], tuple[bytes, bytes | None]] = OrderedDict()
        self._failed_keys: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._require_spectral = False

    def get_cached(self, file_id: int, mtime_ns: int) -> bytes | None:
        key = (file_id, mtime_ns)
        with self._lock:
            val = self._mem_cache.get(key)
            if val is not None:
                self._mem_cache.move_to_end(key)
                return val[0]
            return None

    def get_cached_spectral(self, file_id: int, mtime_ns: int) -> bytes | None:
        key = (file_id, mtime_ns)
        with self._lock:
            val = self._mem_cache.get(key)
            if val is not None:
                self._mem_cache.move_to_end(key)
                return val[1]
            return None

    def drain_ready_ids(self) -> set[int]:
        with self._lock:
            if not self._ready_ids:
                return set()
            ids = set(self._ready_ids)
            self._ready_ids.clear()
            return ids

    def inject(
        self,
        file_id: int,
        mtime_ns: int,
        blob: bytes,
        spectral_blob: bytes | None = None,
    ) -> None:
        if not blob:
            return
        key = (file_id, mtime_ns)
        spec_clean = spectral_blob or None
        should_notify = False
        with self._lock:
            if self._closed:
                return
            existing = self._mem_cache.get(key)
            if spec_clean is None and existing is not None:
                spec_clean = existing[1]
            self._put_mem_cache_locked(key, (blob, spec_clean))
            self._failed_keys.pop(key, None)
            if file_id not in self._ready_ids:
                self._ready_ids.add(file_id)
                should_notify = len(self._ready_ids) == 1
        if should_notify:
            try:
                self.thumbnailsReady.emit()
            except RuntimeError:
                pass

    def request_visible(
        self, items: list[tuple[int, int, Path]], *, require_spectral: bool = False
    ) -> None:
        with self._cond:
            if self._closed:
                return
            self._require_spectral = bool(require_spectral)
            self._pending.clear()
            self._pending_keys.clear()
            for file_id, mtime_ns, path in items[:128]:
                key = (file_id, mtime_ns)
                cached = self._mem_cache.get(key)
                has_enough = cached is not None and (not require_spectral or cached[1] is not None)
                if (
                    has_enough
                    or key in self._failed_keys
                    or key in self._in_flight
                    or key in self._pending_keys
                ):
                    continue
                self._pending.append((file_id, mtime_ns, path))
                self._pending_keys.add(key)

            if self._pending:
                while len(self._threads) < self._max_threads:
                    worker = threading.Thread(
                        target=self._worker_loop,
                        name=f"thumb-worker-{len(self._threads)}",
                        daemon=True,
                    )
                    self._threads.append(worker)
                    worker.start()
                self._cond.notify_all()

    def shutdown(self, timeout_ms: int = 1500) -> None:
        with self._cond:
            self._closed = True
            self._pending.clear()
            self._pending_keys.clear()
            self._ready_ids.clear()
            threads = list(self._threads)
            self._cond.notify_all()
        deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
        for worker in threads:
            rem = max(0.0, deadline - time.monotonic())
            worker.join(timeout=rem)

    def _put_mem_cache_locked(self, key: tuple[int, int], entry: tuple[bytes, bytes | None]) -> None:
        self._mem_cache[key] = entry
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > self._mem_cache_size:
            self._mem_cache.popitem(last=False)

    def _mark_failed_locked(self, key: tuple[int, int]) -> None:
        self._failed_keys[key] = None
        self._failed_keys.move_to_end(key)
        while len(self._failed_keys) > self._mem_cache_size:
            self._failed_keys.popitem(last=False)

    def _worker_loop(self) -> None:
        write_buffer: list[tuple[int, int, bytes, bytes | None]] = []
        try:
            while True:
                with self._cond:
                    while not self._closed and not self._pending:
                        if write_buffer:
                            batch = list(write_buffer)
                            write_buffer.clear()
                            self._cond.release()
                            try:
                                self._flush_batch(batch)
                            finally:
                                self._cond.acquire()
                            continue
                        self._cond.wait(timeout=1.0)
                    if self._closed:
                        break
                    file_id, mtime_ns, path = self._pending.popleft()
                    req_spectral = self._require_spectral
                    key = (file_id, mtime_ns)
                    self._pending_keys.discard(key)
                    self._in_flight.add(key)

                blob: bytes | None = None
                spec_blob: bytes | None = None
                from_db = False
                try:
                    entry = self._store.get_cached_thumbnail_entry(file_id, mtime_ns)
                    if entry is not None and (not req_spectral or entry[1] is not None):
                        blob, spec_blob = entry
                        from_db = True
                    elif path.exists():
                        blob, spec = compute_thumbnail_with_spectral(path)
                        spec_blob = spec or None
                    elif entry is not None:
                        blob, spec_blob = entry
                        from_db = True
                except Exception:
                    blob = None
                    spec_blob = None

                should_notify = False
                with self._cond:
                    self._in_flight.discard(key)
                    if blob is None:
                        self._mark_failed_locked(key)
                    else:
                        self._put_mem_cache_locked(key, (blob, spec_blob))
                    closed = self._closed
                    if blob is not None and not closed:
                        if not from_db:
                            write_buffer.append((file_id, mtime_ns, blob, spec_blob))
                        if file_id not in self._ready_ids:
                            self._ready_ids.add(file_id)
                            should_notify = len(self._ready_ids) == 1

                if should_notify:
                    try:
                        self.thumbnailsReady.emit()
                    except RuntimeError:
                        pass

                if len(write_buffer) >= 20:
                    batch = list(write_buffer)
                    write_buffer.clear()
                    self._flush_batch(batch)
        finally:
            if write_buffer:
                self._flush_batch(list(write_buffer))
                write_buffer.clear()

    def _flush_batch(self, batch: list[tuple[int, int, bytes, bytes | None]]) -> None:
        if not batch:
            return
        with self._db_write_lock:
            try:
                self._store.save_cached_thumbnail_entries(batch)
            except Exception:
                pass
