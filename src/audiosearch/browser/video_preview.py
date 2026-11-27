"""Inspector-embedded video preview canvas, PyAV frame worker, and optional pop-out viewer."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audiosearch.audioio.video import VideoProbeInfo, frame_to_rgb24, probe_video


def _make_popout_icon() -> QIcon:
    logical = 14
    scale = 2
    px = logical * scale
    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, QColor("#D5CFC4")),
        (QIcon.Mode.Active, QColor("#FFFFFF")),
    ):
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(float(scale))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(
            color, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
        )
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(6.2, 2.5)
        path.lineTo(2.5, 2.5)
        path.lineTo(2.5, 11.5)
        path.lineTo(11.5, 11.5)
        path.lineTo(11.5, 7.8)
        p.drawPath(path)
        p.drawLine(QPointF(6.8, 7.2), QPointF(11.6, 2.4))
        p.drawPolyline(QPolygonF([QPointF(8.3, 2.4), QPointF(11.6, 2.4), QPointF(11.6, 5.7)]))
        p.end()
        icon.addPixmap(pm, mode)
    return icon


class VideoFrameWorker(QObject):
    """Single background worker thread decoding coalesced preview frames via PyAV."""

    videoLoaded = Signal(int, object)  # (generation, VideoProbeInfo)
    frameReady = Signal(int, float, object)  # (generation, pts_sec, QImage)
    videoFailed = Signal(int, str)  # (generation, error_message)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cond = threading.Condition()
        self._gen = 0
        self._load_request: tuple[int, Path, int, int] | None = None
        self._pending_time: tuple[int, float] | None = None
        self._active = True
        self._stop = False
        self._max_w = 520
        self._max_h = 360
        self._thread: threading.Thread | None = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="WhistleVideoDecoder", daemon=True
            )
            self._thread.start()

    def set_max_size(self, max_w: int, max_h: int) -> None:
        with self._cond:
            self._max_w = max(120, int(max_w))
            self._max_h = max(90, int(max_h))

    def set_active(self, active: bool) -> None:
        with self._cond:
            self._active = bool(active)
            if self._active:
                self._cond.notify()

    def load_video(self, path: Path, *, max_w: int = 520, max_h: int = 360) -> int:
        with self._cond:
            if self._stop:
                return self._gen
            self._gen += 1
            self._max_w = max(120, int(max_w))
            self._max_h = max(90, int(max_h))
            self._pending_time = None
            self._load_request = (self._gen, Path(path), self._max_w, self._max_h)
            self._ensure_thread_locked()
            self._cond.notify()
            return self._gen

    def request_time(self, generation: int, seconds: float) -> None:
        with self._cond:
            if self._stop or generation != self._gen:
                return
            self._pending_time = (generation, max(0.0, float(seconds)))
            if self._active:
                self._cond.notify()

    def clear(self) -> None:
        with self._cond:
            self._gen += 1
            self._load_request = None
            self._pending_time = None
            self._cond.notify()

    def shutdown(self, timeout: float = 1.5) -> None:
        thread: threading.Thread | None
        with self._cond:
            thread = self._thread
            self._gen += 1
            self._load_request = None
            self._pending_time = None
            self._stop = True
            self._cond.notify_all()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _emit_safe(self, sig: Any, *args: Any) -> None:
        try:
            sig.emit(*args)
        except RuntimeError:
            pass

    @staticmethod
    def _frame_to_qimage(frame: Any, max_w: int, max_h: int) -> QImage:
        rgb_bytes, out_w, out_h = frame_to_rgb24(frame, max_w, max_h)
        # `.copy()` detaches memory from `rgb_bytes` so QImage owns its pixel buffer safely.
        return QImage(rgb_bytes, out_w, out_h, out_w * 3, QImage.Format.Format_RGB888).copy()

    def _run(self) -> None:
        container: Any = None
        vstream: Any = None
        packet_iter: Any = None
        open_gen = -1
        last_pts: float | None = None
        at_eof = False
        av_mod: Any = None

        def _close_local() -> None:
            nonlocal container, vstream, packet_iter, open_gen, last_pts, at_eof
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            container = None
            vstream = None
            packet_iter = None
            open_gen = -1
            last_pts = None
            at_eof = False

        def _should_abort(expected_gen: int) -> bool:
            with self._cond:
                return self._stop or self._gen != expected_gen or self._load_request is not None

        def _decode_until(target_sec: float, expected_gen: int, max_w: int, max_h: int) -> None:
            nonlocal packet_iter, last_pts, at_eof
            if container is None or vstream is None or av_mod is None:
                return

            tb = getattr(vstream, "time_base", None)
            if last_pts is not None and abs(target_sec - last_pts) < 0.012:
                return
            if at_eof and last_pts is not None and target_sec >= last_pts:
                return

            need_seek = (
                last_pts is None
                or target_sec < (last_pts - 0.04)
                or (target_sec - last_pts) > 1.2
            )
            if need_seek:
                at_eof = False
                last_pts = None
                try:
                    container.seek(
                        max(0, int(target_sec * float(av_mod.time_base))),
                        stream=vstream,
                        any_frame=False,
                        backward=True,
                    )
                    try:
                        vstream.codec_context.flush_buffers()
                    except Exception:
                        pass
                    packet_iter = container.demux(vstream)
                except Exception as exc:
                    self._emit_safe(self.videoFailed, expected_gen, str(exc))
                    return

            if packet_iter is None:
                packet_iter = container.demux(vstream)

            best_frame: Any = None
            best_pts: float = 0.0
            reached_target = False
            while not reached_target:
                if _should_abort(expected_gen):
                    return
                # If a newer backward jump superseded this decode, yield immediately.
                with self._cond:
                    if (
                        self._pending_time is not None
                        and self._pending_time[0] == expected_gen
                        and self._pending_time[1] < target_sec - 0.05
                    ):
                        return
                try:
                    packet = next(packet_iter)
                except StopIteration:
                    at_eof = True
                    break
                if getattr(packet, "dts", None) is None and getattr(packet, "size", 1) == 0:
                    continue
                try:
                    frames = packet.decode()
                except Exception:
                    continue
                for frame in frames:
                    pts_val = getattr(frame, "pts", None)
                    pts_sec = float(pts_val * tb) if (pts_val is not None and tb is not None) else 0.0
                    best_frame = frame
                    best_pts = pts_sec
                    if pts_sec >= target_sec - 0.015:
                        reached_target = True
                        break

            if best_frame is not None and not _should_abort(expected_gen):
                last_pts = best_pts
                try:
                    qimg = self._frame_to_qimage(best_frame, max_w, max_h)
                    self._emit_safe(self.frameReady, expected_gen, best_pts, qimg)
                except Exception as exc:
                    self._emit_safe(self.videoFailed, expected_gen, str(exc))

        try:
            while True:
                with self._cond:
                    while not self._stop:
                        if self._gen != open_gen and open_gen != -1 and self._load_request is None:
                            break
                        if self._load_request is not None:
                            break
                        if (
                            self._active
                            and self._pending_time is not None
                            and self._pending_time[0] == open_gen
                        ):
                            break
                        self._cond.wait()

                    if self._stop:
                        break

                    if self._gen != open_gen and open_gen != -1 and self._load_request is None:
                        drop_open = True
                    else:
                        drop_open = False

                    load_req = self._load_request
                    if load_req is not None:
                        self._load_request = None
                        time_req = None
                    elif self._active and self._pending_time is not None:
                        time_req = self._pending_time
                        self._pending_time = None
                    else:
                        time_req = None
                    max_w, max_h = self._max_w, self._max_h

                if drop_open and load_req is None:
                    _close_local()
                    continue

                if load_req is not None:
                    _close_local()
                    req_gen, path, req_w, req_h = load_req
                    if _should_abort(req_gen):
                        continue
                    try:
                        import av

                        av_mod = av
                        info: VideoProbeInfo = probe_video(path)
                        if _should_abort(req_gen):
                            continue
                        self._emit_safe(self.videoLoaded, req_gen, info)
                        container = av.open(str(path), mode="r")
                        if not container.streams.video:
                            self._emit_safe(self.videoFailed, req_gen, "视频文件不包含可用画面流")
                            _close_local()
                            continue
                        vstream = container.streams.video[0]
                        open_gen = req_gen
                        last_pts = None
                        at_eof = False
                        packet_iter = container.demux(vstream)
                        _decode_until(0.0, req_gen, req_w, req_h)
                    except Exception as exc:
                        _close_local()
                        self._emit_safe(self.videoFailed, req_gen, str(exc))
                    continue

                if time_req is not None:
                    t_gen, target_sec = time_req
                    if t_gen == open_gen and not _should_abort(t_gen):
                        _decode_until(target_sec, t_gen, max_w, max_h)
        finally:
            _close_local()


class VideoCanvasWidget(QWidget):
    """Aspect-preserving video surface for both inline Inspector preview and pop-out viewing."""

    togglePlayRequested = Signal()
    popOutRequested = Signal()

    def __init__(self, *, show_popout_btn: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_popout_btn = bool(show_popout_btn)
        self._image: QImage | None = None
        self._meta_text = ""
        self._status_text = "加载视频中…"
        self._playing = False
        self._hovered = False

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.popout_btn = QPushButton("", self)
        self.popout_btn.setFixedSize(26, 26)
        self.popout_btn.setIcon(_make_popout_icon())
        self.popout_btn.setIconSize(QSize(14, 14))
        self.popout_btn.setToolTip("独立大窗预览视频（双击画面也可弹出）")
        self.popout_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.popout_btn.setStyleSheet(
            "QPushButton { background-color: rgba(20, 19, 17, 185); "
            "border: 1px solid rgba(82, 78, 71, 200); border-radius: 6px; padding: 0px; }"
            "QPushButton:hover { background-color: rgba(56, 40, 34, 230); "
            "border-color: #D97757; }"
        )
        self.popout_btn.setVisible(self._show_popout_btn)
        self.popout_btn.clicked.connect(lambda _c=False: self.popOutRequested.emit())

    def has_image(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def current_image(self) -> QImage | None:
        return self._image

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        if image is not None and not image.isNull():
            self._status_text = ""
        self.update()

    def set_meta_text(self, text: str) -> None:
        self._meta_text = text.strip()
        self.update()

    def set_status_text(self, text: str) -> None:
        self._status_text = text.strip()
        self.update()

    def set_playing(self, playing: bool) -> None:
        is_p = bool(playing)
        if self._playing != is_p:
            self._playing = is_p
            self.update()

    def clear(self, message: str = "未加载视频") -> None:
        self._image = None
        self._meta_text = ""
        self._playing = False
        self._status_text = message
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._show_popout_btn:
            self.popout_btn.move(max(6, self.width() - self.popout_btn.width() - 6), 6)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.togglePlayRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._show_popout_btn:
            self.popOutRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            outer = QRectF(0.5, 0.5, max(2.0, self.width() - 1.0), max(2.0, self.height() - 1.0))
            clip = QPainterPath()
            clip.addRoundedRect(outer, 9.0, 9.0)
            p.save()
            p.setClipPath(clip)
            p.fillRect(outer, QColor("#141311"))

            if self._image is not None and not self._image.isNull():
                inner = outer.adjusted(3.0, 3.0, -3.0, -3.0)
                iw = max(1.0, float(self._image.width()))
                ih = max(1.0, float(self._image.height()))
                scale = min(inner.width() / iw, inner.height() / ih)
                dw = iw * scale
                dh = ih * scale
                dx = inner.left() + (inner.width() - dw) * 0.5
                dy = inner.top() + (inner.height() - dh) * 0.5
                p.drawImage(QRectF(dx, dy, dw, dh), self._image)
            elif self._status_text:
                p.setPen(QColor("#8F8A80"))
                p.drawText(outer.adjusted(10, 10, -10, -10), Qt.AlignmentFlag.AlignCenter, self._status_text)

            if self._hovered and not self._playing and self.has_image():
                cx, cy = outer.center().x(), outer.center().y()
                p.setPen(QPen(QColor(236, 233, 226, 175), 1.3))
                p.setBrush(QColor(20, 19, 17, 170))
                p.drawEllipse(QPointF(cx, cy), 17.0, 17.0)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#ECE9E2"))
                p.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(cx - 4.5, cy - 7.0),
                            QPointF(cx + 8.0, cy),
                            QPointF(cx - 4.5, cy + 7.0),
                        ]
                    )
                )

            if self._meta_text:
                f = QFont(self.font())
                f.setPixelSize(11)
                p.setFont(f)
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(self._meta_text) + 12
                th = max(18, fm.height() + 4)
                pill = QRectF(7.0, outer.bottom() - th - 6.0, min(outer.width() - 14.0, float(tw)), float(th))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(18, 17, 15, 195))
                p.drawRoundedRect(pill, 5.0, 5.0)
                p.setPen(QColor("#D5CFC4"))
                p.drawText(pill, Qt.AlignmentFlag.AlignCenter, self._meta_text)
            p.restore()

            p.setPen(QPen(QColor("#34322E"), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(outer, 9.0, 9.0)
        finally:
            p.end()


class VideoPopoutDialog(QDialog):
    """Resizable non-modal companion window for full-size video audition."""

    togglePlayRequested = Signal()
    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("视频预览")
        self.resize(960, 580)
        self.setMinimumSize(480, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.canvas = VideoCanvasWidget(show_popout_btn=False, parent=self)
        self.canvas.togglePlayRequested.connect(self.togglePlayRequested.emit)
        layout.addWidget(self.canvas, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 0)
        footer.setSpacing(8)
        self.info_label = QLabel("")
        self.info_label.setProperty("role", "muted")
        self.play_btn = QPushButton("播放 / 暂停 (Space)")
        self.play_btn.setProperty("variant", "secondary")
        self.play_btn.clicked.connect(lambda _c=False: self.togglePlayRequested.emit())
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.clicked.connect(self.close)
        footer.addWidget(self.info_label, 1)
        footer.addWidget(self.play_btn)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

    def set_header_info(self, title: str, meta_text: str) -> None:
        self.setWindowTitle(f"视频预览 - {title}" if title else "视频预览")
        self.info_label.setText(f"{title}  ·  {meta_text}" if meta_text else title)
        self.canvas.set_meta_text(meta_text)

    def set_playing(self, playing: bool) -> None:
        self.canvas.set_playing(playing)
        self.play_btn.setText("暂停 (Space)" if playing else "播放 (Space)")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.togglePlayRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["VideoCanvasWidget", "VideoFrameWorker", "VideoPopoutDialog"]
