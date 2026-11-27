"""Generate multi-resolution Claude-Dark waveform icon (.ico and .png) for AudioSearchTool."""

from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
_BAR_HEIGHTS = (0.28, 0.50, 0.78, 0.96, 0.64, 0.88, 0.44, 0.24)


def render_icon_png(size: int) -> bytes:
    """Render a crisp anti-aliased Claude-Dark waveform badge PNG at ``size``x``size``."""
    img = QImage(size, size, QImage.Format.Format_RGBA8888)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pad = max(1.0, size * 0.06)
    rect = QRectF(pad, pad, size - 2.0 * pad, size - 2.0 * pad)
    radius = size * 0.22

    bg_path = QPainterPath()
    bg_path.addRoundedRect(rect, radius, radius)
    painter.fillPath(bg_path, QColor("#1B1A18"))

    border_width = max(1.0, size * 0.03)
    painter.setPen(QPen(QColor("#4A463F"), border_width))
    painter.drawRoundedRect(rect, radius, radius)

    # Symmetric warm terracotta waveform bars
    inner_w = rect.width() * 0.72
    inner_h = rect.height() * 0.58
    left = (size - inner_w) / 2.0
    cy = size / 2.0

    n_bars = len(_BAR_HEIGHTS)
    step = inner_w / n_bars
    bar_w = max(1.0, step * 0.58)
    offset_x = (step - bar_w) / 2.0

    for idx, amp in enumerate(_BAR_HEIGHTS):
        h = max(2.0, inner_h * amp)
        bx = left + idx * step + offset_x
        by = cy - h / 2.0
        color = QColor("#F0A387") if idx in (3, 5) else QColor("#D97757")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(
            QRectF(bx, by, bar_w, h),
            bar_w / 2.0,
            bar_w / 2.0,
        )

    painter.end()

    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def build_ico_bytes(frames: list[tuple[int, bytes]]) -> bytes:
    """Pack PNG-encoded square frames into a Windows multi-resolution .ico file."""
    count = len(frames)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries: list[bytes] = []
    payloads: list[bytes] = []
    for size, png_bytes in frames:
        dim = 0 if size >= 256 else size
        entries.append(
            struct.pack(
                "<BBBBHHII",
                dim,
                dim,
                0,
                0,
                1,
                32,
                len(png_bytes),
                offset,
            )
        )
        payloads.append(png_bytes)
        offset += len(png_bytes)
    return header + b"".join(entries) + b"".join(payloads)


def generate_icons(output_dir: Path | None = None) -> tuple[Path, Path]:
    app = QGuiApplication.instance() or QGuiApplication([])
    _ = app

    root = Path(__file__).resolve().parents[1]
    out_dir = output_dir or (root / "src" / "audiosearch" / "browser" / "resources")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [(sz, render_icon_png(sz)) for sz in ICON_SIZES]
    # Prefer the finalized V1 owl source PNG when present in resources
    master_png = out_dir / "whistle.png"
    if master_png.is_file():
        src_img = QImage(str(master_png))
        if not src_img.isNull():
            frames = []
            for sz in ICON_SIZES:
                scaled = src_img.scaled(
                    sz,
                    sz,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                scaled.save(buf, "PNG")
                frames.append((sz, bytes(buf.data())))
    ico_path = out_dir / "audiosearch.ico"
    png_path = out_dir / "audiosearch.png"

    ico_path.write_bytes(build_ico_bytes(frames))
    png_path.write_bytes(frames[-1][1])
    return ico_path, png_path


if __name__ == "__main__":
    ico, png = generate_icons()
    print(f"Generated {ico} ({ico.stat().st_size} bytes) and {png} ({png.stat().st_size} bytes)")
