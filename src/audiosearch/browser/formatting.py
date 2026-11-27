"""Human-readable formatting helpers for table cells and waveform rulers."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0.00s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, rem = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}:{rem:05.2f}"
    hours, mins = divmod(int(mins), 60)
    return f"{hours}:{mins:02d}:{rem:05.2f}"


def format_sample_rate(rate_hz: int) -> str:
    if rate_hz <= 0:
        return "—"
    if rate_hz % 1000 == 0:
        return f"{rate_hz // 1000}k"
    return f"{rate_hz / 1000:.1f}k"


def format_channels(channels: int) -> str:
    if channels <= 0:
        return "—"
    if channels == 1:
        return "1ch"
    if channels == 2:
        return "2ch"
    return f"{channels}ch"


def format_bit_depth(bit_depth: int) -> str:
    if bit_depth <= 0:
        return "—"
    return f"{bit_depth}bit"


def format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_axis_time(seconds: float, span: float) -> str:
    seconds = max(0.0, seconds)
    mins, rem = divmod(seconds, 60.0)
    if span < 10.0:
        return f"{int(mins)}:{rem:06.3f}"
    if span < 60.0:
        return f"{int(mins)}:{rem:05.2f}"
    return f"{int(mins)}:{rem:02.0f}"
