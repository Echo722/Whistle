"""Persistent UI & audition preferences saved alongside the library database."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from audiosearch.browser.theme import normalize_ui_scale


@dataclass
class BrowserSettings:
    autoplay: bool = True
    loop: bool = False
    volume_db: float = 0.0
    preserve_pitch: bool = False
    reset_on_switch: bool = True
    bake_on_drag: bool = True
    inspector_visible: bool = True
    sort_column: str = "filename"
    sort_asc: bool = True
    window_width: int = 1360
    window_height: int = 840
    window_maximized: bool = False
    h_splitter_sizes: list[int] = field(default_factory=lambda: [230, 820, 300])
    v_splitter_sizes: list[int] = field(default_factory=lambda: [540, 200])
    column_widths: dict[str, int] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    hidden_columns: list[str] = field(default_factory=list)
    ai_column_widths: dict[str, int] = field(default_factory=dict)
    ai_column_order: list[str] = field(default_factory=list)
    ai_hidden_columns: list[str] = field(default_factory=list)
    ai_result_limit: int = 500
    ai_profile: str = "large"
    exclude_same_package_similar: bool = False
    spectral_peaks: bool = False
    hide_offline: bool = False
    filter_bar_visible: bool = False
    global_scope: bool = False
    local_roots_expanded: bool = False
    collapsed_collections: list[int] = field(default_factory=list)
    sidebar_splitter_sizes: list[int] = field(default_factory=lambda: [210, 330])
    ui_scale: str = "standard"
    output_device_name: str = ""


def settings_path_for_db(db_path: Path) -> Path:
    return db_path.parent / "browser-settings.json"


def _int_id_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    out: set[int] = set()
    for item in value:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _clean_col_widths(raw_val: object) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(raw_val, dict):
        for k, v in raw_val.items():
            if isinstance(v, (int, float)) and int(v) >= 24:
                out[str(k)] = int(v)
    return out


def _clean_col_list(raw_val: object, *, exclude: tuple[str, ...] = ()) -> list[str]:
    if not isinstance(raw_val, list):
        return []
    excluded = set(exclude)
    return [str(x) for x in raw_val if str(x) not in excluded]


def load_settings(path: Path) -> BrowserSettings:
    if not path.exists():
        return BrowserSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return BrowserSettings()
        defaults = BrowserSettings()
        column_widths = _clean_col_widths(raw.get("column_widths"))
        ai_column_widths = _clean_col_widths(raw.get("ai_column_widths"))
        try:
            raw_limit = int(raw.get("ai_result_limit", defaults.ai_result_limit))
        except (TypeError, ValueError):
            raw_limit = defaults.ai_result_limit
        ai_result_limit = max(50, min(5000, raw_limit))
        return BrowserSettings(
            autoplay=bool(raw.get("autoplay", defaults.autoplay)),
            loop=bool(raw.get("loop", defaults.loop)),
            volume_db=float(raw.get("volume_db", defaults.volume_db)),
            preserve_pitch=bool(raw.get("preserve_pitch", defaults.preserve_pitch)),
            reset_on_switch=bool(raw.get("reset_on_switch", defaults.reset_on_switch)),
            bake_on_drag=bool(raw.get("bake_on_drag", defaults.bake_on_drag)),
            inspector_visible=bool(raw.get("inspector_visible", defaults.inspector_visible)),
            sort_column=str(raw.get("sort_column", defaults.sort_column)),
            sort_asc=bool(raw.get("sort_asc", defaults.sort_asc)),
            window_width=max(640, int(raw.get("window_width", defaults.window_width))),
            window_height=max(480, int(raw.get("window_height", defaults.window_height))),
            window_maximized=bool(raw.get("window_maximized", defaults.window_maximized)),
            h_splitter_sizes=_int_list(raw.get("h_splitter_sizes"), defaults.h_splitter_sizes, 3),
            v_splitter_sizes=_int_list(raw.get("v_splitter_sizes"), defaults.v_splitter_sizes, 2),
            column_widths=column_widths,
            column_order=_clean_col_list(raw.get("column_order")),
            hidden_columns=_clean_col_list(raw.get("hidden_columns"), exclude=("filename",)),
            ai_column_widths=ai_column_widths,
            ai_column_order=_clean_col_list(raw.get("ai_column_order")),
            ai_hidden_columns=_clean_col_list(raw.get("ai_hidden_columns"), exclude=("filename",)),
            ai_result_limit=ai_result_limit,
            ai_profile=str(raw.get("ai_profile", defaults.ai_profile)),
            exclude_same_package_similar=bool(
                raw.get("exclude_same_package_similar", defaults.exclude_same_package_similar)
            ),
            spectral_peaks=bool(raw.get("spectral_peaks", defaults.spectral_peaks)),
            hide_offline=bool(raw.get("hide_offline", defaults.hide_offline)),
            filter_bar_visible=bool(raw.get("filter_bar_visible", defaults.filter_bar_visible)),
            global_scope=bool(raw.get("global_scope", defaults.global_scope)),
            local_roots_expanded=bool(
                raw.get("local_roots_expanded", defaults.local_roots_expanded)
            ),
            collapsed_collections=_int_id_list(raw.get("collapsed_collections")),
            sidebar_splitter_sizes=_int_list(
                raw.get("sidebar_splitter_sizes"), defaults.sidebar_splitter_sizes, 2
            ),
            ui_scale=normalize_ui_scale(raw.get("ui_scale", defaults.ui_scale)),
            output_device_name=str(raw.get("output_device_name", "") or "").strip(),
        )
    except Exception:
        return BrowserSettings()


def save_settings(path: Path, settings: BrowserSettings) -> None:
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(settings), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _int_list(value: object, fallback: list[int], expected_len: int) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_len:
        return list(fallback)
    try:
        return [max(20, int(v)) for v in value]
    except (TypeError, ValueError):
        return list(fallback)
