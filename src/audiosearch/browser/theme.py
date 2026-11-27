"""Claude Dark warm charcoal palette, typography, and stylesheet for the audio browser."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from audiosearch.library.peaks import spectral_color_rgb

CANVAS = QColor("#242320")
SIDEBAR = QColor("#1B1A18")
PANEL = QColor("#1F1E1B")
RAISED = QColor("#2B2A26")
LINE = QColor("#34322E")
BORDER_SOFT = QColor("#3D3A34")

TEXT = QColor("#ECE9E2")
TEXT_DIM = QColor("#A5A096")
TEXT_FAINT = QColor("#736E66")

SELECT = QColor("#47332A")
SELECT_INACTIVE = QColor("#382D28")
SELECT_BORDER = QColor("#8F533E")
SELECT_BORDER_INACTIVE = QColor("#5C3F34")
SELECT_MARK = QColor("#D97757")
SELECT_MARK_INACTIVE = QColor("#9E5C45")
PLAY = QColor("#D97757")
PLAY_SELECTED = QColor("#FF9E7A")
ACCENT_SOFT = QColor("#382822")
ACCENT_TEXT = QColor("#F0A387")
OFFLINE = QColor("#5E5A53")

ALTERNATE_ROW = QColor("#201F1C")

UI_FONT_FAMILY = "Segoe UI, Microsoft YaHei UI"
MONO_FONT_FAMILY = "Cascadia Mono, Consolas"

DEFAULT_UI_SCALE = "standard"


@dataclass(frozen=True, slots=True)
class UiScaleSpec:
    """Responsive density and typography scale preset for the browser UI."""

    key: str
    label: str
    description: str
    scale: float
    base_px: int
    small_px: int
    xs_px: int
    mono_pt: int
    table_row_h: int
    tree_row_h: int
    tree_divider_h: int
    tree_indent: int
    tree_left_pad: int
    input_h: int
    btn_h: int
    knob_w: int
    knob_h: int
    inspector_min_w: int
    inspector_max_w: int
    inspector_key_w: int
    inspector_note_h: int
    cover_size: int


UI_SCALE_PRESETS: dict[str, UiScaleSpec] = {
    "compact": UiScaleSpec(
        key="compact",
        label="紧凑",
        description="12px · 高信息密度（适合小屏）",
        scale=0.92,
        base_px=12,
        small_px=11,
        xs_px=10,
        mono_pt=8,
        table_row_h=25,
        tree_row_h=22,
        tree_divider_h=23,
        tree_indent=17,
        tree_left_pad=10,
        input_h=30,
        btn_h=26,
        knob_w=96,
        knob_h=26,
        inspector_min_w=250,
        inspector_max_w=380,
        inspector_key_w=44,
        inspector_note_h=58,
        cover_size=184,
    ),
    "standard": UiScaleSpec(
        key="standard",
        label="标准（默认）",
        description="13px · 当前默认观感",
        scale=1.00,
        base_px=13,
        small_px=12,
        xs_px=11,
        mono_pt=9,
        table_row_h=28,
        tree_row_h=24,
        tree_divider_h=25,
        tree_indent=18,
        tree_left_pad=10,
        input_h=34,
        btn_h=28,
        knob_w=102,
        knob_h=28,
        inspector_min_w=260,
        inspector_max_w=400,
        inspector_key_w=48,
        inspector_note_h=64,
        cover_size=200,
    ),
    "large": UiScaleSpec(
        key="large",
        label="较大",
        description="15px · 舒适清晰（适合高分屏）",
        scale=1.15,
        base_px=15,
        small_px=14,
        xs_px=12,
        mono_pt=10,
        table_row_h=33,
        tree_row_h=28,
        tree_divider_h=29,
        tree_indent=21,
        tree_left_pad=12,
        input_h=38,
        btn_h=32,
        knob_w=114,
        knob_h=32,
        inspector_min_w=285,
        inspector_max_w=440,
        inspector_key_w=56,
        inspector_note_h=74,
        cover_size=200,
    ),
    "xlarge": UiScaleSpec(
        key="xlarge",
        label="特大",
        description="16px · 远视距 / 4K 易读",
        scale=1.25,
        base_px=16,
        small_px=15,
        xs_px=13,
        mono_pt=11,
        table_row_h=36,
        tree_row_h=31,
        tree_divider_h=32,
        tree_indent=23,
        tree_left_pad=12,
        input_h=40,
        btn_h=34,
        knob_w=122,
        knob_h=34,
        inspector_min_w=305,
        inspector_max_w=470,
        inspector_key_w=62,
        inspector_note_h=82,
        cover_size=200,
    ),
}

_ACTIVE_UI_SCALE = DEFAULT_UI_SCALE


def normalize_ui_scale(key: str | None) -> str:
    val = str(key or "").strip().lower()
    return val if val in UI_SCALE_PRESETS else DEFAULT_UI_SCALE


def current_ui_scale() -> str:
    return _ACTIVE_UI_SCALE


def ui_scale_spec(key: str | None = None) -> UiScaleSpec:
    resolved = _ACTIVE_UI_SCALE if key is None else normalize_ui_scale(key)
    return UI_SCALE_PRESETS[resolved]


def derived_font(base: QFont, delta_px: int = 0, *, min_px: int = 9) -> QFont:
    """Return a resized copy of ``base`` safely handling QSS pixel-sized fonts."""
    out = QFont(base)
    px = out.pixelSize()
    if px > 0:
        out.setPixelSize(max(min_px, px + delta_px))
        return out
    pt = out.pointSizeF()
    if pt > 0:
        min_pt = max(7.0, min_px * 0.75)
        out.setPointSizeF(max(min_pt, pt + delta_px * 0.75))
        return out
    out.setPixelSize(max(min_px, UI_SCALE_PRESETS[_ACTIVE_UI_SCALE].base_px + delta_px))
    return out


_STYLESHEET_TEMPLATE = """
QMainWindow, QWidget {
    background-color: #1B1A18;
    color: #ECE9E2;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: __BASE_PX__px;
}
QFrame#titleBar {
    background-color: #161513;
    border-bottom: 1px solid #2A2824;
}
QFrame#topBar {
    background-color: #1B1A18;
    border-bottom: 1px solid #34322E;
}
QFrame#filterBar {
    background-color: #1F1E1B;
    border-bottom: 1px solid #34322E;
}
QFrame#bottomDock {
    background-color: #1B1A18;
    border-top: 1px solid #34322E;
}
QFrame#waveformFrame {
    background-color: #1D1C19;
    border: 1px solid #34322E;
    border-radius: 7px;
    padding: 2px;
}
QFrame#panel, QWidget#panel, QFrame#inspectorPanel {
    background-color: #1F1E1B;
    border: none;
}
QFrame#inspectorInfoCard {
    background-color: #181715;
    border: 1px solid #2D2B27;
    border-radius: 8px;
}
QFrame#coverCard, QLabel#coverCard {
    background-color: #1B1A18;
    border: 1px solid #34322E;
    border-radius: 10px;
    color: #736E66;
}
QFrame#segTrack {
    background-color: #151412;
    border: 1px solid #34322E;
    border-radius: 8px;
}
QFrame#heroModeTrack {
    background-color: #141311;
    border: 1px solid #4D4942;
    border-radius: 9px;
}
QWidget#aiBanner, QWidget#dupBanner {
    background-color: #1F1E1B;
    border-top: 1px solid #34322E;
}
QLabel[role="muted"] {
    color: #A5A096;
    font-size: __SMALL_PX__px;
    background: transparent;
}
QLabel[role="inspectorTitle"] {
    font-size: __BASE_PX__px;
    font-weight: 600;
    color: #ECE9E2;
    background: transparent;
}
QLabel[role="inspectorKey"] {
    color: #8F8A80;
    font-size: __XS_PX__px;
    background: transparent;
}
QLabel[role="inspectorBreadcrumb"] {
    font-size: __BREADCRUMB_PX__px;
    line-height: 1.45;
    background: transparent;
}
QLabel[role="inspectorPath"] {
    color: #8F8A80;
    font-size: __XS_PX__px;
    background: transparent;
}
QLabel[role="inspectorBrand"] {
    color: #E89274;
    font-size: __SMALL_PX__px;
    font-weight: 600;
    background: transparent;
}
QLabel[role="inspectorPackage"] {
    color: #F2C2AE;
    font-size: __SMALL_PX__px;
    font-weight: 600;
    background: transparent;
}
QLabel[role="inspectorUcs"] {
    color: #E6B87A;
    font-size: __UCS_PX__px;
    background: transparent;
}
QLabel[role="inspectorDetail"] {
    background: transparent;
    color: #A5A096;
    font-size: __SMALL_PX__px;
}
QLineEdit {
    background-color: #181715;
    border: 1px solid #3D3A34;
    border-radius: 7px;
    padding: 4px 9px;
    color: #ECE9E2;
    selection-background-color: #382822;
    selection-color: #F3F0EA;
}
QLineEdit:focus {
    border: 1px solid #D97757;
}
QSpinBox {
    background-color: #181715;
    border: 1px solid #3D3A34;
    border-radius: 7px;
    padding: 4px 8px;
    color: #ECE9E2;
}
QSpinBox:focus {
    border: 1px solid #D97757;
}
QLineEdit#searchInput {
    background-color: #161513;
    border: 1px solid #3D3A34;
    border-radius: 9px;
    padding: 5px 11px;
    color: #ECE9E2;
    font-size: __BASE_PX__px;
}
QLineEdit#searchInput:focus {
    border: 1px solid #D97757;
}
QLineEdit#searchInput[aiMode="true"] {
    background-color: #1F1916;
    border: 1px solid #9E553D;
}
QLineEdit#searchInput[aiMode="true"]:focus {
    border: 1px solid #D97757;
}
QLineEdit#pathEdit {
    background-color: #21201D;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 4px 10px;
    color: #A5A096;
    font-size: __SMALL_PX__px;
}
QLineEdit#pathEdit:focus {
    border: 1px solid #3D3A34;
}
QPlainTextEdit, QPlainTextEdit#noteEdit {
    background-color: #181715;
    border: 1px solid #3D3A34;
    border-radius: 8px;
    padding: 6px 8px;
    color: #ECE9E2;
    font-size: __SMALL_PX__px;
    selection-background-color: #382822;
    selection-color: #F3F0EA;
}
QPlainTextEdit:focus, QPlainTextEdit#noteEdit:focus {
    border: 1px solid #D97757;
}
QPushButton {
    background-color: #2B2A26;
    border: 1px solid #3D3A34;
    border-radius: 7px;
    padding: 4px 10px;
    color: #ECE9E2;
}
QPushButton:hover {
    background-color: #35332E;
    border-color: #524E47;
}
QPushButton:pressed {
    background-color: #252420;
}
QPushButton:checked {
    background-color: #382822;
    border-color: #8A4C38;
    color: #F0A387;
}
QPushButton:disabled {
    color: #5E5A53;
    border-color: #2D2B27;
}
QPushButton[variant="primary"] {
    background-color: #D97757;
    border: 1px solid #D97757;
    color: #191614;
    font-weight: 600;
}
QPushButton[variant="primary"]:hover {
    background-color: #E38566;
    border-color: #E38566;
}
QPushButton[variant="primary"]:pressed {
    background-color: #C46547;
    border-color: #C46547;
}
QPushButton[variant="primary"]:disabled {
    background-color: #382822;
    border-color: #4D342A;
    color: #7D6A61;
}
QPushButton[variant="secondary"] {
    background-color: #2B2A26;
    border: 1px solid #4A463F;
    color: #ECE9E2;
}
QPushButton[variant="secondary"]:hover {
    background-color: #35332E;
    border-color: #5E5950;
}
QPushButton[variant="secondary"]:pressed {
    background-color: #252420;
}
QPushButton[variant="ghost"] {
    background-color: transparent;
    border: 1px solid transparent;
    color: #A5A096;
}
QPushButton[variant="ghost"]:hover {
    background-color: #2E2D29;
    color: #ECE9E2;
}
QPushButton[variant="ghost"]:checked {
    background-color: #382822;
    border-color: #8A4C38;
    color: #F0A387;
}
QPushButton[variant="ghost"]:disabled {
    color: #5C5851;
    background-color: transparent;
}
QPushButton[variant="statusLink"] {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 1px 8px;
    color: #A5A096;
    font-size: __SMALL_PX__px;
}
QPushButton[variant="statusLink"]:hover {
    background-color: #2D2C28;
    color: #ECE9E2;
}
QPushButton[variant="statusLink"][statusState="ok"] {
    color: #7EC89B;
}
QPushButton[variant="statusLink"][statusState="ok"]:hover {
    background-color: #2D2C28;
    color: #97DBA9;
}
QPushButton[variant="statusLink"][statusState="error"] {
    color: #E38566;
}
QPushButton[variant="statusLink"][statusState="error"]:hover {
    background-color: #2D2C28;
    color: #F2A58B;
}
QPushButton[variant="chip"] {
    background-color: transparent;
    border: 1px solid #3D3A34;
    color: #A5A096;
    padding: 3px 9px;
}
QPushButton[variant="chip"]:hover {
    background-color: #2E2D29;
    border-color: #524E47;
    color: #ECE9E2;
}
QPushButton[variant="chip"]:checked {
    background-color: #382822;
    border-color: #8A4C38;
    color: #F0A387;
}
QPushButton[variant="seg"] {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 11px;
    color: #A5A096;
}
QPushButton[variant="seg"]:hover {
    color: #ECE9E2;
}
QPushButton[variant="seg"]:checked {
    background-color: #2E2C28;
    border-color: #423F39;
    color: #ECE9E2;
    font-weight: 600;
}
QPushButton[variant="modeKw"],
QPushButton[variant="modeAi"] {
    background-color: transparent;
    border: none;
    border-radius: 7px;
    padding: 0px 13px;
    min-height: __MODE_BTN_H__px;
    max-height: __MODE_BTN_H__px;
    color: #B8B2A7;
    font-size: __SMALL_PX__px;
    font-weight: 500;
}
QPushButton[variant="modeKw"]:hover,
QPushButton[variant="modeAi"]:hover {
    color: #ECE9E2;
    background-color: #24221F;
}
QPushButton[variant="modeKw"]:checked {
    background-color: #ECE9E2;
    border: none;
    color: #191614;
    font-weight: 600;
}
QPushButton[variant="modeAi"]:checked {
    background-color: #D97757;
    border: none;
    color: #191614;
    font-weight: 600;
}
QPushButton[variant="transportPlay"] {
    background-color: #2B2A26;
    border: 1px solid #4A463F;
    border-radius: 7px;
    padding: 0px;
}
QPushButton[variant="transportPlay"]:hover {
    background-color: #35332E;
    border-color: #5E5950;
}
QPushButton[variant="transportPlay"]:pressed {
    background-color: #252420;
}
QPushButton[variant="transportPlay"][playing="true"] {
    background-color: #382822;
    border: 1px solid #9E553D;
}
QPushButton[variant="transportPlay"][playing="true"]:hover {
    background-color: #463129;
    border-color: #B86449;
}
QPushButton[variant="winCtrl"],
QPushButton[variant="winClose"] {
    background-color: transparent;
    border: none;
    border-radius: 0px;
    padding: 0px;
}
QPushButton[variant="winCtrl"]:hover {
    background-color: #2A2824;
}
QPushButton[variant="winCtrl"]:pressed {
    background-color: #34322E;
}
QPushButton[variant="winClose"]:hover {
    background-color: #C42B1C;
}
QPushButton[variant="winClose"]:pressed {
    background-color: #A62316;
}
QPushButton[iconBtn="true"] {
    padding: 0px;
    font-size: __SMALL_PX__px;
}
QPushButton[compact="true"] {
    padding: 2px 8px;
    font-size: __SMALL_PX__px;
}
QPushButton[variant="cardAction"] {
    background-color: #23211E;
    border: 1px solid #35332E;
    border-radius: 5px;
    padding: 2px 6px;
    min-height: __CARD_BTN_H__px;
    color: #C8C3B9;
    font-size: __XS_PX__px;
}
QPushButton[variant="cardAction"]:hover {
    background-color: #2E2C28;
    border-color: #524E47;
    color: #F3F0EA;
}
QPushButton[variant="cardAction"]:pressed {
    background-color: #1C1B18;
    border-color: #423F39;
}
QPushButton[variant="cardAction"]:disabled {
    background-color: #1C1B18;
    border-color: #2A2824;
    color: #5E5A53;
}
QFrame#settingsSidebar {
    background-color: #171614;
    border-right: 1px solid #2F2D29;
}
QFrame#settingsCard {
    background-color: #1F1E1B;
    border: 1px solid #312F2A;
    border-radius: 10px;
}
QPushButton[variant="settingsNav"] {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 8px 12px;
    text-align: left;
    color: #A5A096;
    font-size: __BASE_PX__px;
}
QPushButton[variant="settingsNav"]:hover {
    background-color: #262420;
    color: #ECE9E2;
}
QPushButton[variant="settingsNav"]:checked {
    background-color: #382822;
    border-color: #8A4C38;
    color: #F2C2AE;
    font-weight: 600;
}
QPushButton[variant="scaleCard"] {
    background-color: #181715;
    border: 1px solid #34322E;
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    color: #ECE9E2;
    font-size: __SMALL_PX__px;
}
QPushButton[variant="scaleCard"]:hover {
    background-color: #252320;
    border-color: #524E47;
}
QPushButton[variant="scaleCard"]:checked {
    background-color: #382822;
    border: 1px solid #D97757;
    color: #F5DDD3;
    font-weight: 600;
}
QTreeWidget#sidebarTree {
    background-color: #1B1A18;
    border: none;
    outline: 0;
    padding: 3px 0px 3px __TREE_LEFT_PAD__px;
}
QTableView#fileTable {
    background-color: #242320;
    alternate-background-color: #201F1C;
    border: none;
    gridline-color: #2B2A26;
    selection-background-color: #47332A;
    selection-color: #F7F4EE;
    outline: 0;
}
QTableView#fileTable::item:hover {
    background-color: #2B2A26;
}
QTableView#fileTable::item:selected {
    background-color: #47332A;
    color: #F7F4EE;
}
QTableView#fileTable::item:selected:!active {
    background-color: #382D28;
    color: #E5E0D7;
}
QTreeWidget:focus, QTableView:focus {
    outline: 0;
}
QHeaderView {
    background-color: #1F1E1B;
    border: none;
}
QHeaderView::section {
    background-color: #1F1E1B;
    color: #A5A096;
    border: none;
    border-right: 1px solid #2D2B27;
    border-bottom: 1px solid #34322E;
    padding: 5px 8px;
    font-size: __SMALL_PX__px;
}
QHeaderView::section:hover {
    background-color: #282622;
    color: #ECE9E2;
}
QSplitter::handle {
    background-color: #34322E;
}
QStatusBar {
    background-color: #1F1E1B;
    border-top: 1px solid #34322E;
    color: #A5A096;
    font-size: __SMALL_PX__px;
}
QStatusBar::item {
    border: none;
}
QComboBox {
    background-color: #2B2A26;
    border: 1px solid #3D3A34;
    border-radius: 6px;
    padding: 3px 22px 3px 9px;
    color: #ECE9E2;
    font-size: __SMALL_PX__px;
}
QComboBox:hover {
    background-color: #35332E;
    border-color: #524E47;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background-color: #272622;
    border: 1px solid #3E3B35;
    border-radius: 6px;
    color: #ECE9E2;
    selection-background-color: #382822;
    selection-color: #F0A387;
    padding: 3px;
    outline: 0;
}
QCheckBox {
    spacing: 6px;
    color: #A5A096;
    font-size: __SMALL_PX__px;
    background: transparent;
}
QCheckBox:hover {
    color: #ECE9E2;
}
QCheckBox::indicator {
    width: __CHECKBOX_IND__px;
    height: __CHECKBOX_IND__px;
    border-radius: 4px;
    border: 1px solid #524E47;
    background-color: #181715;
}
QCheckBox::indicator:checked {
    border: 1px solid #D97757;
    background-color: #D97757;
}
QMenu {
    background-color: #272622;
    border: 1px solid #3E3B35;
    border-radius: 8px;
    padding: 4px;
    color: #ECE9E2;
}
QMenu::item {
    padding: 5px 14px 5px 12px;
    border-radius: 5px;
    font-size: __SMALL_PX__px;
}
QMenu::item:selected {
    background-color: #382822;
    color: #F0A387;
}
QMenu::item:disabled {
    color: #5E5A53;
}
QMenu::separator {
    height: 1px;
    background: #34322E;
    margin: 4px 6px;
}
QToolTip {
    background-color: #2B2A26;
    color: #ECE9E2;
    border: 1px solid #3E3B35;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: __SMALL_PX__px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollBar:vertical {
    background: #1B1A18;
    width: 12px;
    margin: 0px;
    padding: 2px;
    border-left: 1px solid #2A2824;
}
QScrollBar:horizontal {
    background: #1B1A18;
    height: 12px;
    margin: 0px;
    padding: 2px;
    border-top: 1px solid #2A2824;
}
QTableView#fileTable QScrollBar:vertical {
    width: 14px;
    padding: 2px 3px;
}
QTableView#fileTable QScrollBar:horizontal {
    height: 14px;
    padding: 3px 2px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #524E46;
    border-radius: 4px;
    min-height: 36px;
    min-width: 36px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #7A746A;
}
QScrollBar::handle:vertical:pressed, QScrollBar::handle:horizontal:pressed {
    background: #D97757;
}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
    height: 0;
    width: 0;
}
QLabel[pill="spec"] {
    background-color: #2B2A26;
    border: 1px solid #34322E;
    border-radius: 5px;
    color: #A5A096;
    padding: 1px 6px;
    font-size: __XS_PX__px;
}
QLabel[pill="statusOnline"] {
    background-color: #382822;
    border: 1px solid #8A4C38;
    border-radius: 5px;
    color: #F0A387;
    padding: 1px 6px;
    font-size: __XS_PX__px;
}
QLabel[pill="statusOffline"] {
    background-color: #272522;
    border: 1px solid #3D3A34;
    border-radius: 5px;
    color: #736E66;
    padding: 1px 6px;
    font-size: __XS_PX__px;
}
QLabel[pill="dirty"] {
    background-color: #382822;
    border: 1px solid #9E553D;
    border-radius: 5px;
    color: #F0A387;
    padding: 1px 6px;
    font-size: __XS_PX__px;
    font-weight: 600;
}
QDialog {
    background-color: #1B1A18;
    color: #ECE9E2;
}
QTabWidget::pane {
    background-color: #1F1E1B;
    border: 1px solid #34322E;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: #181715;
    border: 1px solid #34322E;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 14px;
    margin-right: 4px;
    color: #A5A096;
}
QTabBar::tab:selected {
    background-color: #1F1E1B;
    border-color: #4A463F;
    color: #ECE9E2;
    font-weight: 600;
}
QListWidget, QTableWidget {
    background-color: #1A1917;
    alternate-background-color: #201F1C;
    border: 1px solid #34322E;
    border-radius: 7px;
    gridline-color: #2B2A26;
    color: #ECE9E2;
    selection-background-color: #382822;
    selection-color: #F3F0EA;
    outline: 0;
}
QListWidget::item {
    padding: 4px 8px;
    border-radius: 4px;
}
QListWidget::item:hover {
    background-color: #2B2A26;
}
QProgressBar {
    background-color: #181715;
    border: 1px solid #34322E;
    border-radius: 6px;
    text-align: center;
    color: #ECE9E2;
    font-size: __XS_PX__px;
    min-height: __PROG_H__px;
}
QProgressBar::chunk {
    background-color: #D97757;
    border-radius: 5px;
}
"""


def build_stylesheet(scale_key: str = DEFAULT_UI_SCALE) -> str:
    """Build the full application stylesheet for ``scale_key``."""
    spec = ui_scale_spec(scale_key)
    mode_btn_h = max(24, spec.input_h - 6)
    card_btn_h = max(18, spec.btn_h - 8)
    checkbox_ind = max(13, min(18, int(round(14 * spec.scale))))
    prog_h = max(15, int(round(16 * spec.scale)))
    replacements = {
        "__BASE_PX__": str(spec.base_px),
        "__SMALL_PX__": str(spec.small_px),
        "__XS_PX__": str(spec.xs_px),
        "__BREADCRUMB_PX__": str(spec.base_px),
        "__UCS_PX__": str(spec.small_px),
        "__MODE_BTN_H__": str(mode_btn_h),
        "__CARD_BTN_H__": str(card_btn_h),
        "__TREE_LEFT_PAD__": str(spec.tree_left_pad),
        "__CHECKBOX_IND__": str(checkbox_ind),
        "__PROG_H__": str(prog_h),
    }
    qss = _STYLESHEET_TEMPLATE
    for token, value in replacements.items():
        qss = qss.replace(token, value)
    return qss


STYLESHEET = build_stylesheet(DEFAULT_UI_SCALE)


def mono_font(point_size: int = 10) -> QFont:
    font = QFont("Cascadia Mono")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(point_size)
    return font


def dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, SIDEBAR)
    palette.setColor(QPalette.ColorRole.WindowText, TEXT)
    palette.setColor(QPalette.ColorRole.Base, CANVAS)
    palette.setColor(QPalette.ColorRole.AlternateBase, ALTERNATE_ROW)
    palette.setColor(QPalette.ColorRole.Text, TEXT)
    palette.setColor(QPalette.ColorRole.Button, RAISED)
    palette.setColor(QPalette.ColorRole.ButtonText, TEXT)
    palette.setColor(QPalette.ColorRole.BrightText, PLAY)
    palette.setColor(QPalette.ColorRole.Highlight, SELECT)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#F7F4EE"))
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, SELECT_INACTIVE)
    palette.setColor(
        QPalette.ColorGroup.Inactive, QPalette.ColorRole.HighlightedText, QColor("#E5E0D7")
    )
    palette.setColor(QPalette.ColorRole.ToolTipBase, RAISED)
    palette.setColor(QPalette.ColorRole.ToolTipText, TEXT)
    palette.setColor(QPalette.ColorRole.PlaceholderText, TEXT_FAINT)
    palette.setColor(QPalette.ColorRole.Link, SELECT_MARK)
    return palette


def apply_theme(app: QApplication, ui_scale: str | None = None) -> UiScaleSpec:
    global _ACTIVE_UI_SCALE
    if ui_scale is not None:
        _ACTIVE_UI_SCALE = normalize_ui_scale(ui_scale)
    st = app.style()
    if st is None or st.objectName().lower() != "fusion":
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
    app.setStyleSheet(build_stylesheet(_ACTIVE_UI_SCALE))
    return UI_SCALE_PRESETS[_ACTIVE_UI_SCALE]


_SPECTRAL_LUTS: tuple[list[list[QColor]], list[list[QColor]]] | None = None


def spectral_color_luts() -> tuple[list[list[QColor]], list[list[QColor]]]:
    """Return cached ``(normal, played)`` ``(64, 8)`` ``QColor`` lookup tables."""
    global _SPECTRAL_LUTS
    if _SPECTRAL_LUTS is None:
        normal: list[list[QColor]] = []
        played: list[list[QColor]] = []
        for h in range(64):
            freq_u8 = min(255, h * 4 + 2)
            row_n: list[QColor] = []
            row_p: list[QColor] = []
            for t in range(8):
                ton_u8 = min(255, t * 32 + 16)
                rn, gn, bn = spectral_color_rgb(freq_u8, ton_u8, played=False)
                rp, gp, bp = spectral_color_rgb(freq_u8, ton_u8, played=True)
                row_n.append(QColor(rn, gn, bn))
                row_p.append(QColor(rp, gp, bp))
            normal.append(row_n)
            played.append(row_p)
        _SPECTRAL_LUTS = (normal, played)
    return _SPECTRAL_LUTS
