"""Read-only REAPER Media Explorer shortcuts parser and idempotent collection importer."""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from audiosearch.browser.reaper_setup import default_reaper_resource_dir
from audiosearch.library.store import LibraryStore

_DIVIDER_RE = re.compile(r"^\s*[-—–─_=~*]{3,}\s*(.*?)\s*[-—–─_=~*]{3,}\s*$")
_SHORTCUT_KEY_RE = re.compile(r"^Shortcut(T)?(\d+)$")
_SPECIAL_NAMES = frozenset(
    {"desktop", "my documents", "documents", "home", "drives", "computer", "this pc"}
)
_UNGROUPED_DEFAULT_NAME = "REAPER 快捷方式"


@dataclass(frozen=True, slots=True)
class ReaperShortcutItem:
    index: int
    raw_path: str
    raw_title: str
    label: str
    kind: str  # "folder"
    match_status: str  # "matched_root" | "matched_folder" | "unindexed_under_root" | "outside_root" | "missing"
    root_id: int | None
    folder_id: int | None
    folder_rel_path: str
    already_in_collection: bool = False


@dataclass(slots=True)
class ReaperShortcutGroup:
    name: str
    is_ungrouped: bool
    order: int
    items: list[ReaperShortcutItem] = field(default_factory=list)
    already_exists: bool = False

    @property
    def matched_items(self) -> list[ReaperShortcutItem]:
        return [
            item
            for item in self.items
            if item.match_status in ("matched_root", "matched_folder") and item.root_id is not None
        ]

    @property
    def new_matched_count(self) -> int:
        return sum(1 for item in self.matched_items if not item.already_in_collection)

    @property
    def existing_matched_count(self) -> int:
        return sum(1 for item in self.matched_items if item.already_in_collection)

    @property
    def unmatched_items(self) -> list[ReaperShortcutItem]:
        return [
            item
            for item in self.items
            if item.match_status not in ("matched_root", "matched_folder")
        ]


@dataclass(frozen=True, slots=True)
class ReaperShortcutsPlan:
    ini_path: Path
    total_raw_entries: int
    groups: list[ReaperShortcutGroup]
    skipped_special: list[tuple[int, str, str]]
    skipped_databases: list[tuple[int, str, str]]

    @property
    def total_matched(self) -> int:
        return sum(len(g.matched_items) for g in self.groups)

    @property
    def total_unmatched(self) -> int:
        return sum(len(g.unmatched_items) for g in self.groups)

    @property
    def has_divider_groups(self) -> bool:
        return any(not g.is_ungrouped for g in self.groups)


@dataclass(frozen=True, slots=True)
class ReaperImportReport:
    collections_created: int
    collections_reused: int
    folders_added: int
    folders_updated: int
    folders_skipped: int
    imported_collection_ids: list[int]


def default_reaper_ini_path(reaper_dir: Path | None = None) -> Path:
    base = Path(reaper_dir) if reaper_dir is not None else default_reaper_resource_dir()
    return base / "reaper.ini"


def _path_basename(raw_path: str) -> str:
    cleaned = raw_path.rstrip("\\/")
    if not cleaned:
        return ""
    if "\\" in cleaned or (len(cleaned) >= 2 and cleaned[1] == ":"):
        return ntpath.basename(cleaned)
    return posixpath.basename(cleaned) or os.path.basename(cleaned)


def _clean_shortcut_label(raw_title: str, raw_path: str) -> str:
    title = raw_title.strip()
    if len(title) >= 2 and title.startswith('"') and title.endswith('"'):
        title = title[1:-1].strip()
    if title:
        return title
    base = _path_basename(raw_path).strip()
    return base or raw_path.strip()


def _normalize_abs_dir(path_str: str) -> str:
    raw = path_str.strip()
    if not raw:
        return ""
    is_win_style = "\\" in raw or (len(raw) >= 2 and raw[1] == ":")
    if is_win_style:
        norm = ntpath.normpath(raw).replace("\\", "/")
    else:
        norm = posixpath.normpath(raw.replace("\\", "/"))
    if len(norm) > 3 or not (len(norm) == 3 and norm[1:3] == ":/"):
        if norm != "/":
            norm = norm.rstrip("/")
    return norm.casefold()


def parse_reaper_ini_shortcuts(
    ini_path: Path,
) -> tuple[
    int,
    list[tuple[str, bool, list[tuple[int, str, str, str]]]],
    list[tuple[int, str, str]],
    list[tuple[int, str, str]],
]:
    """Read REAPER ``[reaper_explorer]`` shortcuts read-only into ordered divider groups."""
    path = Path(ini_path)
    if not path.is_file():
        return (0, [], [], [])

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    in_explorer = False
    raw_entries: dict[int, dict[str, str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_explorer = line.casefold() == "[reaper_explorer]"
            continue
        if not in_explorer or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        m = _SHORTCUT_KEY_RE.match(key)
        if m is None:
            continue
        field_name = "title" if m.group(1) else "path"
        idx = int(m.group(2))
        raw_entries.setdefault(idx, {})[field_name] = value

    groups_order: list[str] = []
    groups_meta: dict[str, tuple[bool, list[tuple[int, str, str, str]]]] = {}
    skipped_special: list[tuple[int, str, str]] = []
    skipped_databases: list[tuple[int, str, str]] = []

    current_group = _UNGROUPED_DEFAULT_NAME
    current_is_ungrouped = True

    def _ensure_group(name: str, is_ungrouped: bool) -> list[tuple[int, str, str, str]]:
        if name not in groups_meta:
            groups_order.append(name)
            groups_meta[name] = (is_ungrouped, [])
        return groups_meta[name][1]

    for idx in sorted(raw_entries):
        entry = raw_entries[idx]
        raw_path = entry.get("path", "").strip()
        raw_title = entry.get("title", "").strip()
        if not raw_path and not raw_title:
            continue

        div_match = _DIVIDER_RE.match(raw_title) or _DIVIDER_RE.match(_path_basename(raw_path))
        if div_match is not None:
            gname = div_match.group(1).strip()
            if gname:
                current_group = gname
                current_is_ungrouped = False
                _ensure_group(current_group, False)
            continue

        low_path = raw_path.casefold()
        if (raw_path.startswith("<") and raw_path.endswith(">")) or low_path in _SPECIAL_NAMES:
            skipped_special.append((idx, raw_path, raw_title))
            continue

        if low_path.endswith(".reaperfilelist") or raw_title.startswith("DB:"):
            skipped_databases.append((idx, raw_path, raw_title))
            continue

        if not raw_path:
            continue

        label = _clean_shortcut_label(raw_title, raw_path)
        bucket = _ensure_group(current_group, current_is_ungrouped)
        bucket.append((idx, raw_path, raw_title, label))

    parsed_groups = [
        (name, groups_meta[name][0], groups_meta[name][1])
        for name in groups_order
        if groups_meta[name][1]
    ]
    return (len(raw_entries), parsed_groups, skipped_special, skipped_databases)


def build_reaper_import_plan(ini_path: Path, store: LibraryStore) -> ReaperShortcutsPlan:
    """Match parsed REAPER shortcuts against indexed roots and folders without mutating DB."""
    path = Path(ini_path)
    total_raw, parsed_groups, skipped_special, skipped_databases = parse_reaper_ini_shortcuts(path)

    roots = store.list_roots()
    root_entries = sorted(
        ((r, _normalize_abs_dir(r.path)) for r in roots if _normalize_abs_dir(r.path)),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    folder_paths_by_root: dict[int, dict[str, tuple[int, str]]] = {
        r.id: store.list_all_folder_paths(r.id) for r, _ in root_entries
    }

    existing_colls = {c.name: c.id for c in store.list_collections()}
    all_cf = store.list_all_collection_folders()
    existing_mounts_by_coll: dict[int, set[tuple[int, str]]] = {
        cid: {(int(cf.root_id), cf.folder_rel_path.strip("/").casefold()) for cf in cfs}
        for cid, cfs in all_cf.items()
    }

    groups: list[ReaperShortcutGroup] = []
    for order_idx, (group_name, is_ungrouped, raw_items) in enumerate(parsed_groups, start=1):
        coll_id = existing_colls.get(group_name)
        coll_mounts = existing_mounts_by_coll.get(coll_id, set()) if coll_id is not None else set()
        seen_in_group: set[tuple[int, str]] = set()
        built_items: list[ReaperShortcutItem] = []

        for idx, raw_path, raw_title, label in raw_items:
            norm_p = _normalize_abs_dir(raw_path)
            match_status = "missing"
            matched_root_id: int | None = None
            matched_folder_id: int | None = None
            matched_rel_path = ""

            matched_under_root = False
            for root_rec, norm_root in root_entries:
                if norm_p == norm_root:
                    match_status = "matched_root"
                    matched_root_id = int(root_rec.id)
                    matched_folder_id = None
                    matched_rel_path = ""
                    matched_under_root = True
                    break
                prefix = norm_root if norm_root.endswith("/") else f"{norm_root}/"
                if norm_p.startswith(prefix):
                    rel_key = norm_p[len(prefix) :].strip("/")
                    hit = folder_paths_by_root.get(root_rec.id, {}).get(rel_key)
                    if hit is not None:
                        match_status = "matched_folder"
                        matched_root_id = int(root_rec.id)
                        matched_folder_id = int(hit[0])
                        matched_rel_path = hit[1]
                    else:
                        match_status = (
                            "unindexed_under_root" if Path(raw_path).is_dir() else "missing"
                        )
                    matched_under_root = True
                    break

            if not matched_under_root:
                match_status = "outside_root" if Path(raw_path).is_dir() else "missing"

            if matched_root_id is not None:
                dedup_key = (matched_root_id, matched_rel_path.casefold())
                if dedup_key in seen_in_group:
                    continue
                seen_in_group.add(dedup_key)
                already_in = dedup_key in coll_mounts
            else:
                already_in = False

            built_items.append(
                ReaperShortcutItem(
                    index=idx,
                    raw_path=raw_path,
                    raw_title=raw_title,
                    label=label,
                    kind="folder",
                    match_status=match_status,
                    root_id=matched_root_id,
                    folder_id=matched_folder_id,
                    folder_rel_path=matched_rel_path,
                    already_in_collection=already_in,
                )
            )

        if built_items:
            groups.append(
                ReaperShortcutGroup(
                    name=group_name,
                    is_ungrouped=is_ungrouped,
                    order=order_idx,
                    items=built_items,
                    already_exists=coll_id is not None,
                )
            )

    return ReaperShortcutsPlan(
        ini_path=path,
        total_raw_entries=total_raw,
        groups=groups,
        skipped_special=skipped_special,
        skipped_databases=skipped_databases,
    )


def apply_reaper_import_plan(
    store: LibraryStore,
    plan: ReaperShortcutsPlan,
    *,
    selected_group_names: Iterable[str] | None = None,
) -> ReaperImportReport:
    """Import selected groups idempotently into ``collections`` and ``collection_folders``."""
    allowed = None if selected_group_names is None else set(selected_group_names)

    collections_created = 0
    collections_reused = 0
    folders_added = 0
    folders_updated = 0
    folders_skipped = 0
    imported_collection_ids: list[int] = []

    for group in plan.groups:
        if allowed is not None and group.name not in allowed:
            folders_skipped += len(group.items)
            continue

        matched = group.matched_items
        folders_skipped += len(group.unmatched_items)
        if not matched:
            continue

        coll, created = store.get_or_create_collection(group.name)
        if created:
            collections_created += 1
        else:
            collections_reused += 1
        imported_collection_ids.append(coll.id)

        existing_keys = {
            (int(cf.root_id), cf.folder_rel_path.strip("/").casefold())
            for cf in store.list_collection_folders(coll.id)
        }

        seen_group_keys: set[tuple[int, str]] = set()
        sort_pos = 0
        for item in matched:
            assert item.root_id is not None
            key = (int(item.root_id), item.folder_rel_path.strip("/").casefold())
            if key in seen_group_keys:
                continue
            seen_group_keys.add(key)
            sort_pos += 1
            store.add_folder_to_collection(
                coll.id,
                int(item.root_id),
                item.folder_id,
                item.folder_rel_path,
                label=item.label,
                sort_order=sort_pos,
            )
            if key in existing_keys:
                folders_updated += 1
            else:
                folders_added += 1
                existing_keys.add(key)

    return ReaperImportReport(
        collections_created=collections_created,
        collections_reused=collections_reused,
        folders_added=folders_added,
        folders_updated=folders_updated,
        folders_skipped=folders_skipped,
        imported_collection_ids=imported_collection_ids,
    )


__all__ = [
    "ReaperImportReport",
    "ReaperShortcutGroup",
    "ReaperShortcutItem",
    "ReaperShortcutsPlan",
    "apply_reaper_import_plan",
    "build_reaper_import_plan",
    "default_reaper_ini_path",
    "parse_reaper_ini_shortcuts",
]
