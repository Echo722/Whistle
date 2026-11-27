"""R5b AI metadata enrichment: CLAP2 zero-shot UCS classification, descriptions, and CN<->EN translation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiosearch.ai.indexer import decode_audio_windows_mono
from audiosearch.ai.query_rules import analyze_ai_query
from audiosearch.ai.search_engine import AiSearchEngine
from audiosearch.ai.vector_store import LibraryVectorIndex
from audiosearch.library.store import LibraryStore
from audiosearch.library.ucs import UCS_CATALOG, UcsEntry, parse_ucs_filename
from audiosearch.retrieval.query.interpret import _TERM_MAP

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NOISE_WORDS = frozenset({"wav", "mp3", "flac", "ogg", "opus", "aif", "aiff", "sfx", "audio"})


def _build_en_to_zh_map() -> dict[str, str]:
    en_zh: dict[str, str] = {}
    # Seed concise dictionary terms first so clean words ("magic" -> "魔法") win over broad synonyms.
    for zh, en in sorted(_TERM_MAP.items(), key=lambda kv: (len(kv[1]), len(kv[0]))):
        en_cf = en.strip().casefold()
        if en_cf and en_cf not in en_zh:
            en_zh[en_cf] = zh
    for entry in UCS_CATALOG:
        en_zh.setdefault(entry.category.casefold(), entry.category_zh)
        en_zh.setdefault(entry.subcategory.casefold(), entry.subcategory_zh)
    for zh, en in sorted(_TERM_MAP.items(), key=lambda kv: len(kv[1]), reverse=True):
        for part in en.strip().casefold().split():
            if len(part) >= 3 and part not in en_zh:
                en_zh[part] = zh
    for entry in UCS_CATALOG:
        for syn in entry.synonyms:
            for tok in syn.split():
                t_cf = tok.strip().casefold()
                if len(t_cf) >= 3 and t_cf not in en_zh:
                    en_zh[t_cf] = entry.subcategory_zh
    return en_zh


_EN_TO_ZH_MAP = _build_en_to_zh_map()
_ZH_TERMS_LONGEST = tuple(sorted(_TERM_MAP.keys(), key=len, reverse=True))


def _tokenize_english_stem(text: str) -> list[str]:
    spaced = _CAMEL_SPLIT_RE.sub(" ", text.replace("_", " ").replace("-", " "))
    raw_tokens = [tok.strip(" ,.;:!?()[]{}") for tok in spaced.split()]
    out: list[str] = []
    for tok in raw_tokens:
        if not tok or tok.isdigit() or tok.casefold() in _NOISE_WORDS:
            continue
        out.append(tok)
    return out


def translate_sound_text(text: str, target_lang: str) -> str:
    """Offline bilingual sound-design translator (``'zh'`` or ``'en'``)."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    lang = target_lang.strip().lower()
    if lang == "en":
        if not _CJK_RE.search(cleaned):
            return cleaned
        plan = analyze_ai_query(cleaned)
        if plan.audio_prompt:
            # Strip wrapper "A clear sound effect of ... ." when compact phrase preferred
            p = plan.audio_prompt.strip().rstrip(".")
            prefix = "A clear sound effect of "
            if p.startswith(prefix):
                return p[len(prefix) :]
            return p
        out = cleaned
        for zh_term in _ZH_TERMS_LONGEST:
            if zh_term in out:
                out = out.replace(zh_term, f" {_TERM_MAP[zh_term]} ")
        return " ".join(out.split())

    if lang == "zh":
        tokens = _tokenize_english_stem(cleaned)
        if not tokens:
            return cleaned
        mapped: list[str] = []
        i = 0
        while i < len(tokens):
            hit = None
            if i + 1 < len(tokens):
                pair_cf = f"{tokens[i]} {tokens[i + 1]}".casefold()
                if pair_cf in _EN_TO_ZH_MAP:
                    hit = _EN_TO_ZH_MAP[pair_cf]
                    i += 2
            if hit is None:
                tok_cf = tokens[i].casefold()
                hit = _EN_TO_ZH_MAP.get(tok_cf, tokens[i])
                i += 1
            if hit not in mapped:
                mapped.append(hit)
        return " ".join(mapped)

    return cleaned


def _ucs_prompt(entry: UcsEntry) -> str:
    syns = ", ".join(entry.synonyms)
    base = f"A sound effect of {entry.category.lower()} {entry.subcategory.lower()}"
    return f"{base}, {syns}." if syns else f"{base}."


@dataclass(frozen=True, slots=True)
class UcsSuggestion:
    entry: UcsEntry
    score: float


@dataclass(frozen=True, slots=True)
class AiMetadataSuggestion:
    file_id: int
    filename: str
    current_fields: dict[str, str]
    suggested_fields: dict[str, str]
    top_ucs: tuple[UcsSuggestion, ...]
    zh_description: str
    en_description: str
    confidence: float
    note: str


class AiMetadataEnricher:
    """CLAP2 zero-shot UCS classifier, description synthesizer, and CN/EN translator."""

    def __init__(
        self,
        store: LibraryStore,
        index: LibraryVectorIndex | None = None,
        engine: AiSearchEngine | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.engine = engine
        self._ucs_entries: tuple[UcsEntry, ...] = UCS_CATALOG
        self._ucs_matrix: np.ndarray | None = None

    def _ensure_ucs_matrix(self) -> np.ndarray | None:
        if self._ucs_matrix is not None:
            return self._ucs_matrix
        if self.engine is None:
            return None
        prompts = [_ucs_prompt(entry) for entry in self._ucs_entries]
        with self.engine._lock:  # noqa: SLF001 - serialize CUDA/model calls under engine lock
            enc = self.engine._get_audio_encoder()  # noqa: SLF001
            raw = enc.encode_texts(prompts)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(self._ucs_entries):
            return None
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        self._ucs_matrix = (arr / norms).astype(np.float32)
        return self._ucs_matrix

    def _get_file_audio_vector(self, file_id: int, rec: Any) -> tuple[np.ndarray | None, str]:
        getter = getattr(self.index, "get_file_audio_vector", None) if self.index is not None else None
        if callable(getter):
            vec = getter(file_id)
            if vec is not None:
                norm = float(np.linalg.norm(vec))
                return (vec / norm if norm > 1e-12 else vec).astype(np.float32), "indexed"
        if self.engine is not None and rec.is_available and rec.full_path.exists():
            win = (0.0, min(10.0, max(0.25, float(rec.duration_sec or 1.0))))
            segs, sr = decode_audio_windows_mono(rec.full_path, [win])
            with self.engine._lock:  # noqa: SLF001
                enc = self.engine._get_audio_encoder()  # noqa: SLF001
                raw = enc.encode_audio_items([(segs[0], sr)], batch_size=1)[0]
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(arr))
            return (arr / norm if norm > 1e-12 else arr).astype(np.float32), "encoded"
        return None, "unavailable"

    def _rank_ucs_by_audio(self, vec: np.ndarray, top_k: int = 3) -> list[UcsSuggestion]:
        mat = self._ensure_ucs_matrix()
        if mat is None or mat.shape[1] != vec.shape[0]:
            return []
        scores = mat @ vec
        order = np.argsort(-scores)[:top_k]
        return [
            UcsSuggestion(entry=self._ucs_entries[int(idx)], score=float(scores[int(idx)]))
            for idx in order
        ]

    @staticmethod
    def _rank_ucs_by_filename(filename: str, top_k: int = 3) -> list[UcsSuggestion]:
        parsed = parse_ucs_filename(filename)
        if parsed.matched and parsed.entry is not None:
            return [UcsSuggestion(entry=parsed.entry, score=0.95)]
        tokens = {tok.casefold() for tok in _tokenize_english_stem(Path(filename).stem)}
        scored: list[tuple[int, UcsEntry]] = []
        for entry in UCS_CATALOG:
            words = {
                entry.cat_id.casefold(),
                entry.category.casefold(),
                entry.subcategory.casefold(),
                *(w.casefold() for s in entry.synonyms for w in s.split()),
            }
            overlap = len(tokens & words)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda item: (-item[0], item[1].cat_id))
        return [
            UcsSuggestion(entry=entry, score=min(0.85, 0.35 + 0.15 * hits))
            for hits, entry in scored[:top_k]
        ]

    def enrich_file(
        self,
        file_id: int,
        *,
        fill_ucs: bool = True,
        fill_desc: bool = True,
        translate_mode: str = "none",  # "none" | "zh" | "en"
        only_empty: bool = True,
    ) -> AiMetadataSuggestion:
        """Analyze one file via CLAP2 (or filename heuristic fallback) and propose metadata."""
        fid = int(file_id)
        rec = self.store.get_file(fid)
        if rec is None:
            return AiMetadataSuggestion(fid, f"#{fid}", {}, {}, (), "", "", 0.0, "文件记录不存在")

        canon_raw = self.store.get_canonical_metadata(fid)
        current = {k: v for k, (v, _s, _d) in canon_raw.items()}

        audio_vec, vec_source = self._get_file_audio_vector(fid, rec)
        top_ucs: list[UcsSuggestion] = []
        note_parts: list[str] = []

        if audio_vec is not None:
            try:
                top_ucs = self._rank_ucs_by_audio(audio_vec, top_k=3)
                if top_ucs:
                    note_parts.append("CLAP2 声波零样本分类" if vec_source == "indexed" else "CLAP2 即时声波分类")
            except Exception as exc:
                note_parts.append(f"CLAP2 分类降级：{exc}")

        if not top_ucs:
            top_ucs = self._rank_ucs_by_filename(rec.filename, top_k=3)
            if top_ucs:
                note_parts.append("文件名关键词启发式匹配")
            else:
                note_parts.append("未命中明确声波/文件名分类")

        best = top_ucs[0] if top_ucs else None
        confidence = max(0.0, float(best.score)) if best is not None else 0.0

        suggested: dict[str, str] = {}
        if fill_ucs and best is not None:
            for key, val in (
                ("cat_id", best.entry.cat_id),
                ("category", best.entry.category),
                ("subcategory", best.entry.subcategory),
            ):
                if not only_empty or not current.get(key, "").strip():
                    suggested[key] = val

        stem_tokens = _tokenize_english_stem(Path(rec.filename).stem)
        stem_phrase = " ".join(stem_tokens)

        en_desc = current.get("description", "").strip()
        if not en_desc:
            if best is not None and stem_phrase:
                en_desc = f"{best.entry.category.title()} {best.entry.subcategory.title()} - {stem_phrase}"
            elif best is not None:
                en_desc = f"{best.entry.category.title()} {best.entry.subcategory.title()}"
            else:
                en_desc = stem_phrase
        elif _CJK_RE.search(en_desc):
            en_desc = translate_sound_text(en_desc, "en")

        zh_desc = (
            f"{best.entry.category_zh}/{best.entry.subcategory_zh}：{translate_sound_text(stem_phrase, 'zh')}"
            if best is not None and stem_phrase
            else (f"{best.entry.category_zh}/{best.entry.subcategory_zh}" if best is not None else translate_sound_text(en_desc, "zh"))
        )

        if fill_desc:
            if translate_mode == "zh":
                proposed_desc = zh_desc or translate_sound_text(current.get("description", ""), "zh")
            elif translate_mode == "en":
                proposed_desc = translate_sound_text(current.get("description", "") or en_desc, "en")
            else:
                proposed_desc = (
                    f"{best.entry.category_zh}/{best.entry.subcategory_zh} ({en_desc})"
                    if best is not None and en_desc
                    else (zh_desc or en_desc)
                )

            if proposed_desc and (not only_empty or translate_mode != "none" or not current.get("description", "").strip()):
                suggested["description"] = proposed_desc

            kw_parts: list[str] = []
            if best is not None:
                kw_parts.extend([best.entry.category_zh, best.entry.subcategory_zh, best.entry.category.lower(), best.entry.subcategory.lower()])
                for syn in best.entry.synonyms[:2]:
                    kw_parts.extend(syn.split()[:3])
            kw_parts.extend(t.lower() for t in stem_tokens[:4])
            kw_text = ", ".join(dict.fromkeys(p for p in kw_parts if p))
            if kw_text and (not only_empty or not current.get("keywords", "").strip()):
                suggested["keywords"] = kw_text

            if not current.get("fx_name", "").strip() and stem_phrase:
                suggested["fx_name"] = stem_phrase

        return AiMetadataSuggestion(
            file_id=fid,
            filename=rec.filename,
            current_fields=current,
            suggested_fields=suggested,
            top_ucs=tuple(top_ucs),
            zh_description=zh_desc,
            en_description=en_desc,
            confidence=confidence,
            note=" · ".join(note_parts),
        )

    def enrich_files(
        self,
        file_ids: list[int],
        *,
        fill_ucs: bool = True,
        fill_desc: bool = True,
        translate_mode: str = "none",
        only_empty: bool = True,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[AiMetadataSuggestion]:
        unique_ids = list(dict.fromkeys(int(x) for x in file_ids))
        total = len(unique_ids)
        out: list[AiMetadataSuggestion] = []
        for idx, fid in enumerate(unique_ids, start=1):
            sug = self.enrich_file(
                fid,
                fill_ucs=fill_ucs,
                fill_desc=fill_desc,
                translate_mode=translate_mode,
                only_empty=only_empty,
            )
            out.append(sug)
            if on_progress is not None:
                on_progress(idx, total, sug.filename)
        return out


__all__ = [
    "AiMetadataEnricher",
    "AiMetadataSuggestion",
    "UcsSuggestion",
    "translate_sound_text",
]
