"""AI query planning: CN->EN translation bridge, full-sentence CLAP2 prompt,
ME hard-exclusion extraction, and dynamic text/audio route weighting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from audiosearch.retrieval.query.interpret import interpret

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_NEG_TOKEN_RE = re.compile(
    r'(?:(?<=^)|(?<=\s))(?:-"([^"]+)"|-([^\s]+)|NOT\s+"([^"]+)"|NOT\s+([^\s]+))',
    re.IGNORECASE,
)
_CN_NEG_CLAUSE_RE = re.compile(r"(?:不要|排除|不含|没有|去掉|别)[\u4e00-\u9fffA-Za-z0-9_-]*")

# Spoken numeric+unit phrases normalized before ``interpret()`` (which splits digits & CJK).
_PRE_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(?:8|八)\s*-?\s*位"), " eight-bit chiptune "),
    (re.compile(r"(?i)(?:16|十六)\s*-?\s*位"), " sixteen-bit retro "),
)


@dataclass(frozen=True, slots=True)
class AiQueryPlan:
    raw_query: str
    positive_text_query: str
    audio_prompt: str
    is_chinese: bool
    translation_coverage: float
    hard_exclusions: tuple[str, ...]
    omitted_spans: tuple[str, ...]
    text_weight: float
    audio_weight: float


def _has_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def _normalize_for_interpret(text: str) -> str:
    out = text
    for pattern, repl in _PRE_NORMALIZATIONS:
        out = pattern.sub(repl, out)
    return " ".join(out.split())


def _extract_me_exclusions(query: str) -> tuple[str, list[str]]:
    exclusions: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        raw_val = next((g for g in match.groups() if g is not None), "").strip()
        if raw_val:
            exclusions.append(raw_val)
        return " "

    positive = _NEG_TOKEN_RE.sub(_collect, query)
    positive = " ".join(positive.split())
    return positive, exclusions


def _expand_exclusion_terms(raw_terms: list[str], extra_translated: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []

    def add(term: str) -> None:
        cleaned = term.strip().casefold()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)

    for raw in raw_terms:
        add(raw)
        if _has_cjk(raw):
            interp = interpret(_normalize_for_interpret(raw))
            for desc in interp.audio_descriptions:
                for token in desc.split():
                    if not _has_cjk(token):
                        add(token)
    for term in extra_translated:
        for token in term.split():
            add(token)
    return tuple(seen)


def build_clap2_prompt(raw_desc: str) -> str:
    """Format an English description as a clean full sentence for CLAP2."""
    # Strip any untranslated CJK spans so CLAP2's English text tower receives pure English.
    ascii_parts = _ASCII_TOKEN_RE.findall(raw_desc)
    deduped: list[str] = []
    for part in ascii_parts:
        token = part.strip()
        if not token:
            continue
        if not deduped or deduped[-1].casefold() != token.casefold():
            deduped.append(token)
    cleaned = " ".join(deduped)
    if not cleaned:
        return ""
    lowered = cleaned.casefold()
    words = cleaned.split()
    if len(words) >= 7 or lowered.startswith(("a ", "an ", "the ")):
        sentence = cleaned[0].upper() + cleaned[1:]
        return sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."
    return f"A clear sound effect of {cleaned}."


def analyze_ai_query(query: str) -> AiQueryPlan:
    """Build dual-route queries, hard exclusions, and dynamic CN/EN fusion weights."""
    raw = query.strip()
    if not raw:
        return AiQueryPlan(
            raw_query="",
            positive_text_query="",
            audio_prompt="",
            is_chinese=False,
            translation_coverage=1.0,
            hard_exclusions=(),
            omitted_spans=(),
            text_weight=1.0,
            audio_weight=1.0,
        )

    positive_raw, me_excluded = _extract_me_exclusions(raw)
    is_chinese = _has_cjk(raw)

    interp_input = _normalize_for_interpret(positive_raw)
    interpreted = interpret(interp_input) if interp_input else None

    if interpreted is not None and interpreted.exclusion_scope == "recognized_not_enforced":
        # Remove Chinese negation clauses from positive query sent to dense text tower.
        cleaned_pos = _CN_NEG_CLAUSE_RE.sub(" ", positive_raw)
        cleaned_pos = " ".join(cleaned_pos.split())
        if cleaned_pos:
            positive_text_query = cleaned_pos
        elif interpreted.audio_descriptions:
            positive_text_query = interpreted.audio_descriptions[0]
        else:
            positive_text_query = ""
    else:
        positive_text_query = positive_raw

    extra_excl = interpreted.excluded_terms if interpreted is not None else ()
    hard_exclusions = _expand_exclusion_terms(me_excluded, extra_excl)
    omitted_spans = interpreted.omitted_spans if interpreted is not None else ()

    if interpreted is None:
        translation_coverage = 1.0
        audio_prompt = ""
    else:
        if _has_cjk(interp_input):
            total_cjk = sum(len(span) for span in _CJK_RE.findall(interp_input))
            filler_cjk = sum(len(span) for span in _CJK_RE.findall("".join(interpreted.dropped_fillers)))
            content_cjk = max(0, total_cjk - filler_cjk)
            omitted_cjk = sum(len(span) for span in _CJK_RE.findall(" ".join(omitted_spans)))
            if content_cjk <= 0:
                translation_coverage = 1.0
            else:
                translation_coverage = max(0.0, min(1.0, 1.0 - (omitted_cjk / content_cjk)))
        else:
            translation_coverage = 1.0

        if not _has_cjk(positive_raw) and not (
            interpreted.exclusion_scope == "recognized_not_enforced"
        ):
            # Pure English query without negation: use full sentence form directly.
            audio_prompt = build_clap2_prompt(positive_raw)
        elif interpreted.audio_descriptions:
            audio_prompt = build_clap2_prompt(interpreted.audio_descriptions[0])
        else:
            audio_prompt = ""

    if not audio_prompt or (is_chinese and translation_coverage < 0.25):
        text_weight, audio_weight = 1.5, 0.25
    elif is_chinese and translation_coverage < 0.85:
        text_weight, audio_weight = 1.25, 0.75
    else:
        text_weight, audio_weight = 1.0, 1.0

    return AiQueryPlan(
        raw_query=raw,
        positive_text_query=positive_text_query,
        audio_prompt=audio_prompt,
        is_chinese=is_chinese,
        translation_coverage=round(translation_coverage, 4),
        hard_exclusions=hard_exclusions,
        omitted_spans=omitted_spans,
        text_weight=text_weight,
        audio_weight=audio_weight,
    )


__all__ = [
    "AiQueryPlan",
    "analyze_ai_query",
    "build_clap2_prompt",
]
