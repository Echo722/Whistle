"""Rules-based query interpretation tests.

This module is a small, versioned rule set; the tests pin the behaviour that the
retrievers depend on (keyword groups for AND/OR, English descriptions for CLAP,
raw text preserved for BGE-M3) rather than any notion of cleverness.
"""

from __future__ import annotations

import pytest

from audiosearch.retrieval.query.interpret import (
    CODE_EXCLUSION_NOT_ENFORCED,
    CODE_EXCLUSION_SCOPE_UNCERTAIN,
    CODE_PARTIAL_UNCONVERTED,
    RULES_VERSION,
    build_plan,
    interpret,
)


def test_english_query_keeps_terms_and_builds_a_description():
    result = interpret("metal gong hit")
    assert result.keyword_groups == [["metal"], ["gong"], ["hit"]]
    assert result.audio_descriptions == ["metal gong hit"]
    assert result.text_query == "metal gong hit"


def test_chinese_terms_become_english_descriptions_for_clap():
    result = interpret("魔法 施法 呼啸")
    assert result.audio_descriptions == ["magic spell casting whoosh"]
    assert result.text_query == "魔法 施法 呼啸"


def test_original_text_is_never_translated_away():
    result = interpret("水滴声")
    assert result.text_query == "水滴声"
    assert result.audio_descriptions == ["water drop"]


def test_fillers_are_dropped_but_reported():
    result = interpret("帮我找一下 火焰 的声音")
    assert "找" in result.dropped_fillers
    assert "声音" in result.dropped_fillers
    assert all("帮我" not in group for group in result.keyword_groups)
    assert result.audio_descriptions[0] == "flame"


def test_mixed_language_keeps_both_sides_and_english_order():
    result = interpret("glass shatter 玻璃碎裂")
    assert result.text_query == "glass shatter 玻璃碎裂"
    assert result.audio_descriptions[0].startswith("glass shatter")
    assert "glass shatter" in result.audio_descriptions[0]


def test_unknown_continuous_span_is_kept_whole_with_omission_hint():
    # Two characters outside the vocabulary must survive as one span, not as
    # claimed-understood tags and not as silently dropped letters.
    result = interpret("\u9515\u9f8d")
    assert result.keyword_groups == [["\u9515\u9f8d"]]
    assert "\u9515\u9f8d" in result.audio_descriptions[0]
    assert "\u9515\u9f8d" in result.omitted_spans
    assert CODE_PARTIAL_UNCONVERTED in result.warning_codes
    assert any("部分未转换" in warning for warning in result.warnings)


def test_one_char_term_does_not_split_an_unknown_compound():
    # 「火」is a known term, but it must not carve "fire" out of an unlisted
    # compound. This phrase is intentionally not in the freeze table.
    result = interpret("火折子轻响")
    assert all("fire" != term for group in result.keyword_groups for term in group)
    assert "fire" not in result.audio_descriptions[0]
    assert any("火折子" in span for span in result.omitted_spans)
    assert CODE_PARTIAL_UNCONVERTED in result.warning_codes


def test_known_gaps_map_to_explainable_english():
    assert interpret("犬吠").audio_descriptions == ["dog bark"]
    assert interpret("引擎声").audio_descriptions == ["engine"]
    assert interpret("摩擦声").audio_descriptions == ["friction"]
    assert interpret("碎裂声").audio_descriptions == ["shatter"]


def test_new_combination_is_composed_not_memorized():
    # Not a freeze-table row: modifier + material + action still compose.
    result = interpret("潮湿木头缓慢敲击")
    assert result.text_query == "潮湿木头缓慢敲击"
    description = result.audio_descriptions[0]
    assert "damp" in description
    assert "wood" in description
    assert "slow" in description
    assert "knock" in description


def test_negation_does_not_become_a_positive_audio_description():
    result = interpret("玻璃碎裂，不要爆炸")
    assert result.text_query == "玻璃碎裂，不要爆炸"
    assert "explosion" not in result.audio_descriptions[0]
    assert "glass shatter" in result.audio_descriptions[0]
    assert "explosion" in result.excluded_terms
    assert result.exclusion_scope == "recognized_not_enforced"
    assert CODE_EXCLUSION_NOT_ENFORCED in result.warning_codes
    assert any("排除未强制执行" in warning for warning in result.warnings)


def test_only_exclusion_does_not_invent_a_source():
    result = interpret("不要爆炸")
    assert result.text_query == "不要爆炸"
    assert result.audio_descriptions == []
    assert "explosion" not in " ".join(result.audio_descriptions)
    assert result.keyword_groups == []
    assert result.exclusion_scope == "recognized_not_enforced"
    assert CODE_EXCLUSION_NOT_ENFORCED in result.warning_codes


def test_double_negation_is_conservative_and_keeps_original():
    result = interpret("不是没有水花")
    assert result.text_query == "不是没有水花"
    assert result.audio_descriptions == ["不是没有水花"]
    assert result.exclusion_scope == "uncertain"
    assert CODE_EXCLUSION_SCOPE_UNCERTAIN in result.warning_codes
    assert result.excluded_terms == ()
    assert any("无法可靠判定" in warning for warning in result.warnings)


def test_english_without_is_kept_out_of_the_positive_description():
    result = interpret("whoosh without explosion")
    assert result.text_query == "whoosh without explosion"
    assert result.audio_descriptions == ["whoosh"]
    assert "explosion" in result.excluded_terms
    assert CODE_EXCLUSION_NOT_ENFORCED in result.warning_codes


def test_filename_style_is_kept_as_one_token():
    result = interpret("metal_hit_01.wav")
    assert result.text_query == "metal_hit_01.wav"
    assert result.audio_descriptions == ["metal_hit_01.wav"]
    assert result.keyword_groups == [["metal_hit_01.wav"]]


def test_plan_records_that_it_came_from_rules():
    plan = build_plan("metal gong")
    assert plan.interpretation == "rules"
    assert plan.interpretation_version == RULES_VERSION
    assert RULES_VERSION == "query-rules-v2"
    assert plan.original_text == "metal gong"


def test_empty_query_produces_no_descriptions_or_rule_claim():
    result = interpret("   ")
    assert result.audio_descriptions == []
    assert result.keyword_groups == []
    assert result.warnings == ()
    assert result.warning_codes == ()


def test_description_count_is_capped():
    result = interpret("fire wind water metal glass", max_descriptions=3)
    assert len(result.audio_descriptions) <= 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("水花", "water splash"),
        ("电流 嗡嗡", "electric hum"),
        ("火焰 噼啪", "flame crackling"),
    ],
)
def test_known_terms_map_to_english(text, expected):
    assert interpret(text).audio_descriptions[0] == expected


@pytest.mark.parametrize(
    "query",
    [
        "a distant dog in the forest",
        "the sound of a small bell by the river",
        "tap tap",
        "Rock and Roll.wav",
        'kick drum, "dry" take!',
        "whoosh flyby (v2)",
    ],
)
def test_plain_english_keeps_wording_order_repeats_and_punctuation(query):
    result = interpret(query)
    assert result.text_query == query
    assert result.audio_descriptions == [query]
    assert result.exclusion_scope == "none"


def test_plain_english_does_not_drop_articles_or_fold_repeats():
    # These are the independently observed failure modes, plus a constructed repeat.
    assert interpret("a distant dog in the forest").audio_descriptions == [
        "a distant dog in the forest"
    ]
    assert interpret("Rock and Roll.wav").audio_descriptions == ["Rock and Roll.wav"]
    assert interpret("tap tap").audio_descriptions == ["tap tap"]


def test_clear_english_negation_still_drops_the_excluded_object():
    result = interpret("whoosh without explosion")
    assert result.text_query == "whoosh without explosion"
    assert result.audio_descriptions == ["whoosh"]
    assert "explosion" in result.excluded_terms
    assert result.exclusion_scope == "recognized_not_enforced"
