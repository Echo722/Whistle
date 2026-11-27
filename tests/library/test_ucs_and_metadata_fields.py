"""Behavioral tests for UCS v8.2 catalog, filename reverse-parser, and batch transforms."""

from __future__ import annotations

import pytest

from audiosearch.library.metadata_fields import EDITABLE_FIELD_KEYS, apply_field_transform
from audiosearch.library.ucs import (
    list_ucs_entries,
    lookup_cat_id,
    parse_ucs_filename,
    search_ucs,
    validate_cat_id,
)


def test_ucs_catalog_lookup_validation_and_bilingual_search():
    entries = list_ucs_entries()
    assert len(entries) >= 150

    mag = lookup_cat_id("magspel")
    assert mag is not None
    assert mag.cat_id == "MAGSpel"
    assert mag.category == "MAGIC"
    assert mag.subcategory == "SPELL"

    # Supports UCS v8.2 CatID-UserCategory form
    mag_ext = lookup_cat_id("MAGSpel-Fireball")
    assert mag_ext is not None
    assert mag_ext.cat_id == "MAGSpel"

    ok, matched, msg = validate_cat_id("MAGSpel-Ice")
    assert ok is True
    assert matched is mag
    assert "Ice" in msg

    ok_bad, matched_bad, _msg_bad = validate_cat_id("bad id with spaces")
    assert ok_bad is False
    assert matched_bad is None

    zh_hits = search_ucs("魔法")
    assert any(e.cat_id == "MAGSpel" for e in zh_hits)

    en_hits = search_ucs("explosion")
    assert any(e.category == "EXPLOSIONS" for e in en_hits)


def test_ucs_filename_reverse_parsing():
    strict = parse_ucs_filename("MAGSpel-Cast_ArcaneBolt_Echo_MagicPack_02.wav")
    assert strict.matched is True
    assert strict.is_strict_ucs is True
    assert strict.cat_id == "MAGSpel-Cast"
    assert strict.fields["category"] == "MAGIC"
    assert strict.fields["subcategory"] == "SPELL"
    assert strict.fields["fx_name"] == "ArcaneBolt"
    assert strict.fields["description"] == "ArcaneBolt"
    assert strict.fields["designer"] == "Echo"
    assert strict.fields["library"] == "MagicPack"
    assert strict.fields["notes"] == "02"

    non_ucs = parse_ucs_filename("random_coin_sound_01.wav")
    assert non_ucs.matched is False
    assert non_ucs.is_strict_ucs is False
    assert non_ucs.fields == {}


def test_batch_field_transform_operations():
    assert len(EDITABLE_FIELD_KEYS) == 9

    assert apply_field_transform("old", "set", "new desc") == "new desc"
    assert apply_field_transform("whoosh", "prepend", "Fire ") == "Fire whoosh"
    assert apply_field_transform("Fire", "append", " loop") == "Fire loop"
    assert (
        apply_field_transform("Coin Pickup", "replace", find="coin", replace_with="Gold", case_sensitive=False)
        == "Gold Pickup"
    )
    assert (
        apply_field_transform("Coin Pickup", "replace", find="coin", replace_with="Gold", case_sensitive=True)
        == "Coin Pickup"
    )
    assert (
        apply_field_transform("SFX_Sword_012", "regex_replace", find=r"_(\d+)$", replace_with=r"_v\1")
        == "SFX_Sword_v012"
    )
    assert apply_field_transform("anything", "clear") == ""

    with pytest.raises(ValueError, match="正则表达式无效"):
        apply_field_transform("abc", "regex_replace", find="[unclosed", replace_with="x")
