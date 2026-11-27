"""内嵌元数据解析：边界、编码、截断与容器。

这些都是纯解析测试：不写数据库，也不碰真实素材库。每个用例只针对一种文件形状，
失败时能直接看出是哪条规则被破坏。
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from wav_metadata import (
    bext_chunk,
    chunk,
    fmt_chunk,
    info_chunk,
    ixml_chunk,
    wave_file,
    write_file,
)

from audiosearch.catalog import embedded
from audiosearch.catalog.embedded import (
    ROLE_DESCRIPTION,
    ROLE_KEYWORD,
    Coverage,
    EmbeddedStatus,
    read_embedded_metadata,
)


def values_of(metadata, field: str) -> list[str]:
    return [value.value for value in metadata.values if value.field == field]


def parse(path: Path):
    return read_embedded_metadata(path)


# ------------------------------------------------------------------ 无元数据 / 容器
def test_wave_without_metadata_chunks_reports_no_metadata(tmp_path):
    path = write_file(tmp_path / "plain.wav", wave_file())
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.NO_METADATA
    assert metadata.values == ()
    assert metadata.chunks == ("fmt ", "data")


def test_empty_metadata_chunks_are_not_a_parse_failure(tmp_path):
    """工具写了空块：字段全空属于"没有元数据"，不是解析失败。"""
    path = write_file(
        tmp_path / "empty.wav",
        wave_file(after_data=(bext_chunk(), ixml_chunk("<BWFXML></BWFXML>"))),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.NO_METADATA
    assert metadata.values == ()


def test_rf64_is_reported_as_unsupported_instead_of_misparsed(tmp_path):
    path = write_file(tmp_path / "rf64.wav", wave_file(riff_id=b"RF64"))
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.UNSUPPORTED_CONTAINER
    assert metadata.container == "RF64"
    assert metadata.values == ()


def test_non_riff_file_is_unsupported(tmp_path):
    path = write_file(tmp_path / "notwave.wav", b"ID3\x04\x00\x00" + b"\x00" * 20)
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.UNSUPPORTED_CONTAINER
    assert metadata.values == ()


def test_short_file_is_malformed(tmp_path):
    path = write_file(tmp_path / "short.wav", b"RIFF\x00")
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.MALFORMED


def test_missing_file_is_failed_not_an_exception(tmp_path):
    metadata = parse(tmp_path / "gone.wav")
    assert metadata.status is EmbeddedStatus.FAILED
    assert metadata.notes


# ---------------------------------------------------------------------------- bext
def test_bext_description_and_provenance(tmp_path):
    path = write_file(
        tmp_path / "bext.wav",
        wave_file(
            after_data=(
                bext_chunk(
                    description="Sudden and extremely fast whoosh-by.",
                    originator="2014 Example Library",
                    originator_reference="www.example.com",
                    date=b"2014-05-01",
                    time_value=b"12:34:56",
                    time_reference=48000,
                    version=2,
                    coding_history=b"A=PCM,F=48000,W=24,M=stereo\r\n",
                ),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.OK
    assert metadata.description == "Sudden and extremely fast whoosh-by."
    description = [value for value in metadata.values if value.role == ROLE_DESCRIPTION]
    assert len(description) == 1
    assert description[0].source == "embedded:bext"
    assert values_of(metadata, "bext_originator") == ["2014 Example Library"]
    assert values_of(metadata, "bext_originator_reference") == ["www.example.com"]
    assert values_of(metadata, "bext_origination_date") == ["2014-05-01"]
    assert values_of(metadata, "bext_origination_time") == ["12:34:56"]
    assert values_of(metadata, "bext_time_reference") == ["48000"]
    assert values_of(metadata, "bext_version") == ["2"]
    assert values_of(metadata, "bext_coding_history") == ["A=PCM,F=48000,W=24,M=stereo"]


def test_binary_date_field_is_skipped_and_reported(tmp_path):
    """真实样本里 origination date 是二进制：必须跳过并说明，而不是当成日期。"""
    path = write_file(
        tmp_path / "baddate.wav",
        wave_file(
            after_data=(
                bext_chunk(
                    description="Still readable.",
                    date=b"\x00\x00\x00\x80\x00\x00\x00\x80h\xdc",
                    time_value=b"\x00\x00\x00\x80-\xeeH\xc0",
                ),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.OK
    assert metadata.description == "Still readable."
    assert values_of(metadata, "bext_origination_date") == []
    assert values_of(metadata, "bext_origination_time") == []
    assert any("OriginationDate" in note for note in metadata.notes)
    assert any("OriginationTime" in note for note in metadata.notes)


def test_impossible_calendar_date_is_skipped(tmp_path):
    path = write_file(
        tmp_path / "calendar.wav", wave_file(after_data=(bext_chunk(date=b"2014-19-45"),))
    )
    metadata = parse(path)
    assert values_of(metadata, "bext_origination_date") == []
    assert any("YYYY-MM-DD" in note for note in metadata.notes)


def test_short_bext_is_parsed_but_reported(tmp_path):
    """固定长度不足 602 字节：能读的字段照读，同时说明块被截短。"""
    path = write_file(
        tmp_path / "shortbext.wav",
        wave_file(after_data=(bext_chunk(description="Only the head.", fixed_bytes=348),)),
    )
    metadata = parse(path)
    assert metadata.description == "Only the head."
    assert any("602" in note for note in metadata.notes)


# ----------------------------------------------------------------------- LIST/INFO
def test_list_info_values_and_keyword_mapping(tmp_path):
    path = write_file(
        tmp_path / "info.wav",
        wave_file(
            after_data=(
                info_chunk(
                    [
                        (b"INAM", b"Metal Impact 01"),
                        (b"ICMT", b"recorded on a wet floor"),
                        (b"ISBJ", b"impact, metal, wet"),
                        (b"ISFT", b"SoundApp 1.2"),
                    ]
                ),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.OK
    assert values_of(metadata, "info_title") == ["Metal Impact 01"]
    assert values_of(metadata, "info_software") == ["SoundApp 1.2"]
    keywords = [value for value in metadata.values if value.role == ROLE_KEYWORD]
    assert {value.value for value in keywords} == {
        "recorded on a wet floor",
        "impact, metal, wet",
    }
    assert {value.source for value in keywords} == {"embedded:info:ICMT", "embedded:info:ISBJ"}


def test_list_info_chinese_utf8_survives(tmp_path):
    path = write_file(
        tmp_path / "zh.wav",
        wave_file(after_data=(info_chunk([(b"ICMT", "雨天金属撞击，带水声".encode())]),)),
    )
    metadata = parse(path)
    assert values_of(metadata, "info_comment") == ["雨天金属撞击，带水声"]


def test_list_non_utf8_text_falls_back_and_says_so(tmp_path):
    """非法 UTF-8：按 cp1252 解码并注明，不静默丢弃、也不假装是 UTF-8。"""
    path = write_file(
        tmp_path / "latin.wav",
        wave_file(after_data=(info_chunk([(b"ICMT", b"Gr\xfcn und Blau")]),)),
    )
    metadata = parse(path)
    assert values_of(metadata, "info_comment") == ["Grün und Blau"]
    assert any("cp1252" in note for note in metadata.notes)


def test_list_adtl_labels_are_not_imported(tmp_path):
    """adtl 的 labl 是时间轴标记（Tempo: 120.0），不是素材元数据。"""
    path = write_file(
        tmp_path / "adtl.wav",
        wave_file(
            after_data=(
                info_chunk([(b"labl", b"\x01\x00\x00\x00Tempo: 120.0")], list_type=b"adtl"),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.NO_METADATA
    assert metadata.values == ()
    assert any("adtl" in note for note in metadata.notes)


def test_odd_sized_info_value_keeps_the_next_chunk_aligned(tmp_path):
    """奇数长度子块后的填充字节必须跳过，否则 iXML 会被错位解析。"""
    path = write_file(
        tmp_path / "odd.wav",
        wave_file(
            after_data=(
                info_chunk([(b"ICMT", b"abc")]),  # 3 字节正文 -> 1 字节填充
                ixml_chunk("<BWFXML><PROJECT>Odd</PROJECT></BWFXML>"),
            )
        ),
    )
    metadata = parse(path)
    assert values_of(metadata, "info_comment") == ["abc"]
    assert values_of(metadata, "ixml_project") == ["Odd"]


# ---------------------------------------------------------------------------- iXML
def test_ixml_production_fields_are_not_promoted_to_description(tmp_path):
    """工程名/场次/轨道名是制作标识，不能当音效描述。"""
    path = write_file(
        tmp_path / "ixml.wav",
        wave_file(
            after_data=(
                ixml_chunk(
                    "<?xml version='1.0' encoding='UTF-8'?>"
                    "<BWFXML><IXML_VERSION>1.52</IXML_VERSION>"
                    "<PROJECT>Forest Ambience</PROJECT><SCENE>sc01</SCENE>"
                    "<TAKE>12</TAKE><TAPE>T001</TAPE>"
                    "<NOTE>careful with the close mic</NOTE>"
                    "<TRACK_LIST><TRACK><NAME>Boom L</NAME></TRACK>"
                    "<TRACK><NAME>Boom R</NAME></TRACK></TRACK_LIST>"
                    "</BWFXML>"
                ),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.OK
    assert metadata.description is None
    assert values_of(metadata, "ixml_project") == ["Forest Ambience"]
    assert values_of(metadata, "ixml_scene") == ["sc01"]
    assert values_of(metadata, "ixml_take") == ["12"]
    assert values_of(metadata, "ixml_tape") == ["T001"]
    assert values_of(metadata, "ixml_version") == ["1.52"]
    assert values_of(metadata, "ixml_track_name") == ["Boom L", "Boom R"]
    assert [value.value for value in metadata.values if value.role == ROLE_KEYWORD] == [
        "careful with the close mic"
    ]


def test_ixml_with_doctype_is_refused(tmp_path):
    """iXML 规范里没有 DTD；出现 DTD/实体声明就直接拒绝，避免外部实体解析。"""
    path = write_file(
        tmp_path / "dtd.wav",
        wave_file(
            after_data=(
                ixml_chunk(
                    "<?xml version='1.0'?><!DOCTYPE BWFXML [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>"
                    "<BWFXML><NOTE>&x;</NOTE></BWFXML>"
                ),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.values == ()
    assert metadata.status is EmbeddedStatus.MALFORMED
    assert any("DOCTYPE" in note for note in metadata.notes)


def test_broken_xml_is_reported_without_partial_values(tmp_path):
    path = write_file(
        tmp_path / "broken.wav",
        wave_file(after_data=(ixml_chunk("<BWFXML><PROJECT>Unclosed"),)),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.MALFORMED
    assert metadata.values == ()
    assert any("XML 解析失败" in note for note in metadata.notes)


def test_broken_xml_does_not_hide_a_good_bext(tmp_path):
    """一个块坏了不该让另一个块的有效字段消失，但坏块要留下说明。"""
    path = write_file(
        tmp_path / "mixed.wav",
        wave_file(
            after_data=(
                bext_chunk(description="Good text."),
                ixml_chunk("<BWFXML><PROJECT>Unclosed"),
            )
        ),
    )
    metadata = parse(path)
    assert metadata.status is EmbeddedStatus.PARTIAL
    assert metadata.description == "Good text."
    assert any("XML 解析失败" in note for note in metadata.notes)
    # bext 完整读到；LIST/INFO 文件里确实没有，完整扫描确认缺失也算覆盖。
    # iXML 坏了，所以只有它能挡住旧值被删。
    assert metadata.family_coverage("bext") is Coverage.COMPLETE
    assert metadata.family_coverage("info") is Coverage.COMPLETE
    assert metadata.family_coverage("ixml") is Coverage.PARTIAL
    assert metadata.replaced_families() == frozenset({"bext", "info"})


# ---------------------------------------------------------------- 长度、截断与读取量
def test_truncated_metadata_chunk_is_reported(tmp_path):
    """chunk 声明长度超出文件末尾：按截断读取并说明，不能错位到下一块。"""
    payload = wave_file(after_data=(bext_chunk(description="Truncated but readable."),))
    # 砍掉 chunk 尾部：声明长度仍然写着完整长度。
    truncated = payload[:-120]
    path = write_file(tmp_path / "truncated.wav", truncated)
    metadata = parse(path)
    assert metadata.description == "Truncated but readable."
    # 读到了内容，但这一来源没有完整覆盖：只能是 partial，不能报 ok。
    assert metadata.status is EmbeddedStatus.PARTIAL
    assert metadata.family_coverage("bext") is Coverage.PARTIAL
    assert "bext" not in metadata.replaced_families()
    assert any("只剩" in note for note in metadata.notes)


def test_declared_size_past_eof_is_not_glued_to_the_next_chunk(tmp_path):
    """一个超长 chunk 后面即使跟着真实块，也不能把两者拼起来解析。"""
    body = struct.pack("<I", 64) + b"short body"
    bogus = b"LIST" + body
    payload = wave_file(after_data=(bogus, ixml_chunk("<BWFXML><PROJECT>Never</PROJECT></BWFXML>")))
    path = write_file(tmp_path / "past_eof.wav", payload)
    metadata = parse(path)
    assert values_of(metadata, "ixml_project") == []
    # 越界长度让扫描提前结束：没有来源被完整覆盖，不能授权删除旧值。
    assert metadata.status is EmbeddedStatus.PARTIAL
    assert metadata.replaced_families() == frozenset()
    assert metadata.family_coverage("info") is Coverage.PARTIAL


def test_data_chunk_is_skipped_not_read(tmp_path):
    """大文件也只读头部元数据：读取量必须远小于文件体积。"""

    class CountingStream(io.BytesIO):
        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self.read_bytes = 0

        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            self.read_bytes += len(data)
            return data

    payload = wave_file(
        frames=48000 * 10,  # 10 秒 16bit mono ≈ 960 KB 的 data
        after_data=(bext_chunk(description="Skip the audio."),),
    )
    stream = CountingStream(payload)
    metadata = embedded.parse_embedded_stream(stream, file_size=len(payload))
    assert metadata.description == "Skip the audio."
    assert metadata.chunks == ("fmt ", "data", "bext")
    # 只读了头部和 602 字节的 bext；音频正文一个字节都没读。
    assert stream.read_bytes < 4096
    assert stream.read_bytes < len(payload) // 100


def test_oversized_metadata_chunk_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(embedded, "MAX_METADATA_CHUNK_BYTES", 64)
    path = write_file(
        tmp_path / "big.wav",
        wave_file(after_data=(bext_chunk(description="x" * 200),)),
    )
    metadata = parse(path)
    assert metadata.values == ()
    assert metadata.status is EmbeddedStatus.MALFORMED
    assert any("超过上限" in note for note in metadata.notes)


def test_placeholder_data_size_stops_the_walk_responsibly(tmp_path):
    """data 长度是占位值（0xFFFFFFFF）时，不能假设其后还有可定位的块。"""
    body = b"WAVE" + fmt_chunk() + b"data" + struct.pack("<I", 0xFFFFFFFF) + b"\x00" * 32
    payload = b"RIFF" + struct.pack("<I", len(body)) + body
    path = write_file(tmp_path / "stream.wav", payload)
    metadata = parse(path)
    # 扫描提前结束且一无所获：不能冒充"没有元数据"，也没有任何来源可被替换。
    assert metadata.status is EmbeddedStatus.PARTIAL
    assert metadata.replaced_families() == frozenset()
    assert all(coverage is Coverage.NONE for coverage in metadata.coverage.values())
    assert any("占位值" in note for note in metadata.notes)
    assert metadata.chunks[-1] == "data"


def test_bext_after_data_is_still_found(tmp_path):
    """bext 在 data 之后（真实样本就是这样）也必须读到。"""
    path = write_file(
        tmp_path / "after.wav",
        wave_file(before_data=(), after_data=(bext_chunk(description="After the audio."),)),
    )
    assert parse(path).description == "After the audio."


def test_data_chunk_absent_is_reported(tmp_path):
    body = b"WAVE" + fmt_chunk() + bext_chunk(description="No audio here.")
    payload = b"RIFF" + struct.pack("<I", len(body)) + body
    path = write_file(tmp_path / "nodata.wav", payload)
    metadata = parse(path)
    # 解析器只需要 chunk 表；没有 data 的容器仍然能读出元数据。
    assert metadata.description == "No audio here."


def test_chunk_count_limit_stops_a_pathological_file(tmp_path, monkeypatch):
    monkeypatch.setattr(embedded, "MAX_CHUNKS", 3)
    noise = tuple(chunk(b"junk", b"") for _ in range(10))
    path = write_file(tmp_path / "noisy.wav", wave_file(before_data=noise))
    metadata = parse(path)
    # 扫描被上限截断、什么都没读到：这是"没扫完"，不是"元数据不存在"。
    assert metadata.status is EmbeddedStatus.PARTIAL
    assert metadata.replaced_families() == frozenset()
    assert any("上限" in note for note in metadata.notes)


def test_reader_accepts_a_path_and_a_stream(tmp_path):
    payload = wave_file(after_data=(bext_chunk(description="Same both ways."),))
    path = write_file(tmp_path / "both.wav", payload)
    from_path = read_embedded_metadata(path)
    from_stream = embedded.parse_embedded_stream(io.BytesIO(payload), file_size=len(payload))
    assert from_path.as_dict() == from_stream.as_dict()


@pytest.mark.parametrize(
    ("chunk_id", "expected"),
    [
        (b"bext", "bext"),
        (b"LIST", "LIST"),
        (b"iXML", "iXML"),
        (b"\x01\x02\x03\x04", "01020304"),
    ],
)
def test_chunk_names_are_readable(chunk_id, expected):
    assert embedded._chunk_name(chunk_id) == expected
