"""Cover resolution: the MediaEye priority chain and its directory walk."""

from __future__ import annotations

import struct
from pathlib import Path

from audiosearch.audition.covers import CoverResolver, embedded_picture, find_folder_image

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 24
JPG = b"\xff\xd8\xff" + b"1" * 24


def make_wav(path: Path, *, frames: int = 480) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x00\x00" * frames
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
    data = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(data)) + data)
    return path


def embed_id3_png(
    path: Path,
    picture: bytes = PNG,
    *,
    chunk_id: bytes = b"id3 ",
    odd_padding_junk: bool = False,
) -> Path:
    """Append an ``id3 `` / ``ID3 `` chunk carrying a v2.3 APIC frame."""
    apic = (
        b"\x00"
        + b"image/png\x00"
        + b"\x03"
        + b"\x00"
        + picture
    )
    frame = b"APIC" + struct.pack(">I", len(apic)) + b"\x00\x00" + apic
    body = frame
    size = len(body)
    syncsafe = bytes(((size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F))
    tag = b"ID3\x03\x00\x00" + syncsafe + body
    raw = path.read_bytes()
    extra = b""
    if odd_padding_junk:
        extra = b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
    chunk = chunk_id + struct.pack("<I", len(tag)) + tag + (b"\x00" if len(tag) % 2 else b"")
    riff_size = struct.unpack("<I", raw[4:8])[0] + len(extra) + len(chunk)
    path.write_bytes(raw[:4] + struct.pack("<I", riff_size) + raw[8:] + extra + chunk)
    return path


def test_a_named_cover_wins_over_a_loose_image(tmp_path):
    audio = make_wav(tmp_path / "Pack" / "Sounds" / "hit.wav")
    (audio.parent / "random.png").write_bytes(JPG)
    (audio.parent / "cover.png").write_bytes(PNG)

    cover = CoverResolver().resolve(audio)

    assert cover is not None
    assert cover.source == "folder-named"
    assert cover.data == PNG
    assert cover.media_type == "image/png"


def test_an_embedded_picture_beats_a_loose_image(tmp_path):
    audio = embed_id3_png(make_wav(tmp_path / "Pack" / "hit.wav"))
    (audio.parent / "contact.png").write_bytes(JPG)

    cover = CoverResolver().resolve(audio)

    assert cover is not None
    assert cover.source == "embedded"
    assert cover.data == PNG


def test_a_named_folder_image_beats_the_embedded_picture(tmp_path):
    audio = embed_id3_png(make_wav(tmp_path / "Pack" / "hit.wav"))
    (audio.parent / "folder.jpg").write_bytes(JPG)

    cover = CoverResolver().resolve(audio)

    assert cover is not None
    assert cover.source == "folder-named"
    assert cover.data == JPG
    assert cover.media_type == "image/jpeg"


def test_an_artwork_folder_two_levels_up_is_found(tmp_path):
    audio = make_wav(tmp_path / "Pack" / "Sounds" / "Sub" / "hit.wav")
    artwork = tmp_path / "Pack" / "Artwork"
    artwork.mkdir(parents=True)
    (artwork / "pack.png").write_bytes(PNG)

    cover = CoverResolver().resolve(audio)

    assert cover is not None
    assert cover.data == PNG


def test_a_sibling_pack_cover_is_not_taken(tmp_path):
    audio = make_wav(tmp_path / "PackA" / "Sounds" / "hit.wav")
    other = tmp_path / "PackB"
    other.mkdir(parents=True)
    (other / "cover.png").write_bytes(PNG)

    cover = CoverResolver().resolve(audio)

    assert cover is None, "同一个父目录下另一个素材包的封面不属于这一条"


def test_nothing_nearby_resolves_to_none(tmp_path):
    audio = make_wav(tmp_path / "Pack" / "hit.wav")
    assert CoverResolver().resolve(audio) is None


def test_the_directory_scan_is_cached(tmp_path, monkeypatch):
    audio = make_wav(tmp_path / "Pack" / "one.wav")
    make_wav(tmp_path / "Pack" / "two.wav")
    (audio.parent / "cover.png").write_bytes(PNG)
    calls = {"n": 0}
    import audiosearch.audition.covers as covers_module

    original = covers_module.find_folder_image

    def counting(path):
        calls["n"] += 1
        return original(path)

    monkeypatch.setattr(covers_module, "find_folder_image", counting)
    resolver = CoverResolver()
    resolver.resolve(audio)
    resolver.resolve(audio.parent / "two.wav")

    assert calls["n"] == 1, "同一目录的第二个文件不应再扫一遍"


def test_embedded_picture_ignores_a_plain_file(tmp_path):
    audio = make_wav(tmp_path / "Pack" / "hit.wav")
    assert embedded_picture(audio) is None


def test_find_folder_image_reports_whether_the_name_claims_a_cover(tmp_path):
    audio = make_wav(tmp_path / "Pack" / "hit.wav")
    (audio.parent / "hit.png").write_bytes(PNG)

    path, named = find_folder_image(audio)

    assert path is not None and path.name == "hit.png"
    assert named is False, "同名图只是兜底，不是命名封面"


def test_uppercase_id3_chunk_and_odd_padding_in_wav(tmp_path):
    audio = embed_id3_png(
        make_wav(tmp_path / "Pack" / "khron.wav"),
        picture=JPG,
        chunk_id=b"ID3 ",
        odd_padding_junk=True,
    )
    cover = CoverResolver().resolve(audio)
    assert cover is not None
    assert cover.source == "embedded"
    assert cover.data == JPG


def test_rf64_wav_with_ds64_and_aiff_embedded_id3(tmp_path):
    # RF64 with ds64 defining true data size so seeking reaches trailing ID3 chunk
    payload = b"\x00\x00" * 240
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
    apic = b"\x00image/jpeg\x00\x03\x00" + JPG
    frame = b"APIC" + struct.pack(">I", len(apic)) + b"\x00\x00" + apic
    tag = (
        b"ID3\x03\x00\x00"
        + bytes(((len(frame) >> 21) & 0x7F, (len(frame) >> 14) & 0x7F, (len(frame) >> 7) & 0x7F, len(frame) & 0x7F))
        + frame
    )
    id3_chunk = b"ID3 " + struct.pack("<I", len(tag)) + tag + (b"\x00" if len(tag) % 2 else b"")
    data_chunk = b"data" + struct.pack("<I", 0xFFFFFFFF) + payload
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    ds64_body = struct.pack("<QQQI", 128, len(payload), 240, 0)
    ds64_chunk = b"ds64" + struct.pack("<I", len(ds64_body)) + ds64_body
    rf64 = tmp_path / "big.wav"
    rf64.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64_chunk + fmt_chunk + data_chunk + id3_chunk)

    cov_rf64 = CoverResolver().resolve(rf64)
    assert cov_rf64 is not None
    assert cov_rf64.source == "embedded"
    assert cov_rf64.data == JPG

    # AIFF with big-endian ID3 chunk
    ssnd = b"SSND" + struct.pack(">I", 8) + b"\x00" * 8
    aiff_id3 = b"ID3 " + struct.pack(">I", len(tag)) + tag + (b"\x00" if len(tag) % 2 else b"")
    aiff_body = b"AIFF" + ssnd + aiff_id3
    aiff = tmp_path / "pack.aiff"
    aiff.write_bytes(b"FORM" + struct.pack(">I", len(aiff_body)) + aiff_body)

    cov_aiff = CoverResolver().resolve(aiff)
    assert cov_aiff is not None
    assert cov_aiff.source == "embedded"
    assert cov_aiff.data == JPG


def test_id3v2_unsynchronised_and_empty_mime_magic_sniffing(tmp_path):
    # 1. ID3v2.3 tag-level unsync (flags=0x80): \xff\x00 in payload restores to \xff
    apic_unsync = b"\x00image/jpeg\x00\x03\x00" + b"\xff\x00\xd8\xff\x00" + b"J" * 16
    frame_u = b"APIC" + struct.pack(">I", len(apic_unsync)) + b"\x00\x00" + apic_unsync
    sz_u = len(frame_u)
    sync_u = bytes(((sz_u >> 21) & 0x7F, (sz_u >> 14) & 0x7F, (sz_u >> 7) & 0x7F, sz_u & 0x7F))
    tag_u = b"ID3\x03\x00\x80" + sync_u + frame_u
    wav_u = make_wav(tmp_path / "Pack" / "unsync.wav")
    raw = wav_u.read_bytes()
    chunk_u = b"ID3 " + struct.pack("<I", len(tag_u)) + tag_u + (b"\x00" if len(tag_u) % 2 else b"")
    riff_sz = struct.unpack("<I", raw[4:8])[0] + len(chunk_u)
    wav_u.write_bytes(raw[:4] + struct.pack("<I", riff_sz) + raw[8:] + chunk_u)

    cov_u = CoverResolver().resolve(wav_u)
    assert cov_u is not None
    assert cov_u.media_type == "image/jpeg"
    assert cov_u.data == b"\xff\xd8\xff" + b"J" * 16

    # 2. Empty MIME string in APIC (detected via magic bytes just like MediaEye)
    apic_empty = b"\x00\x00\x03\x00" + PNG
    frame_e = b"APIC" + struct.pack(">I", len(apic_empty)) + b"\x00\x00" + apic_empty
    sz_e = len(frame_e)
    sync_e = bytes(((sz_e >> 21) & 0x7F, (sz_e >> 14) & 0x7F, (sz_e >> 7) & 0x7F, sz_e & 0x7F))
    tag_e = b"ID3\x03\x00\x00" + sync_e + frame_e
    wav_e = make_wav(tmp_path / "Pack" / "empty_mime.wav")
    raw_e = wav_e.read_bytes()
    chunk_e = b"ID3 " + struct.pack("<I", len(tag_e)) + tag_e + (b"\x00" if len(tag_e) % 2 else b"")
    riff_sz_e = struct.unpack("<I", raw_e[4:8])[0] + len(chunk_e)
    wav_e.write_bytes(raw_e[:4] + struct.pack("<I", riff_sz_e) + raw_e[8:] + chunk_e)

    cov_e = CoverResolver().resolve(wav_e)
    assert cov_e is not None
    assert cov_e.media_type == "image/png"
    assert cov_e.data == PNG
