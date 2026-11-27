"""Behavioral tests for ReaperSpotClient file-channel protocol."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from audiosearch.browser.reaper_spot import SPOT_PROTOCOL, ReaperSpotClient
from audiosearch.integrations.reaper.channel import atomic_write, decode_message, encode_message


def test_reaper_spot_status_fresh_stale_and_closed(tmp_path: Path):
    client = ReaperSpotClient(tmp_path / "spot", heartbeat_ttl=2.0)
    assert client.status().connected is False

    client.channel_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        client.heartbeat_path,
        encode_message(
            {
                "protocol": SPOT_PROTOCOL,
                "epoch": time.time(),
                "project": "测试工程.rpp",
                "track_name": "SFX_01",
                "cursor": 12.5,
            }
        ),
    )
    st = client.status()
    assert st.connected is True
    assert st.project_name == "测试工程.rpp"
    assert st.track_name == "SFX_01"
    assert abs(st.cursor_sec - 12.5) < 1e-6

    # Closed protocol or stale timestamp => disconnected
    atomic_write(
        client.heartbeat_path,
        encode_message({"protocol": "closed", "epoch": time.time()}),
    )
    assert client.status().connected is False

    atomic_write(
        client.heartbeat_path,
        encode_message({"protocol": SPOT_PROTOCOL, "epoch": time.time() - 30.0}),
    )
    # Also backdate mtime so mtime fallback sees it as stale
    old = time.time() - 30.0
    import os

    os.utime(client.heartbeat_path, (old, old))
    assert client.status().connected is False


def test_reaper_spot_roundtrip_chinese_path_and_timeout_cleanup(tmp_path: Path):
    channel = tmp_path / "spot"
    client = ReaperSpotClient(channel, heartbeat_ttl=2.0)
    wav = tmp_path / "中文音效_火球.wav"
    wav.write_bytes(b"RIFF")

    captured: list[dict[str, str]] = []
    stop_flag = threading.Event()

    def fake_lua_worker() -> None:
        inbox = channel / "inbox"
        outbox = channel / "outbox"
        while not stop_flag.is_set():
            if inbox.exists():
                for cmd_file in sorted(inbox.glob("*.cmd")):
                    fields = decode_message(cmd_file.read_bytes())
                    if fields is None:
                        continue
                    cmd_file.unlink(missing_ok=True)
                    captured.append(fields)
                    rid = fields["request_id"]
                    atomic_write(
                        outbox / f"{rid}.rcpt",
                        encode_message(
                            {
                                "protocol": SPOT_PROTOCOL,
                                "request_id": rid,
                                "ok": "1",
                                "message": "ok",
                                "track_name": "SFX Main",
                                "position": 4.25,
                                "length": 0.8,
                            }
                        ),
                    )
            time.sleep(0.01)

    worker = threading.Thread(target=fake_lua_worker, daemon=True)
    worker.start()
    try:
        res = client.spot(
            wav,
            start_sec=0.1,
            end_sec=0.9,
            pitch_semitones=-3.5,
            rate=1.25,
            preserve_pitch=True,
            apply_tone=True,
            move_cursor=True,
            timeout=1.5,
        )
    finally:
        stop_flag.set()
        worker.join(timeout=1.0)

    assert res.ok is True
    assert res.track_name == "SFX Main"
    assert abs(res.position - 4.25) < 1e-6
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd["path"] == str(wav)
    assert float(cmd["start_sec"]) == 0.1
    assert float(cmd["end_sec"]) == 0.9
    assert float(cmd["pitch_semitones"]) == -3.5
    assert float(cmd["rate"]) == 1.25
    assert cmd["preserve_pitch"] == "1"
    assert cmd["apply_tone"] == "1"
    assert cmd["take_name"] == "中文音效_火球.wav"

    # Without a worker running, timeout cleans up the .cmd file cleanly
    res_timeout = client.spot(wav, timeout=0.08)
    assert res_timeout.ok is False
    assert list((channel / "inbox").glob("*.cmd")) == []


def test_reaper_spot_handles_permission_error_gracefully(tmp_path: Path, monkeypatch):
    from audiosearch.browser import reaper_spot

    client = reaper_spot.ReaperSpotClient(tmp_path / "spot", heartbeat_ttl=2.0)
    wav = tmp_path / "kick.wav"
    wav.write_bytes(b"RIFF")

    def _deny(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(tmp_path / "cmd.tmp"))

    monkeypatch.setattr(reaper_spot, "atomic_write", _deny)
    res = client.spot(wav)
    assert res.ok is False
    assert "无权限写入" in res.message
