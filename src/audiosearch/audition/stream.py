"""Low-latency streaming decoder with polyphase/cubic resampling, loop, and volume."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audiosearch.audioio.decode import open_audio_file
from audiosearch.audioio.resample import _design, _ratio
from audiosearch.audition.player import AudioSink

STREAM_ZERO_CROSSINGS = 12
STREAM_KAISER_BETA = 6.0
FIRST_CHUNK_SECONDS = 0.06
CHUNK_SECONDS = 0.04
MAX_BUFFERED_SECONDS = 0.16


def _filter_window(
    window_src: np.ndarray,
    up: int,
    down: int,
    bank: np.ndarray,
    out_frames: int,
) -> np.ndarray:
    """Apply polyphase ``bank`` over a pre-padded ``(past + in + future, channels)`` buffer."""
    rows = bank.shape[0]
    channels = window_src.shape[1]
    budget = max(1, (32 << 20) // max(1, rows * channels * 8))
    block = max(1, min(1 << 17, budget, out_frames))
    offsets = np.arange(rows, dtype=np.int64)
    src = window_src.astype(np.float64, copy=False)
    out = np.zeros((out_frames, channels), dtype=np.float64)
    for start in range(0, out_frames, block):
        stop = min(out_frames, start + block)
        positions = np.arange(start, stop, dtype=np.int64) * down
        base = positions // up
        phase = positions - base * up
        window = src[base[:, np.newaxis] + offsets[np.newaxis, :]]
        taps = bank[:, phase].astype(np.float64).T
        out[start:stop] = np.einsum("brc,br->bc", window, taps, optimize=True)
    return out.astype(np.float32)


class StreamingResampler:
    """Block-wise polyphase resampler with overlap-save state (zero boundary clicks)."""

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        channels: int,
        *,
        zero_crossings: int = STREAM_ZERO_CROSSINGS,
        beta: float = STREAM_KAISER_BETA,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0 or channels <= 0:
            raise ValueError("rates and channels must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.channels = channels
        self.passthrough = src_rate == dst_rate
        if not self.passthrough:
            self._up, self._down = _ratio(src_rate, dst_rate)
            self._bank, self._past, self._future = _design(
                self._up, self._down, zero_crossings, beta
            )
            self._history = np.zeros((self._past, channels), dtype=np.float32)
            self._pending = np.zeros((0, channels), dtype=np.float32)
        else:
            self._up = self._down = 1
            self._past = self._future = 0

    def process(self, block: np.ndarray, *, eof: bool = False) -> np.ndarray:
        """Resample ``block`` ``(frames, channels)`` float32; flush tail when ``eof=True``."""
        data = np.asarray(block, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if self.passthrough:
            return np.ascontiguousarray(data, dtype=np.float32)

        if data.shape[0] > 0:
            self._pending = (
                data if self._pending.shape[0] == 0 else np.concatenate([self._pending, data], axis=0)
            )

        outputs: list[np.ndarray] = []
        avail = self._pending.shape[0]
        if avail >= self._down + self._future:
            steps = (avail - self._future) // self._down
            in_consume = steps * self._down
            out_frames = steps * self._up
            window_src = np.concatenate(
                [self._history, self._pending[: in_consume + self._future]], axis=0
            )
            outputs.append(
                _filter_window(window_src, self._up, self._down, self._bank, out_frames)
            )
            if self._past > 0:
                self._history = window_src[in_consume : in_consume + self._past].copy()
            self._pending = self._pending[in_consume:]

        if eof and self._pending.shape[0] > 0:
            rem = self._pending.shape[0]
            out_frames = int(math.ceil(rem * self._up / self._down))
            pad_tail = np.zeros((self._future, self.channels), dtype=np.float32)
            window_src = np.concatenate([self._history, self._pending, pad_tail], axis=0)
            outputs.append(
                _filter_window(window_src, self._up, self._down, self._bank, out_frames)
            )
            self._pending = np.zeros((0, self.channels), dtype=np.float32)
            self._history = np.zeros((self._past, self.channels), dtype=np.float32)

        if not outputs:
            return np.zeros((0, self.channels), dtype=np.float32)
        if len(outputs) == 1:
            return np.ascontiguousarray(outputs[0], dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)


class FractionalResampler:
    """Streaming 4-point Catmull-Rom cubic varispeed resampler for pitch/rate audition."""

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        channels: int,
        *,
        speed: float = 1.0,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0 or channels <= 0 or speed <= 0:
            raise ValueError("rates, channels, and speed must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.channels = channels
        self.speed = float(speed)
        self._step = (src_rate / dst_rate) * self.speed
        # Seed with 1 zero sample at index 0 so index 1 corresponds to input sample 0.
        self._buf = np.zeros((1, channels), dtype=np.float32)
        self._pos = 1.0

    def set_speed(self, speed: float) -> None:
        """Update playback speed ratio in-place without dropping buffered state."""
        clamped = max(0.0625, min(float(speed), 16.0))
        self.speed = clamped
        self._step = (self.src_rate / self.dst_rate) * clamped

    def process(self, block: np.ndarray, *, eof: bool = False) -> np.ndarray:
        data = np.asarray(block, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.shape[0] > 0:
            self._buf = np.concatenate([self._buf, data], axis=0)

        if eof:
            # Pad 2 zero samples at tail so p2/p3 stay valid up to the last input frame.
            pad = np.zeros((2, self.channels), dtype=np.float32)
            work = np.concatenate([self._buf, pad], axis=0)
            max_pos = float(work.shape[0] - 3)
        else:
            work = self._buf
            max_pos = float(work.shape[0] - 3)

        if max_pos < self._pos or work.shape[0] < 4:
            if eof:
                self._buf = np.zeros((1, self.channels), dtype=np.float32)
                self._pos = 1.0
            return np.zeros((0, self.channels), dtype=np.float32)

        n_out = int(math.floor((max_pos - self._pos) / self._step)) + 1
        if n_out <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        positions = self._pos + np.arange(n_out, dtype=np.float64) * self._step
        idx = np.floor(positions).astype(np.int64)
        t = (positions - idx)[:, np.newaxis]
        t2 = t * t
        t3 = t2 * t

        p0 = work[idx - 1].astype(np.float64)
        p1 = work[idx].astype(np.float64)
        p2 = work[idx + 1].astype(np.float64)
        p3 = work[idx + 2].astype(np.float64)

        out = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

        self._pos = self._pos + n_out * self._step
        if eof:
            self._buf = np.zeros((1, self.channels), dtype=np.float32)
            self._pos = 1.0
        else:
            drop = max(int(math.floor(self._pos)) - 1, 0)
            if drop > 0:
                self._buf = self._buf[drop:]
                self._pos -= drop

        return np.ascontiguousarray(out, dtype=np.float32)


class StreamingWSOLA:
    """Low-latency pure-NumPy streaming WSOLA time-scaler with 50% periodic Hann COLA."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        win_ms: float = 36.0,
        search_ms: float = 10.0,
    ) -> None:
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        self.sr = int(sample_rate)
        self.ch = int(channels)
        self.win = max(64, int(round(self.sr * win_ms / 1000.0)) // 2 * 2)
        self.hop_out = self.win // 2
        self.search = max(16, int(round(self.sr * search_ms / 1000.0)))
        # Periodic Hann sums to exactly 1.0 at 50% overlap.
        idx = np.arange(self.win, dtype=np.float32)
        self.hann = (0.5 * (1.0 - np.cos(2.0 * np.pi * idx / self.win))).astype(np.float32)[:, None]
        self._buf = np.zeros((0, self.ch), dtype=np.float32)
        self._read_pos = 0.0
        self._ola_tail = np.zeros((self.hop_out, self.ch), dtype=np.float32)
        self._ref_tail = np.zeros((self.hop_out,), dtype=np.float32)
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def has_pending(self) -> bool:
        return self._started or self._buf.shape[0] > 0

    def reset(self) -> None:
        self._buf = np.zeros((0, self.ch), dtype=np.float32)
        self._read_pos = 0.0
        self._ola_tail.fill(0.0)
        self._ref_tail.fill(0.0)
        self._started = False

    def _emit_ready(self, stretch_factor: float, *, final: bool = False) -> np.ndarray:
        hop_in = max(1.0, self.hop_out * max(0.0625, min(float(stretch_factor), 16.0)))
        real_len = self._buf.shape[0]
        chunks: list[np.ndarray] = []

        while True:
            base = int(round(self._read_pos))
            lo = max(0, base - self.search)
            hi = base + self.search
            if not final and hi + self.win > real_len:
                break
            if final:
                if base >= real_len:
                    break
                hi = min(hi, max(lo, real_len - 1))
                need = max(hi, base) + self.win - self._buf.shape[0]
                if need > 0:
                    self._buf = np.concatenate(
                        [self._buf, np.zeros((need, self.ch), dtype=np.float32)], axis=0
                    )

            if not self._started or final:
                best = base
                self._started = True
            else:
                seg = self._buf[lo : hi + self.hop_out].mean(axis=1)
                n_cand = hi - lo + 1
                if n_cand > 1 and np.any(self._ref_tail):
                    seg2 = seg[::2]
                    ref2 = self._ref_tail[::2]
                    m = len(ref2)
                    if len(seg2) >= m:
                        wins = np.lib.stride_tricks.sliding_window_view(seg2, m)
                        dots = wins @ ref2
                        norms = np.sqrt(np.sum(wins * wins, axis=1) + 1e-12)
                        coarse = int(np.argmax(dots / norms)) * 2
                        r_lo = max(0, coarse - 2)
                        r_hi = min(n_cand - 1, coarse + 2)
                        sub = np.lib.stride_tricks.sliding_window_view(
                            seg[r_lo : r_hi + self.hop_out], self.hop_out
                        )
                        s_dots = sub @ self._ref_tail
                        s_norms = np.sqrt(np.sum(sub * sub, axis=1) + 1e-12)
                        best = lo + r_lo + int(np.argmax(s_dots / s_norms))
                    else:
                        best = base
                else:
                    best = base

            frame = self._buf[best : best + self.win]
            windowed = frame * self.hann
            out = self._ola_tail + windowed[: self.hop_out]
            chunks.append(out)
            self._ola_tail = windowed[self.hop_out :].copy()

            nat = best + self.hop_out
            if nat + self.hop_out <= self._buf.shape[0]:
                self._ref_tail = self._buf[nat : nat + self.hop_out].mean(axis=1).copy()
            else:
                self._ref_tail = frame[self.hop_out :].mean(axis=1).copy()

            self._read_pos += hop_in
            if not final:
                drop = max(0, int(self._read_pos) - self.search - self.win)
                if drop > 8192:
                    self._buf = self._buf[drop:]
                    self._read_pos -= drop
                    real_len = self._buf.shape[0]

        if final:
            if self._started and np.any(self._ola_tail):
                chunks.append(self._ola_tail.copy())
            self.reset()

        if not chunks:
            return np.zeros((0, self.ch), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    def process(self, frames: np.ndarray, stretch_factor: float) -> np.ndarray:
        data = np.asarray(frames, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.shape[0] > 0:
            self._buf = (
                data if self._buf.shape[0] == 0 else np.concatenate([self._buf, data], axis=0)
            )
        return self._emit_ready(stretch_factor, final=False)

    def flush(self, stretch_factor: float) -> np.ndarray:
        return self._emit_ready(stretch_factor, final=True)


@dataclass(slots=True)
class _QueuedChunk:
    samples: np.ndarray
    orig_start_sec: float
    orig_sec_per_frame: float


class StreamingPlayer:
    """Streams decoded PCM chunks from disk with loop, selection, volume, pitch & rate."""

    def __init__(self, sink: AudioSink, target_rate_hz: int) -> None:
        self._sink = sink
        self._target_rate = target_rate_hz
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._chunks: deque[_QueuedChunk] = deque()
        self._buffered_frames = 0
        self._max_buffer_frames = int(MAX_BUFFERED_SECONDS * target_rate_hz)
        self._decode_eof = False
        self._state = "empty"  # empty | playing | paused | finished
        self._position_seconds = 0.0
        self._generation = 0
        self._path: Path | None = None
        self._selection: tuple[float, float] | None = None
        self._loop = False
        self._volume_gain = 1.0
        self._pitch_semitones = 0.0
        self._rate = 1.0
        self._preserve_pitch = False
        self._stop_event: threading.Event | None = None
        self._closed = False
        self._callbacks: tuple[
            Callable[[int, float], None] | None,
            Callable[[int, str], None] | None,
            Callable[[int], None] | None,
        ] = (None, None, None)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def position_seconds(self) -> float:
        with self._lock:
            return self._position_seconds

    @property
    def volume_gain(self) -> float:
        with self._lock:
            return self._volume_gain

    @property
    def loop(self) -> bool:
        with self._lock:
            return self._loop

    @property
    def pitch_semitones(self) -> float:
        with self._lock:
            return self._pitch_semitones

    @property
    def rate(self) -> float:
        with self._lock:
            return self._rate

    @property
    def preserve_pitch(self) -> bool:
        with self._lock:
            return self._preserve_pitch

    @property
    def has_buffered_audio(self) -> bool:
        with self._lock:
            return self._buffered_frames > 0 or not self._decode_eof

    @property
    def selection(self) -> tuple[float, float] | None:
        with self._lock:
            return self._selection

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def set_volume_gain(self, gain: float) -> None:
        with self._lock:
            self._volume_gain = max(0.0, min(float(gain), 4.0))

    def set_loop(self, enabled: bool) -> None:
        with self._cond:
            self._loop = bool(enabled)
            self._cond.notify_all()

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        with self._cond:
            self._selection = selection
            self._cond.notify_all()

    def set_preserve_pitch(self, enabled: bool) -> None:
        """Toggle independent Pitch/Rate mode in real time without restarting decoder."""
        with self._lock:
            self._preserve_pitch = bool(enabled)

    def set_pitch_rate(self, pitch_semitones: float, rate: float) -> None:
        """Update pitch (semitones) and rate in real time without restarting decoder."""
        with self._lock:
            self._pitch_semitones = max(-24.0, min(float(pitch_semitones), 24.0))
            self._rate = max(0.25, min(float(rate), 4.0))

    def _effective_speed(self) -> float:
        with self._lock:
            return self._rate * (2.0 ** (self._pitch_semitones / 12.0))

    def load_and_play(
        self,
        path: Path,
        start_seconds: float = 0.0,
        *,
        end_seconds: float | None = None,
        on_ready: Callable[[int, float], None] | None = None,
        on_error: Callable[[int, str], None] | None = None,
        on_finished: Callable[[int], None] | None = None,
    ) -> int:
        """Cancel any active stream and start streaming ``path`` from ``start_seconds``."""
        self._sink.stop()
        with self._cond:
            if self._stop_event is not None:
                self._stop_event.set()
            self._generation += 1
            gen = self._generation
            self._chunks.clear()
            self._buffered_frames = 0
            self._decode_eof = False
            self._path = path
            self._position_seconds = max(0.0, start_seconds)
            self._state = "paused"
            self._callbacks = (on_ready, on_error, on_finished)
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._cond.notify_all()

        thread = threading.Thread(
            target=self._decode_worker,
            args=(
                gen,
                path,
                max(0.0, start_seconds),
                end_seconds,
                stop_event,
                on_ready,
                on_error,
                on_finished,
            ),
            daemon=True,
        )
        thread.start()
        return gen

    def pause(self) -> None:
        with self._lock:
            if self._state != "playing":
                return
            self._state = "paused"
        self._sink.stop()

    def resume(
        self,
        *,
        on_ready: Callable[[int, float], None] | None = None,
        on_error: Callable[[int, str], None] | None = None,
        on_finished: Callable[[int], None] | None = None,
    ) -> bool:
        """Resume from current cursor, or replay from selection/0 if already finished."""
        with self._cond:
            state = self._state
            path = self._path
            sel = self._selection
            has_audio = self._buffered_frames > 0 or not self._decode_eof
            if state == "paused" and has_audio:
                self._state = "playing"
                start_sink = True
                self._cond.notify_all()
            else:
                start_sink = False
        if start_sink:
            self._sink.start(self._pull, 2)
            return True
        if (state == "finished" or (state == "paused" and not has_audio)) and path is not None:
            start = sel[0] if sel is not None else 0.0
            end = sel[1] if sel is not None else None
            self.load_and_play(
                path,
                start,
                end_seconds=end,
                on_ready=on_ready or self._callbacks[0],
                on_error=on_error or self._callbacks[1],
                on_finished=on_finished or self._callbacks[2],
            )
            return True
        return False

    def stop(self) -> None:
        self._sink.stop()
        with self._cond:
            if self._stop_event is not None:
                self._stop_event.set()
            self._generation += 1
            self._chunks.clear()
            self._buffered_frames = 0
            self._decode_eof = True
            self._position_seconds = 0.0
            self._state = "paused" if self._path is not None else "empty"
            self._cond.notify_all()

    def close(self) -> None:
        self._closed = True
        self.stop()
        self._sink.close()

    def set_target_rate(self, target_rate_hz: int) -> None:
        """Update playback target sample rate when switching output audio devices."""
        self.stop()
        with self._cond:
            self._target_rate = max(8000, int(target_rate_hz))
            self._max_buffer_frames = int(MAX_BUFFERED_SECONDS * self._target_rate)

    def _pull(self, frames: int) -> np.ndarray:
        """Audio callback: reads only pre-decoded chunks from memory, never blocks."""
        need = max(frames, 0)
        if need == 0:
            return np.zeros((0, 2), dtype=np.float32)

        with self._cond:
            if self._state != "playing":
                return np.zeros((need, 2), dtype=np.float32)

            gain = self._volume_gain
            parts: list[np.ndarray] = []
            got = 0
            while got < need and self._chunks:
                head = self._chunks[0]
                avail = head.samples.shape[0]
                take = min(need - got, avail)
                parts.append(head.samples[:take])
                self._position_seconds = head.orig_start_sec + take * head.orig_sec_per_frame
                if take == avail:
                    self._chunks.popleft()
                else:
                    head.samples = head.samples[take:]
                    head.orig_start_sec = self._position_seconds
                self._buffered_frames -= take
                got += take

            finished_now = got < need and self._decode_eof and not self._chunks
            if finished_now:
                self._state = "finished"
            self._cond.notify_all()

        if not parts:
            return np.zeros((0 if finished_now else need, 2), dtype=np.float32)
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if gain != 1.0:
            out = out * gain
        return out

    def _enqueue_chunk(
        self,
        gen: int,
        stop_event: threading.Event,
        samples: np.ndarray,
        chunk_orig_start: float,
        orig_sec_per_out_frame: float,
    ) -> bool:
        if samples.shape[0] <= 0:
            return True
        with self._cond:
            while (
                self._buffered_frames >= self._max_buffer_frames
                and not stop_event.is_set()
                and not self._closed
                and gen == self._generation
            ):
                self._cond.wait(timeout=0.05)
            if stop_event.is_set() or self._closed or gen != self._generation:
                return False
            self._chunks.append(
                _QueuedChunk(samples, chunk_orig_start, orig_sec_per_out_frame)
            )
            self._buffered_frames += samples.shape[0]
            return True

    def _decode_worker(
        self,
        gen: int,
        path: Path,
        start_seconds: float,
        end_seconds: float | None,
        stop_event: threading.Event,
        on_ready: Callable[[int, float], None] | None,
        on_error: Callable[[int, str], None] | None,
        on_finished: Callable[[int], None] | None,
    ) -> None:
        try:
            with open_audio_file(path) as handle:
                src_rate = int(handle.samplerate)
                total_frames = int(handle.frames)
                channels = max(int(handle.channels), 1)
                if src_rate <= 0 or total_frames <= 0:
                    raise ValueError("音频没有可播放的采样")

                cur_frame = min(int(round(start_seconds * src_rate)), max(total_frames - 1, 0))
                if cur_frame > 0:
                    handle.seek(cur_frame)

                resampler = FractionalResampler(src_rate, self._target_rate, channels, speed=1.0)
                wsola = StreamingWSOLA(self._target_rate, channels)

                started_output = False
                is_first_chunk = True
                pass_orig_sec = cur_frame / src_rate
                last_stretch = 1.0

                while not stop_event.is_set() and not self._closed:
                    with self._lock:
                        if gen != self._generation:
                            return
                        sel = self._selection
                        pitch = self._pitch_semitones
                        rate = self._rate
                        preserve_pitch = self._preserve_pitch

                    pitch_ratio = 2.0 ** (pitch / 12.0)
                    eff_speed = rate * pitch_ratio
                    stretch_factor = rate / pitch_ratio
                    consume_rate = rate if preserve_pitch else eff_speed
                    target_sec = FIRST_CHUNK_SECONDS if is_first_chunk else CHUNK_SECONDS
                    read_budget = max(
                        256,
                        min(
                            int(src_rate * 0.5),
                            int(round(src_rate * target_sec * max(1.0, consume_rate))),
                        ),
                    )
                    if is_first_chunk:
                        read_budget = max(read_budget, 1024)
                    is_first_chunk = False

                    # Determine dynamic pass bounds from selection / end_seconds.
                    if sel is not None:
                        bound_end_sec = sel[1]
                    elif end_seconds is not None:
                        bound_end_sec = end_seconds
                    else:
                        bound_end_sec = total_frames / src_rate

                    end_frame = min(
                        total_frames, max(cur_frame + 1, int(round(bound_end_sec * src_rate)))
                    )
                    remaining = max(0, end_frame - cur_frame)
                    to_read = min(read_budget, remaining) if remaining > 0 else 0

                    raw = (
                        handle.read(frames=to_read, dtype="float32", always_2d=True)
                        if to_read > 0
                        else np.zeros((0, channels), dtype=np.float32)
                    )
                    cur_frame += raw.shape[0]
                    hit_bound = cur_frame >= end_frame or raw.shape[0] < to_read

                    if not preserve_pitch:
                        if wsola.has_pending:
                            tail = wsola.flush(last_stretch)
                            if tail.shape[0] > 0:
                                tail_rate = last_stretch * pitch_ratio / self._target_rate
                                if not self._enqueue_chunk(
                                    gen, stop_event, tail, pass_orig_sec, tail_rate
                                ):
                                    return
                                pass_orig_sec += tail.shape[0] * tail_rate
                        resampler.set_speed(eff_speed)
                        out = resampler.process(raw, eof=hit_bound)
                        orig_sec_per_frame = eff_speed / self._target_rate
                    else:
                        resampler.set_speed(pitch_ratio)
                        mid = resampler.process(raw, eof=hit_bound)
                        orig_sec_per_frame = rate / self._target_rate
                        if not wsola.has_pending and abs(stretch_factor - 1.0) < 1e-4:
                            out = mid
                        else:
                            out = wsola.process(mid, stretch_factor)
                            last_stretch = stretch_factor
                            if hit_bound and wsola.has_pending:
                                tail = wsola.flush(stretch_factor)
                                if tail.shape[0] > 0:
                                    out = (
                                        tail
                                        if out.shape[0] == 0
                                        else np.concatenate([out, tail], axis=0)
                                    )

                    if out.shape[0] > 0:
                        if not self._enqueue_chunk(
                            gen, stop_event, out, pass_orig_sec, orig_sec_per_frame
                        ):
                            return
                        pass_orig_sec += out.shape[0] * orig_sec_per_frame

                    if not started_output and (out.shape[0] > 0 or hit_bound):
                        started_output = True
                        with self._lock:
                            if stop_event.is_set() or self._closed or gen != self._generation:
                                return
                            self._state = "playing"
                        self._sink.start(self._pull, channels)
                        if on_ready is not None:
                            on_ready(gen, start_seconds)

                    if hit_bound:
                        with self._cond:
                            if gen != self._generation:
                                return
                            loop_now = self._loop
                            sel_now = self._selection
                            if loop_now:
                                loop_start_sec = sel_now[0] if sel_now is not None else 0.0
                                cur_frame = min(
                                    max(int(round(loop_start_sec * src_rate)), 0),
                                    max(total_frames - 1, 0),
                                )
                                handle.seek(cur_frame)
                                resampler = FractionalResampler(
                                    src_rate, self._target_rate, channels, speed=1.0
                                )
                                wsola.reset()
                                pass_orig_sec = cur_frame / src_rate
                                is_first_chunk = True
                                continue
                            self._decode_eof = True
                        # Wait until playback finishes or user toggles Loop ON before EOF finishes.
                        restarted_loop = False
                        while not stop_event.is_set() and not self._closed:
                            with self._cond:
                                if gen != self._generation or self._state == "finished":
                                    break
                                if self._loop:
                                    loop_start_sec = (
                                        self._selection[0] if self._selection is not None else 0.0
                                    )
                                    cur_frame = min(
                                        max(int(round(loop_start_sec * src_rate)), 0),
                                        max(total_frames - 1, 0),
                                    )
                                    handle.seek(cur_frame)
                                    resampler = FractionalResampler(
                                        src_rate, self._target_rate, channels, speed=1.0
                                    )
                                    wsola.reset()
                                    pass_orig_sec = cur_frame / src_rate
                                    self._decode_eof = False
                                    restarted_loop = True
                                    break
                                self._cond.wait(timeout=0.05)
                        if restarted_loop:
                            is_first_chunk = True
                            continue
                        break

            if not stop_event.is_set() and not self._closed:
                with self._lock:
                    if gen == self._generation and self._state == "finished" and on_finished is not None:
                        on_finished(gen)
        except Exception as exc:
            if not stop_event.is_set() and not self._closed and on_error is not None:
                on_error(gen, str(exc))
