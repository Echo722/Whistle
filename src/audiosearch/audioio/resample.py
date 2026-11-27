"""Deterministic band-limited resampling without SciPy.

CLAP expects 48 kHz mono while the sample library is 96 kHz stereo, so every
window has to be resampled. Doing it in-process keeps the preprocessing version
reproducible: same taps, same phase, same input to the model every run.

Design (and why the previous version was wrong):

* The rational ratio is ``up/down`` after dividing by the gcd. Filtering happens
  at the upsampled rate, so the ideal low-pass corner is ``0.5 * min(1/up,
  1/down)`` cycles per upsampled sample — the *output* Nyquist for decimation and
  the *input* Nyquist for interpolation. Using ``0.5 * min(1, down/up)`` instead
  put the corner at the input Nyquist while decimating, so everything between the
  output and input Nyquist folded back with no attenuation at all.
* The bank is two-sided (past *and* future taps). Indexing only forward from the
  sample under the output position used an asymmetric half of the kernel.
* Every polyphase branch is normalised to sum to one, so DC gain is exactly 1 for
  every phase and a constant input stays constant.
* The kernel length is derived from the corner frequency, and ``up`` is allowed to
  exceed the tap count: the bank always has ``ceil(length/up) + 1`` rows per
  phase, so a 44.1 → 48 kHz conversion cannot collapse into a single-tap filter.
* Blocks partition the *output* and read from one padded copy of the input, so no
  block boundary ever restarts the filter history; every output sample sums the
  same taps in the same order whatever the block size, which makes results
  bit-identical across block sizes.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

DEFAULT_ZERO_CROSSINGS = 24
DEFAULT_KAISER_BETA = 8.6
# Name recorded in artifacts and in the preprocessing version.
RESAMPLE_IMPLEMENTATION = "polyphase-v2"
RESAMPLE_VERSION = (
    f"{RESAMPLE_IMPLEMENTATION}/zc{DEFAULT_ZERO_CROSSINGS}-kaiser{DEFAULT_KAISER_BETA}"
)
# Bound on intermediate buffers: how many output frames one block may hold.
_MAX_BLOCK_OUTPUT = 1 << 17


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _ratio(src_rate: int, dst_rate: int) -> tuple[int, int]:
    divisor = _gcd(src_rate, dst_rate)
    return dst_rate // divisor, src_rate // divisor


def _kernel_length(up: int, down: int, zero_crossings: int) -> tuple[int, int, float]:
    """(length, centre, corner) of the proto-filter at the upsampled rate."""
    corner = 0.5 * min(1.0 / up, 1.0 / down)  # cycles per upsampled sample
    half = zero_crossings / (2.0 * corner)
    length = 2 * int(math.ceil(half)) + 1
    return length, (length - 1) // 2, corner


@lru_cache(maxsize=32)
def _design(up: int, down: int, zero_crossings: int, beta: float) -> tuple[np.ndarray, int, int]:
    """Return ``(bank, past_rows, future_rows)`` for this rational ratio.

    ``bank[r, p]`` is the tap applied to input sample ``k0 + (r - past_rows)`` for
    the output sample whose position is ``k0 + p/up`` in input samples. Each
    column is normalised to sum to one, so the DC gain of every phase is exactly
    unity and a constant input stays constant.
    """
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive")
    length, centre, corner = _kernel_length(up, down, zero_crossings)
    n = np.arange(length, dtype=np.float64) - centre
    taps = np.sinc(2.0 * corner * n) * np.kaiser(length, beta) * (2.0 * corner)

    past = centre // up
    future = (centre + up - 1) // up + 1
    rows = past + future
    bank = np.zeros((rows, up), dtype=np.float64)
    for row in range(rows):
        offset = (row - past) * up
        indices = centre - offset + np.arange(up)
        valid = (indices >= 0) & (indices < length)
        bank[row, valid] = taps[indices[valid]]
    column_sums = bank.sum(axis=0)
    if np.any(np.abs(column_sums) < 1e-9):
        # Degenerate design (corner too low for this tap count): refuse rather
        # than divide by a near-zero DC gain and call the result a conversion.
        raise ValueError(
            f"cannot design a usable {up}/{down} kernel with {zero_crossings} zero crossings"
        )
    bank /= column_sums[np.newaxis, :]
    return bank.astype(np.float32), past, future


def _resample_axis(
    data: np.ndarray,
    up: int,
    down: int,
    bank: np.ndarray,
    past: int,
    future: int,
    out_frames: int,
) -> np.ndarray:
    """Filter one ``(frames, channels)`` array by the rational ratio.

    The input is padded once and every output sample reads a contiguous run of
    ``rows`` padded samples, so a block boundary is only a place where the loop
    stops — never a place where filter history restarts. Blocks are sized by a
    memory budget, so results are bit-identical for any block size.
    """
    rows = bank.shape[0]
    channels = data.shape[1]
    padded = data.astype(np.float64, copy=False)
    if past:
        padded = np.concatenate([np.zeros((past, channels), dtype=np.float64), padded], axis=0)
    if future:
        padded = np.concatenate([padded, np.zeros((future, channels), dtype=np.float64)], axis=0)

    # Keep the (block, rows, channels) gather under ~32 MiB of float64.
    budget = max(1, (32 << 20) // max(1, rows * channels * 8))
    block = max(1, min(_MAX_BLOCK_OUTPUT, budget, out_frames))
    offsets = np.arange(rows, dtype=np.int64)
    out = np.zeros((out_frames, channels), dtype=np.float64)
    for start in range(0, out_frames, block):
        stop = min(out_frames, start + block)
        positions = np.arange(start, stop, dtype=np.int64) * down
        base = positions // up
        phase = positions - base * up
        window = padded[base[:, np.newaxis] + offsets[np.newaxis, :]]
        taps = bank[:, phase].astype(np.float64).T  # (block, rows)
        out[start:stop] = np.einsum("brc,br->bc", window, taps, optimize=True)
    return out.astype(np.float32)


def resample(
    samples: np.ndarray,
    src_rate: int,
    dst_rate: int,
    *,
    zero_crossings: int = DEFAULT_ZERO_CROSSINGS,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """Resample mono ``(frames,)`` or multi-channel ``(frames, channels)`` float32.

    Output length is ``ceil(frames * dst_rate / src_rate)``. Time keeps its
    meaning: output sample ``n`` covers input position ``n * src_rate /
    dst_rate``, with no extra delay introduced by the filter (the kernel is
    centred), so original-file seconds are unchanged.
    """
    data = np.asarray(samples, dtype=np.float32)
    single = data.ndim == 1
    if single:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError("samples must be 1-D mono or 2-D (frames, channels)")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if data.shape[0] == 0:
        return data[:, 0] if single else data
    if src_rate == dst_rate:
        return data[:, 0] if single else data

    up, down = _ratio(src_rate, dst_rate)
    bank, past, future = _design(up, down, zero_crossings, beta)
    out_frames = int(math.ceil(data.shape[0] * up / down))
    out = _resample_axis(data, up, down, bank, past, future, out_frames)
    return out[:, 0] if single else out


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Mean across channels; mono input is returned unchanged."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    return data.mean(axis=1, dtype=np.float32)


__all__ = [
    "DEFAULT_KAISER_BETA",
    "DEFAULT_ZERO_CROSSINGS",
    "RESAMPLE_IMPLEMENTATION",
    "RESAMPLE_VERSION",
    "resample",
    "to_mono",
]
