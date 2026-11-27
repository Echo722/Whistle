# ruff: noqa: E402  (the numpy guard must run before the project imports)
"""Numeric behaviour of the resampler, measured independently of its own filter.

Every expectation below is computed from the signal, not from the implementation:
gain from RMS/peak of a known sine, alias suppression from an off-band tone that
*should* disappear, continuity from a constant input, and the reference for the
block-size test is the same call with a different internal block budget.

The three defects recorded in ``docs/reports/RESAMPLE-005-R0.md`` each have a test
that fails on the old implementation.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip(
    "numpy",
    reason="resampler tests need numpy; it is an optional extra, not a shared dev dependency",
)

import audiosearch.audioio.resample as resample_module
from audiosearch.audioio.resample import RESAMPLE_VERSION, resample

RATES = [(48000, 96000), (32000, 48000), (48000, 24000), (96000, 48000), (44100, 48000)]


def tone(frames: int, rate: int, frequency: float, amplitude: float = 0.25) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(math.sqrt(float(np.mean(data * data))))


def peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def interior(values: np.ndarray, rate: int, settle_seconds: float = 0.05) -> np.ndarray:
    """Drop the filter's start/end transient; every internal block stays inside."""
    margin = min(len(values) // 4, int(rate * settle_seconds))
    return values[margin : len(values) - margin]


def expected_length(frames: int, src: int, dst: int) -> int:
    return int(math.ceil(frames * dst / src))


@pytest.mark.parametrize(("src", "dst"), RATES)
def test_length_is_ceil_of_the_ratio(src: int, dst: int) -> None:
    for frames in (1, 7, 1000, 44100 + 3):
        out = resample(tone(frames, src, 100.0), src, dst)
        assert out.shape[0] == expected_length(frames, src, dst)
        assert out.dtype == np.float32


@pytest.mark.parametrize(("src", "dst"), RATES)
def test_passband_sine_keeps_its_amplitude(src: int, dst: int) -> None:
    """1 kHz is far below either Nyquist: the gain must be ~1 for every ratio."""
    frames = src  # one second
    out = resample(tone(frames, src, 1000.0, amplitude=0.25), src, dst)
    body = interior(out, dst)
    expected_rms = 0.25 / math.sqrt(2.0)
    measured = rms(body)
    gain_db = 20 * math.log10(measured / expected_rms)
    assert abs(gain_db) < 0.1, f"{src}->{dst}: gain {gain_db:+.3f} dB"
    assert peak(body) == pytest.approx(0.25, abs=0.002)


@pytest.mark.parametrize(
    ("src", "dst", "tone_hz"),
    [
        (96000, 48000, 30000.0),  # R0 defect #1: folded to 18 kHz with no attenuation
        (96000, 48000, 28000.0),
        (96000, 48000, 36000.0),
        (48000, 24000, 15000.0),
        (48000, 24000, 14000.0),
        (44100, 22050, 14000.0),
    ],
)
def test_above_nyquist_input_is_suppressed_not_folded(src: int, dst: int, tone_hz: float) -> None:
    """Full stopband depth: at least 60 dB down, measured on the alias that appears."""
    assert tone_hz > dst / 2
    frames = src * 2
    out = resample(tone(frames, src, tone_hz, amplitude=0.25), src, dst)
    body = interior(out, dst, settle_seconds=0.2)
    reference = 0.25 / math.sqrt(2.0)  # what an unattenuated alias would measure
    suppression_db = 20 * math.log10(max(rms(body), 1e-12) / reference)
    assert suppression_db <= -60.0, f"{tone_hz:g} Hz at {src}->{dst}: {suppression_db:.1f} dB"


def test_the_transition_band_is_narrow_and_documented() -> None:
    """Honest boundary of this design: how close to Nyquist the stopband starts.

    With 24 zero crossings at 2:1 the passband is flat to ~22 kHz and the
    stopband reaches 60 dB by ~27 kHz; the band between is a gradual roll-off,
    not a claim of instant cut-off. Recorded here so the report and the tests
    cannot drift apart.
    """
    reference = 0.25 / math.sqrt(2.0)

    def suppression(frequency: float) -> float:
        out = resample(tone(96000 * 2, 96000, frequency), 96000, 48000)
        body = interior(out, 48000, settle_seconds=0.2)
        return 20 * math.log10(max(rms(body), 1e-12) / reference)

    assert suppression(20000.0) > -0.5, "20 kHz must stay in the passband"
    assert suppression(22000.0) > -1.0, "22 kHz must stay in the passband"
    assert suppression(25000.0) < -10.0, "roll-off must have started by 25 kHz"
    assert suppression(28000.0) < -60.0, "by 28 kHz the alias must be gone"
    assert suppression(30000.0) < -60.0
    # Monotone: a louder alias further from the band would mean a broken kernel.
    measured = [suppression(f) for f in (22000.0, 25000.0, 28000.0, 30000.0)]
    assert measured == sorted(measured, reverse=True)


@pytest.mark.parametrize(("src", "dst"), RATES)
def test_a_constant_input_stays_constant(src: int, dst: int) -> None:
    """No periodic dropouts: R0 defect #2 came from resetting history per block."""
    frames = src * 3
    out = resample(np.full(frames, 0.25, dtype=np.float32), src, dst)
    body = interior(out, dst, settle_seconds=0.1)
    assert float(np.min(body)) == pytest.approx(0.25, abs=1e-4)
    assert float(np.max(body)) == pytest.approx(0.25, abs=1e-4)
    # The old code produced ~48 near-zero samples at every 262144-frame boundary.
    assert float(np.min(np.abs(body))) > 0.24


def test_a_44k1_sine_has_no_spike() -> None:
    """R0 defect #3: a single output sample of 36.7 on a 0.25 sine."""
    out = resample(tone(44100, 44100, 220.0, amplitude=0.25), 44100, 48000)
    assert peak(out) <= 0.26
    body = interior(out, 48000)
    assert rms(body) == pytest.approx(0.25 / math.sqrt(2.0), rel=0.01)


def test_no_output_sample_is_far_above_the_input_peak() -> None:
    """A resampler may not invent amplitude; check a broad-band input too."""
    rng = np.random.default_rng(7)
    signal = (rng.standard_normal(96000) * 0.1).astype(np.float32)
    out = resample(signal, 96000, 48000)
    assert peak(out) <= 1.5 * peak(signal)


def test_block_size_does_not_change_a_single_sample() -> None:
    frames = 96000 * 3
    signal = tone(frames, 96000, 3000.0) + tone(frames, 96000, 17000.0)
    original = resample_module._MAX_BLOCK_OUTPUT
    try:
        resample_module._MAX_BLOCK_OUTPUT = frames  # one block
        reference = resample(signal, 96000, 48000)
        for block in (777, 4096, 1 << 15, 262144):
            resample_module._MAX_BLOCK_OUTPUT = block
            out = resample(signal, 96000, 48000)
            assert np.array_equal(out, reference), f"block {block} changed the output"
    finally:
        resample_module._MAX_BLOCK_OUTPUT = original


def test_channels_stay_independent_and_aligned() -> None:
    rate = 48000
    t = np.arange(rate, dtype=np.float64) / rate
    left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    stereo = resample(np.stack([left, right], axis=1), rate, 96000)
    mono = resample(left, rate, 96000)
    assert stereo.shape == (96000, 2)
    assert np.array_equal(stereo[:, 0], mono)
    assert rms(stereo[:, 1]) == pytest.approx(0.25 / math.sqrt(2.0), rel=0.01)


def test_identity_rates_return_the_same_samples() -> None:
    signal = np.arange(100, dtype=np.float32)
    assert np.array_equal(resample(signal, 48000, 48000), signal)
    stereo = np.stack([signal, -signal], axis=1)
    assert np.array_equal(resample(stereo, 48000, 48000), stereo)


def test_empty_input_stays_empty() -> None:
    assert resample(np.zeros(0, dtype=np.float32), 96000, 48000).shape == (0,)
    assert resample(np.zeros((0, 2), dtype=np.float32), 96000, 48000).shape == (0, 2)


def test_a_single_sample_is_flushed_by_the_padding() -> None:
    out = resample(np.array([0.5], dtype=np.float32), 96000, 48000)
    assert out.shape == (1,)
    assert np.isfinite(out).all()


def test_bad_rates_and_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        resample(np.zeros(10, dtype=np.float32), 0, 48000)
    with pytest.raises(ValueError):
        resample(np.zeros(10, dtype=np.float32), 48000, 0)
    with pytest.raises(ValueError):
        resample(np.zeros((2, 3, 4), dtype=np.float32), 48000, 24000)


def test_a_dc_kernel_has_unity_gain_on_every_phase() -> None:
    """Independent check: the filter's taps sum to one per phase, not per ratio."""
    up, down = resample_module._ratio(44100, 48000)
    bank, _past, _future = resample_module._design(up, down, 24, 8.6)
    sums = bank.sum(axis=0)
    assert bank.shape[1] == up
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_the_designed_corner_is_the_output_nyquist() -> None:
    """The old design put it at the input Nyquist; that is the aliasing bug."""
    up, down = resample_module._ratio(96000, 48000)
    _length, _centre, corner = resample_module._kernel_length(up, down, 24)
    assert corner == pytest.approx(0.5 * min(1.0 / up, 1.0 / down))
    assert corner == pytest.approx(0.25)


def test_version_names_the_implementation() -> None:
    assert "polyphase" in RESAMPLE_VERSION
    assert RESAMPLE_VERSION.startswith("polyphase-v2")
