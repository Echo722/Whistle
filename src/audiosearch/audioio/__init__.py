"""Agent B: audio input and waveform generation.

Public surface for the rest of the project:

- ``ManifestCatalog`` resolves the private sample manifest and root map.
- ``open_wav`` / ``WavReader`` decode original-file ranges in bounded blocks.
- ``WavWaveformProvider`` implements the common ``WaveformProvider`` port with
  real per-channel peaks.
"""

from .catalog import ManifestCatalog, ResolvedAsset, load_manifest
from .decode import DEFAULT_BLOCK_FRAMES, DecodedWindow, WavReader, open_wav, probe
from .errors import (
    AssetChangedError,
    AssetNotFoundError,
    AssetOfflineError,
    AudioError,
    DecodeError,
    SourceRangeError,
    UnsupportedFormatError,
)
from .wav import WavInfo, parse_wav, read_wav_info
from .waveform import PeakResult, WavWaveformProvider, compute_peaks

__all__ = [
    "DEFAULT_BLOCK_FRAMES",
    "AssetChangedError",
    "AssetNotFoundError",
    "AssetOfflineError",
    "AudioError",
    "DecodeError",
    "DecodedWindow",
    "ManifestCatalog",
    "PeakResult",
    "ResolvedAsset",
    "SourceRangeError",
    "UnsupportedFormatError",
    "WavInfo",
    "WavReader",
    "WavWaveformProvider",
    "compute_peaks",
    "load_manifest",
    "open_wav",
    "parse_wav",
    "probe",
    "read_wav_info",
]
