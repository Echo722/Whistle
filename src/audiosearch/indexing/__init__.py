"""Agent B: vector/index generations.

Files, not a database: A owns schema and migrations, so B publishes derived
artifacts (header + rows + vectors) that A can later import or point at.
"""

from .encode import (
    AUDIO_HOP_SECONDS,
    AUDIO_WINDOW_POLICY,
    AUDIO_WINDOW_SECONDS,
    AssetFailure,
    EncodeReport,
    audio_input_fingerprint,
    encode_audio_segments,
    encode_text_assets,
    plan_asset_windows,
    text_input_fingerprint,
    window_policy,
)
from .store import (
    SCHEMA_VERSION,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactIncompatibleError,
    EmbeddingBatch,
    EmbeddingRow,
    EmbeddingStore,
    ValidityReport,
    write_json_atomic,
)

__all__ = [
    "AUDIO_HOP_SECONDS",
    "AUDIO_WINDOW_POLICY",
    "AUDIO_WINDOW_SECONDS",
    "SCHEMA_VERSION",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactIncompatibleError",
    "AssetFailure",
    "EmbeddingBatch",
    "EmbeddingRow",
    "EmbeddingStore",
    "EncodeReport",
    "ValidityReport",
    "audio_input_fingerprint",
    "encode_audio_segments",
    "encode_text_assets",
    "plan_asset_windows",
    "text_input_fingerprint",
    "window_policy",
    "write_json_atomic",
]
