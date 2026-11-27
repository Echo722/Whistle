"""Agent B: model inference.

Importing this package does not import torch. Each encoder loads lazily and
raises ``ModelUnavailableError`` when its weights or runtime are missing.
"""

from .clap2_encoder import (
    CLAP2_PREPROCESSING_VERSION,
    CLAP2_TARGET_SAMPLE_RATE,
    DEFAULT_CLAP2_MODEL,
    DEFAULT_CLAP2_REVISION,
    Clap2Encoder,
    Clap2EncoderConfig,
    clap2_space,
)
from .clap_encoder import (
    PREPROCESSING_VERSION,
    TARGET_SAMPLE_RATE,
    WINDOW_SECONDS,
    AudioWindow,
    ClapEncoder,
    ClapEncoderConfig,
    coverage,
    plan_windows,
)
from .runtime import (
    DevicePlan,
    ResourceMeter,
    last_peak,
    peak_memory,
    plan_device,
    release_memory,
)
from .spaces import (
    EmbeddingSpace,
    ModelFingerprint,
    ModelUnavailableError,
    SpaceMismatchError,
    l2_normalize,
)
from .text_encoder import (
    TEMPLATE_VERSION,
    BgeM3TextEncoder,
    TextEncoderConfig,
    TextFieldPlan,
    plan_asset_text,
)

__all__ = [
    "CLAP2_PREPROCESSING_VERSION",
    "CLAP2_TARGET_SAMPLE_RATE",
    "DEFAULT_CLAP2_MODEL",
    "DEFAULT_CLAP2_REVISION",
    "PREPROCESSING_VERSION",
    "TARGET_SAMPLE_RATE",
    "TEMPLATE_VERSION",
    "WINDOW_SECONDS",
    "AudioWindow",
    "BgeM3TextEncoder",
    "Clap2Encoder",
    "Clap2EncoderConfig",
    "ClapEncoder",
    "ClapEncoderConfig",
    "DevicePlan",
    "EmbeddingSpace",
    "ModelFingerprint",
    "ModelUnavailableError",
    "ResourceMeter",
    "SpaceMismatchError",
    "TextFieldPlan",
    "TextEncoderConfig",
    "clap2_space",
    "coverage",
    "l2_normalize",
    "last_peak",
    "peak_memory",
    "plan_asset_text",
    "plan_device",
    "plan_windows",
    "release_memory",
]
