"""Real REAPER bridge (protocol asb1) and adapter interface (reaper-adapter-v0).

Importing this package starts nothing and needs only the standard library,
pydantic and the public contracts (no numpy/torch/Qt).
"""

from .adapter import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_INSERT_TIMEOUT,
    DEFAULT_PREVIEW_TIMEOUT,
    MAX_CLOCK_SKEW_SECONDS,
    PlaybackBridge,
)
from .bridge import ReaperBridge, ReaperBridgeConfig
from .channel import CONFIG_NAME, PROTOCOL
from .errors import (
    BridgeClosedError,
    BridgeError,
    BridgeUsageError,
    CommandRejected,
)
from .sources import (
    ResolvedSource,
    SourceResolver,
    SourceUnavailableError,
    catalog_source_resolver,
)
from .status import (
    ADAPTER_PROTOCOL,
    BridgeCapabilities,
    BridgeCommand,
    BridgeStatus,
    ConnectionState,
    PlaybackSnapshot,
    ProjectInfo,
    TrackInfo,
)

__all__ = [
    "ADAPTER_PROTOCOL",
    "CONFIG_NAME",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_INSERT_TIMEOUT",
    "DEFAULT_PREVIEW_TIMEOUT",
    "MAX_CLOCK_SKEW_SECONDS",
    "PROTOCOL",
    "BridgeCapabilities",
    "BridgeClosedError",
    "BridgeCommand",
    "BridgeError",
    "BridgeStatus",
    "BridgeUsageError",
    "CommandRejected",
    "ConnectionState",
    "PlaybackBridge",
    "PlaybackSnapshot",
    "ProjectInfo",
    "ReaperBridge",
    "ReaperBridgeConfig",
    "ResolvedSource",
    "SourceResolver",
    "SourceUnavailableError",
    "TrackInfo",
    "catalog_source_resolver",
]
