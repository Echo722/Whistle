"""Wire contracts shared by catalog, retrieval, desktop and REAPER adapters.

Times in SourceRange are seconds on the ORIGINAL file, with an exclusive end.
Pydantic validation checks structure; adapters must also check current state,
asset bounds, session identity and command expiration before performing work.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

PROTOCOL_VERSION = "0.1.0"
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Seconds = Annotated[float, Field(ge=0, strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class Versioned(Contract):
    protocol_version: Literal["0.1.0"] = PROTOCOL_VERSION


class Availability(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class RetrievalSource(StrEnum):
    KEYWORD = "keyword"
    TEXT = "text"
    AUDIO = "audio"


class SourceRange(Contract):
    start_seconds: Seconds
    end_seconds: Seconds
    # None means all channels; channel numbers are zero-based original channels.
    channels: list[Count] | None = None

    @model_validator(mode="after")
    def check_range(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.channels is not None and (
            not self.channels or len(set(self.channels)) != len(self.channels)
        ):
            raise ValueError("channels must be non-empty and unique, or null for all channels")
        return self


class AssetRef(Contract):
    asset_id: UUID
    content_revision: NonEmpty


class AssetLocation(Contract):
    location_id: UUID
    root_id: NonEmpty
    relative_path: str = Field(min_length=1)
    availability: Availability
    size_bytes: Count
    mtime_ns: Count

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        value = value.replace("\\", "/")
        if (
            value.startswith("/")
            or ":" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("relative_path must be root-relative without traversal or drive names")
        return value


class AssetRecord(Versioned):
    asset_id: UUID
    content_revision: NonEmpty
    name: NonEmpty
    library_id: NonEmpty
    package: str | None = None
    description: str | None = None
    description_source: Literal["file", "media_explorer", "user", "machine", "fixture"] | None = (
        None
    )
    keywords: list[str] = Field(default_factory=list)
    duration_seconds: Seconds | None = None
    sample_rate_hz: Annotated[int, Field(gt=0, strict=True)] | None = None
    channel_count: Annotated[int, Field(gt=0, strict=True)] | None = None
    bit_depth: Annotated[int, Field(gt=0, strict=True)] | None = None
    locations: list[AssetLocation] = Field(min_length=1)
    audio_analysis: Literal["not_started", "sampled", "complete", "failed"] = "not_started"

    @model_validator(mode="after")
    def unique_locations(self) -> Self:
        ids = [location.location_id for location in self.locations]
        if len(set(ids)) != len(ids):
            raise ValueError("location IDs must be unique")
        return self


class HardFilters(Contract):
    library_ids: list[NonEmpty] = Field(default_factory=list)
    duration_min_seconds: Seconds | None = None
    duration_max_seconds: Seconds | None = None
    sample_rates_hz: list[Annotated[int, Field(gt=0, strict=True)]] = Field(default_factory=list)
    channel_counts: list[Annotated[int, Field(gt=0, strict=True)]] = Field(default_factory=list)
    online_only: bool = False

    @model_validator(mode="after")
    def valid_duration(self) -> Self:
        lo, hi = self.duration_min_seconds, self.duration_max_seconds
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("minimum duration must not exceed maximum")
        return self


class QueryPlan(Versioned):
    original_text: str = Field(max_length=4096)
    mode: Literal["natural", "exact"] = "natural"
    # Inner lists are synonyms (OR); groups are combined with AND.
    keyword_groups: list[list[NonEmpty]] = Field(default_factory=list)
    text_query: str | None = Field(default=None, max_length=4096)
    audio_descriptions: list[NonEmpty] = Field(default_factory=list, max_length=3)
    filters: HardFilters = Field(default_factory=HardFilters)
    soft_preferences: list[NonEmpty] = Field(default_factory=list)
    interpretation: Literal["raw", "rules", "model", "fixture"] = "raw"
    interpretation_version: NonEmpty = "none"

    @field_validator("keyword_groups")
    @classmethod
    def no_empty_groups(cls, value: list[list[str]]) -> list[list[str]]:
        if any(not group for group in value):
            raise ValueError("keyword groups must not be empty")
        return value


class SearchRequest(Versioned):
    query_id: UUID
    plan: QueryPlan
    page_size: Annotated[int, Field(ge=1, le=100, strict=True)] = 50
    cursor: str | None = Field(default=None, max_length=4096)


class RouteStatus(Contract):
    source: RetrievalSource
    state: Literal["ready", "running", "unavailable", "failed"]
    reason: str | None = None


class SearchResult(Contract):
    asset: AssetRecord
    rank: Annotated[int, Field(ge=1, strict=True)]
    sources: list[RetrievalSource] = Field(min_length=1)
    matched_range: SourceRange | None = None
    explanation: list[str] = Field(default_factory=list)
    # Not a probability or a percentage of correctness.
    score: float | None = None

    @model_validator(mode="after")
    def valid_match(self) -> Self:
        if self.matched_range is not None:
            validate_source_range(self.asset, self.matched_range)
        return self


class SearchPage(Versioned):
    query_id: UUID
    index_generation: NonEmpty
    data_kind: Literal["fixture", "local"]
    results: list[SearchResult]
    routes: list[RouteStatus]
    is_complete: bool
    next_cursor: str | None = None
    total_candidates: Count | None = None
    count_scope: Literal["candidate_snapshot", "exact_keyword"] = "candidate_snapshot"
    warnings: list[str] = Field(default_factory=list)


class SimilarRequest(Versioned):
    """Reference sound → similar sounds over the audio vector route.

    Deliberately has no cursor: a neighbour list is one short ranked page, and a
    cursor into a graph the user cannot reproduce would only invent pagination.
    """

    # Echoed back unchanged: the desktop rejects a page whose query id is not the
    # one it is waiting for, exactly as it does for ``/v1/search``.
    query_id: UUID
    asset_id: UUID
    # None means the whole file; a range narrows it to the part the user picked out.
    source_range: SourceRange | None = None
    limit: Annotated[int, Field(ge=1, le=100, strict=True)] = 20


class WaveformRequest(Versioned):
    asset: AssetRef
    source_range: SourceRange
    points: Annotated[int, Field(ge=16, le=4096, strict=True)] = 256


class Peak(Contract):
    minimum: Annotated[float, Field(ge=-1, le=1)]
    maximum: Annotated[float, Field(ge=-1, le=1)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("peak minimum must not exceed maximum")
        return self


class WaveformData(Versioned):
    asset: AssetRef
    source_range: SourceRange
    data_kind: Literal["fixture", "local"]
    status: Literal["ready", "pending", "failed"]
    # Number of equally spaced buckets spanning source_range, per channel.
    points: Annotated[int, Field(gt=0, strict=True)]
    channel_indices: list[Count]
    peaks: list[list[Peak]]
    message: str | None = None

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if len(set(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("waveform channel indices must be unique")
        if self.status == "ready":
            if not self.channel_indices or len(self.peaks) != len(self.channel_indices):
                raise ValueError("ready waveforms need one peak array per channel")
            if (
                self.source_range.channels is not None
                and self.channel_indices != self.source_range.channels
            ):
                raise ValueError("waveform channels must match the requested original channels")
            if any(len(channel) != self.points for channel in self.peaks):
                raise ValueError("every channel must have points buckets")
        elif self.peaks:
            raise ValueError("pending/failed waveforms must not expose partial peaks")
        return self


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    ASSET_NOT_FOUND = "asset_not_found"
    ASSET_OFFLINE = "asset_offline"
    ASSET_CHANGED = "asset_changed"
    MODEL_UNAVAILABLE = "model_unavailable"
    BRIDGE_UNAVAILABLE = "bridge_unavailable"
    REQUEST_EXPIRED = "request_expired"
    TARGET_CHANGED = "target_changed"
    CURSOR_INVALID = "cursor_invalid"
    EXECUTION_UNCERTAIN = "execution_uncertain"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(Contract):
    code: ErrorCode
    message: NonEmpty
    retryable: bool = False


class ErrorResponse(Versioned):
    error: ErrorDetail


class CommandContext(Versioned):
    request_id: UUID
    reaper_instance_id: NonEmpty
    session_id: UUID
    issued_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


class PlaybackOptions(Contract):
    gain_db: Annotated[float, Field(ge=-60, le=12)] = 0
    play_rate: Annotated[float, Field(gt=0, le=16)] = 1
    pitch_semitones: Annotated[float, Field(ge=-48, le=48)] = 0
    preserve_pitch: bool = True
    loop: bool = False


class PlaybackStartRequest(Versioned):
    context: CommandContext
    generation: Count
    asset: AssetRef
    source_range: SourceRange | None = None
    options: PlaybackOptions = Field(default_factory=PlaybackOptions)
    output_project_id: NonEmpty | None = None
    output_track_id: NonEmpty | None = None

    @model_validator(mode="after")
    def track_needs_project(self) -> Self:
        if self.output_track_id is not None and self.output_project_id is None:
            raise ValueError("track output needs an explicit project")
        return self


class PlaybackStopRequest(Versioned):
    context: CommandContext
    generation: Count


class PlaybackSeekRequest(PlaybackStopRequest):
    source_position_seconds: Seconds


class InsertTarget(Contract):
    project_id: NonEmpty
    # An empty ``track_id`` is a deliberate value, not a missing field: it means
    # "nothing is selected, make a track". The bridge creates that track inside
    # the same undo block as the item, so a failure leaves neither behind.
    track_id: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""
    timeline_position_seconds: Seconds


class InsertRequest(Versioned):
    context: CommandContext
    asset: AssetRef
    source_range: SourceRange | None = None
    target: InsertTarget
    # Explicit opt-in: null means original playback settings.
    apply_playback_options: PlaybackOptions | None = None


class CommandReceipt(Versioned):
    request_id: UUID
    status: Literal["accepted", "completed", "rejected", "uncertain"]
    created_item_ids: list[NonEmpty] = Field(default_factory=list)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def valid_receipt(self) -> Self:
        if self.status in {"rejected", "uncertain"} and self.error is None:
            raise ValueError("rejected/uncertain receipts need an error")
        if self.status in {"accepted", "completed"} and self.error is not None:
            raise ValueError("accepted/completed receipt cannot contain an error")
        if self.status in {"accepted", "rejected"} and self.created_item_ids:
            raise ValueError("accepted/rejected receipt cannot claim created items")
        return self


class JobStatus(Versioned):
    job_id: UUID
    stage: NonEmpty
    processing_version: NonEmpty
    state: Literal["pending", "running", "paused", "completed", "failed", "offline", "cancelled"]
    unit: Literal["files", "segments"] = "files"
    total: Count
    completed: Count = 0
    failed: Count = 0
    skipped: Count = 0
    resume_token: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def valid_progress(self) -> Self:
        processed = self.completed + self.failed + self.skipped
        if processed > self.total:
            raise ValueError("processed count exceeds total")
        if self.state == "completed" and processed != self.total:
            raise ValueError("completed jobs must account for all work items")
        return self


class ServiceCapabilities(Versioned):
    data_kind: Literal["fixture", "local"]
    routes: list[RouteStatus]
    bridge_connected: bool = False
    playback_backend: Literal["unavailable", "reaper", "local"] = "unavailable"
    commands: list[str] = Field(default_factory=list)


class SampleManifest(Versioned):
    manifest_id: NonEmpty
    generated_at: AwareDatetime
    selection_method: Literal["sha256-relative-path-v1"]
    # 上界曾固定为 1000，那是 bootstrap 抽样阶段为了避免误把整库当成试用清单。
    # 真实素材库需要全量清册，因此只保留下界；条数本身不是产品边界，规模由生成方决定。
    selection_limit: Annotated[int, Field(ge=1, strict=True)]
    library_id: NonEmpty
    root_id: NonEmpty
    data_kind: Literal["local_manifest"] = "local_manifest"
    # Snapshot paths/metadata are private and not committed to Git.
    assets: list[AssetRecord]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_assets(self) -> Self:
        if len(self.assets) > self.selection_limit:
            raise ValueError("manifest exceeds selection limit")
        ids = [asset.asset_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest asset IDs must be unique")
        for asset in self.assets:
            if asset.library_id != self.library_id:
                raise ValueError("asset library differs from manifest library")
            if any(location.root_id != self.root_id for location in asset.locations):
                raise ValueError("asset root differs from manifest root")
        return self


def validate_source_range(asset: AssetRecord, source_range: SourceRange) -> None:
    """Validate file bounds using known metadata; no I/O or project-time conversion."""
    if (
        asset.duration_seconds is not None
        and source_range.end_seconds > asset.duration_seconds + 1e-6
    ):
        raise ValueError("source range exceeds asset duration")
    if (
        source_range.channels is not None
        and asset.channel_count is not None
        and any(channel >= asset.channel_count for channel in source_range.channels)
    ):
        raise ValueError("source channel exceeds asset channel count")


PUBLIC_MODELS = (
    AssetRef,
    AssetRecord,
    SourceRange,
    QueryPlan,
    SearchRequest,
    SearchPage,
    WaveformRequest,
    WaveformData,
    PlaybackStartRequest,
    PlaybackStopRequest,
    PlaybackSeekRequest,
    InsertRequest,
    CommandReceipt,
    JobStatus,
    ServiceCapabilities,
    ErrorResponse,
    SampleManifest,
)
