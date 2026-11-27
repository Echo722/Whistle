"""In-process adapter boundaries; wire data is defined in models.py."""

from typing import Protocol
from uuid import UUID

from .models import AssetRecord, QueryPlan, SearchResult, WaveformData, WaveformRequest


class AssetCatalog(Protocol):
    def get_asset(self, asset_id: UUID) -> AssetRecord | None: ...


class Retriever(Protocol):
    def retrieve(self, plan: QueryPlan, *, limit: int) -> list[SearchResult]: ...


class WaveformProvider(Protocol):
    def waveform(self, request: WaveformRequest) -> WaveformData: ...
