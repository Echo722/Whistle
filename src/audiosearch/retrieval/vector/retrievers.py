"""Vector retrievers exposed through the common ``Retriever`` port.

Two retrievers, never one merged space:

``TextVectorRetriever``  BGE-M3 text query -> BGE-M3 asset vectors.
``AudioVectorRetriever`` CLAP text query -> CLAP audio window vectors.

The audio retriever also answers "find me sounds like this one" by encoding a
reference window with the CLAP *audio* tower and searching the same space.

Two rules keep results honest as artifacts evolve:

* stored rows are filtered against the live catalog *before* any truncation —
  an asset whose ``content_revision`` (or encoded-input fingerprint) no longer
  matches is excluded, never returned while labelled with the new record;
* the in-memory index is keyed by the store's published generation, so a
  re-encode is picked up on the next call instead of being masked by a cache.

Results are ``SearchResult`` from the shared contracts, so they can go straight
into a ``SearchPage``; ``sources`` says which tower produced the rank, and
``matched_range`` carries the original-file segment of the best window.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np

from audiosearch.audioio.catalog import ManifestCatalog
from audiosearch.contracts.models import (
    AssetRecord,
    QueryPlan,
    RetrievalSource,
    SearchResult,
    SourceRange,
    validate_source_range,
)
from audiosearch.indexing.encode import (
    AUDIO_HOP_SECONDS,
    AUDIO_WINDOW_SECONDS,
    audio_input_fingerprint,
    plan_asset_windows,
    text_input_fingerprint,
)
from audiosearch.indexing.store import EmbeddingStore, ValidityReport
from audiosearch.inference.clap_encoder import ClapEncoder
from audiosearch.inference.spaces import ModelUnavailableError, SpaceMismatchError
from audiosearch.inference.text_encoder import BgeM3TextEncoder, plan_asset_text
from audiosearch.retrieval.vector.exact import ExactVectorIndex, VectorHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """Results plus the honest state of the route that produced them."""

    results: list[SearchResult]
    state: str
    reason: str | None = None
    stale_assets: int = 0
    missing_assets: int = 0
    unverified_assets: int = 0

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.stale_assets:
            notes.append(
                f"{self.stale_assets} stored vectors no longer match the catalog or their "
                "encoded input and were excluded; re-run the encode step"
            )
        if self.missing_assets:
            notes.append(f"{self.missing_assets} catalog assets have no stored vectors yet")
        if self.unverified_assets:
            notes.append(
                f"{self.unverified_assets} vectors come from an artifact without input "
                "fingerprints; metadata-only changes are not detected until it is re-encoded"
            )
        return notes


@dataclass
class _IndexState:
    """Cached index plus the validity report it was built from.

    ``artifact`` is the published generation, ``params`` the hash of that
    generation's parameters and ``catalog`` a snapshot of the catalog fields that
    feed the input fingerprints. All three are captured from the *same* loaded
    generation; caching on the artifact alone would keep answering from stale
    validity data when a record or the window policy changes.
    """

    artifact: str | None
    params: str
    catalog: str
    index: ExactVectorIndex
    validity: ValidityReport
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _apply_hard_filters(plan: QueryPlan, asset: AssetRecord) -> bool:
    filters = plan.filters
    if filters.library_ids and asset.library_id not in filters.library_ids:
        return False
    duration = asset.duration_seconds
    if filters.duration_min_seconds is not None and (
        duration is None or duration < filters.duration_min_seconds
    ):
        return False
    if filters.duration_max_seconds is not None and (
        duration is None or duration > filters.duration_max_seconds
    ):
        return False
    if filters.sample_rates_hz and asset.sample_rate_hz not in filters.sample_rates_hz:
        return False
    if filters.channel_counts and asset.channel_count not in filters.channel_counts:
        return False
    if filters.online_only and not any(loc.availability == "online" for loc in asset.locations):
        return False
    return True


def _to_search_result(
    *,
    rank: int,
    asset: AssetRecord,
    hit: VectorHit,
    source: RetrievalSource,
    explanation: list[str],
) -> SearchResult:
    """Build a contract result, keeping the hit's original-file range when valid."""
    matched: SourceRange | None = None
    if hit.end_seconds > hit.start_seconds:
        candidate = SourceRange(start_seconds=hit.start_seconds, end_seconds=hit.end_seconds)
        try:
            validate_source_range(asset, candidate)
        except ValueError:
            matched = None
        else:
            matched = candidate
    return SearchResult(
        asset=asset,
        rank=rank,
        sources=[source],
        matched_range=matched,
        explanation=explanation,
        score=round(hit.score, 6),
    )


class _VectorRetrieverBase:
    """Shared artifact loading: filter stale rows, cache by artifact and catalog."""

    source: RetrievalSource

    def __init__(self, catalog: ManifestCatalog, store: EmbeddingStore) -> None:
        self.catalog = catalog
        self.store = store
        self._state: _IndexState | None = None

    # -- artifact ------------------------------------------------------------
    def _catalog_inputs(self) -> list[str]:
        """Per-asset material that determines its input fingerprint.

        Kept separate from ``_catalog_fingerprints`` so the cache check stays cheap:
        building this list never hashes per asset and, for audio, never plans
        windows. It must depend on exactly the same fields — and on the same
        generation's window parameters — as the fingerprint itself.
        """
        raise NotImplementedError

    def _catalog_fingerprints(self) -> dict[str, str]:
        raise NotImplementedError

    def _catalog_token(self) -> str:
        material = "\n".join(self._catalog_inputs())
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _params_token(self) -> str:
        """Hash of the window/preprocessing parameters of the loaded generation."""
        payload = json.dumps(
            self.store.params, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _require_store(self) -> None:
        if self.store.generation is None:
            raise ModelUnavailableError(
                f"no vectors at {self.store.directory}; run the encode step first"
            )

    def index_state(self, *, force: bool = False) -> _IndexState:
        """Load/reload the artifact and return only rows valid for this catalog.

        The cache is keyed on the artifact generation, the generation's parameters
        and the catalog input snapshot, so that a re-encode, a different window
        policy or a changed record is picked up on the next call.

        Ordering matters: window parameters live in the artifact's header, so the
        generation must be loaded *before* any fingerprint is derived from it.
        Otherwise a non-default policy would be checked against the default one on
        the first query after opening a store or republishing it.
        """
        self._require_store()
        artifact = self.store.generation
        if (
            not force
            and self._state is not None
            and self._state.artifact == artifact
            and self._state.params == self._params_token()
        ):
            # The header in memory is the one this state was built from; only the
            # catalog may have moved since.
            if self._state.catalog == self._catalog_token():
                return self._state

        # First use, new generation, or rewritten header: read this generation
        # before deriving parameters or fingerprints from it.
        self.store.load()
        if self.store.is_empty:
            raise ModelUnavailableError(
                f"no vectors at {self.store.directory}; run the encode step first"
            )
        artifact = self.store.generation
        params_token = self._params_token()
        space = self._space()
        if self.store.space_id != space.space_id:
            # A different model, revision or preprocessing: never mix the spaces.
            raise SpaceMismatchError(space.space_id, str(self.store.space_id))
        catalog_token = self._catalog_token()
        fingerprints = self._catalog_fingerprints()
        validity = self.store.validity(space, list(self.catalog.assets), fingerprints)
        batch = self.store.subset(set(validity.valid))
        warnings: list[str] = []
        if validity.unverified:
            warnings.append(
                f"{len(validity.unverified)} vectors come from a legacy artifact without input "
                "fingerprints; they are excluded until the artifact is re-encoded"
            )
        self._state = _IndexState(
            artifact=artifact,
            params=params_token,
            catalog=catalog_token,
            index=ExactVectorIndex.from_batch(batch),
            validity=validity,
            warnings=tuple(warnings),
        )
        return self._state

    def _space(self):
        """The embedding space this artifact declares."""
        raise NotImplementedError

    def _outcome(self, state: _IndexState, results: list[SearchResult]) -> RetrievalOutcome:
        return RetrievalOutcome(
            results=results,
            state="ready",
            stale_assets=len(state.validity.stale),
            missing_assets=len(state.validity.missing),
            unverified_assets=len(state.validity.unverified),
        )

    def _empty_reason(self, state: _IndexState, *, what: str) -> str | None:
        """Explain an empty candidate set instead of reporting a clean zero."""
        if state.index.size == 0:
            counts = state.validity.as_dict()
            if counts["unverified"]:
                return (
                    f"no verified {what}: {counts['unverified']} rows come from a legacy "
                    "artifact without input fingerprints; re-encode to enable search"
                )
            return (
                f"no usable {what} for the current catalog "
                f"(stale={counts['stale']}, missing={counts['missing']}); re-run the encode step"
            )
        return None


class TextVectorRetriever(_VectorRetrieverBase):
    """BGE-M3 text space: natural-language query -> asset text vectors."""

    source = RetrievalSource.TEXT

    def __init__(
        self, catalog: ManifestCatalog, store: EmbeddingStore, encoder: BgeM3TextEncoder
    ) -> None:
        super().__init__(catalog, store)
        self.encoder = encoder

    def _catalog_inputs(self) -> list[str]:
        return [
            f"{asset.asset_id}\x00{asset.content_revision}\x00{self._text_of(asset)}"
            for asset in self.catalog.assets
        ]

    @staticmethod
    def _text_of(asset: AssetRecord) -> str:
        """The exact string the text fingerprint is taken over."""
        return plan_asset_text(asset).text

    def _catalog_fingerprints(self) -> dict[str, str]:
        return {str(asset.asset_id): text_input_fingerprint(asset) for asset in self.catalog.assets}

    def _space(self):
        if self.encoder.space is None:
            self.encoder.load()
        return self.encoder.space

    def index(self) -> ExactVectorIndex:
        return self.index_state().index

    def retrieve(self, plan: QueryPlan, *, limit: int) -> list[SearchResult]:
        return self.search(plan, limit=limit).results

    def search(self, plan: QueryPlan, *, limit: int) -> RetrievalOutcome:
        query_text = plan.text_query or plan.original_text
        if not query_text.strip():
            return RetrievalOutcome([], "unavailable", "empty text query")
        state = self.index_state()
        reason = self._empty_reason(state, what="text vectors")
        if reason is not None:
            return RetrievalOutcome(
                [],
                "unavailable",
                reason,
                stale_assets=len(state.validity.stale),
                missing_assets=len(state.validity.missing),
                unverified_assets=len(state.validity.unverified),
            )
        vector = self.encoder.encode_query(query_text)
        allowed = {
            str(asset.asset_id) for asset in self.catalog.assets if _apply_hard_filters(plan, asset)
        } & set(state.validity.valid)
        if not allowed:
            return RetrievalOutcome([], "ready", "no asset matches the hard filters")
        hits = state.index.search(
            vector, space=self.encoder.space, limit=max(limit, 1), allowed_assets=allowed
        )
        results = []
        for rank, hit in enumerate(hits, start=1):
            asset = self.catalog.get_asset(UUID(hit.asset_id))
            if asset is None:  # pragma: no cover - validity already checked membership
                continue
            results.append(
                _to_search_result(
                    rank=rank,
                    asset=asset,
                    hit=hit,
                    source=self.source,
                    explanation=[
                        f"BGE-M3 text cosine {hit.score:.4f}",
                        f"query: {query_text[:120]}",
                    ],
                )
            )
        return self._outcome(state, results)


class AudioVectorRetriever(_VectorRetrieverBase):
    """CLAP audio space: text->audio and audio->audio in one paired space."""

    source = RetrievalSource.AUDIO

    def __init__(
        self, catalog: ManifestCatalog, store: EmbeddingStore, encoder: ClapEncoder
    ) -> None:
        super().__init__(catalog, store)
        self.encoder = encoder

    # -- window policy actually recorded in the artifact --------------------
    def _window_params(self) -> tuple[float, float]:
        """Window seconds/hop the artifact was built with, not today's defaults.

        Reads the header of the currently loaded generation; callers must ensure
        ``store.load()`` has run for the generation they care about.
        """
        params = self.store.params
        window = params.get("window_seconds", AUDIO_WINDOW_SECONDS)
        hop = params.get("hop_seconds", AUDIO_HOP_SECONDS)
        return float(window), float(hop)

    def _catalog_inputs(self) -> list[str]:
        window, hop = self._window_params()
        return [
            f"{asset.asset_id}\x00{asset.content_revision}\x00"
            f"{float(asset.duration_seconds or 0.0)!r}\x00{window!r}\x00{hop!r}"
            for asset in self.catalog.assets
        ]

    def _catalog_fingerprints(self) -> dict[str, str]:
        window, hop = self._window_params()
        return {
            str(asset.asset_id): audio_input_fingerprint(
                plan_asset_windows(asset, window_seconds=window, hop_seconds=hop)
            )
            for asset in self.catalog.assets
        }

    def _space(self):
        if self.encoder.space is None:
            self.encoder.load()
        return self.encoder.space

    def index(self) -> ExactVectorIndex:
        return self.index_state().index

    def retrieve(self, plan: QueryPlan, *, limit: int) -> list[SearchResult]:
        return self.search_text(plan, limit=limit).results

    def search_text(self, plan: QueryPlan, *, limit: int) -> RetrievalOutcome:
        """Text description -> audio windows (text and audio tower of the same checkpoint)."""
        descriptions = plan.audio_descriptions or (
            [plan.text_query] if plan.text_query else [plan.original_text]
        )
        descriptions = [text for text in descriptions if text and text.strip()]
        if not descriptions:
            return RetrievalOutcome([], "unavailable", "empty audio description")
        state = self.index_state()
        reason = self._empty_reason(state, what="audio vectors")
        if reason is not None:
            return RetrievalOutcome(
                [],
                "unavailable",
                reason,
                stale_assets=len(state.validity.stale),
                missing_assets=len(state.validity.missing),
                unverified_assets=len(state.validity.unverified),
            )
        allowed = {
            str(asset.asset_id) for asset in self.catalog.assets if _apply_hard_filters(plan, asset)
        } & set(state.validity.valid)
        if not allowed:
            return RetrievalOutcome([], "ready", "no asset matches the hard filters")

        # Collapse to one window per asset inside the scan, so a single long file
        # cannot consume the candidate budget before other assets are considered.
        best: dict[str, VectorHit] = {}
        for vector in self.encoder.encode_texts(descriptions):
            for hit in state.index.search(
                vector,
                space=self.encoder.space,
                limit=max(limit, 1),
                allowed_assets=allowed,
                collapse_per_asset=True,
            ):
                current = best.get(hit.asset_id)
                if current is None or hit.score > current.score:
                    best[hit.asset_id] = hit
        ordered = sorted(best.values(), key=lambda hit: -hit.score)[: max(limit, 1)]
        return self._outcome(
            state,
            self._build(ordered, query_label=" + ".join(descriptions)[:160]),
        )

    def search_reference(
        self,
        ref_asset_id: UUID | str,
        *,
        start_seconds: float,
        end_seconds: float,
        limit: int = 20,
        exclude_self: bool = True,
    ) -> RetrievalOutcome:
        """Reference sound -> similar windows, using the CLAP audio tower."""
        state = self.index_state()
        reason = self._empty_reason(state, what="audio vectors")
        if reason is not None:
            return RetrievalOutcome(
                [],
                "unavailable",
                reason,
                stale_assets=len(state.validity.stale),
                missing_assets=len(state.validity.missing),
                unverified_assets=len(state.validity.unverified),
            )
        asset = self.catalog.require_asset(UUID(str(ref_asset_id)))
        ref_uuid = asset.asset_id
        path: Path = self.catalog.path_for(asset)
        from audiosearch.audioio.decode import open_wav  # noqa: PLC0415 - keeps import light

        with open_wav(path) as reader:
            samples, _ = reader.read_range(start_seconds, end_seconds)
            rate = reader.sample_rate_hz
        mono = samples.mean(axis=1, dtype=np.float32) if samples.shape[1] > 1 else samples[:, 0]
        # A reference longer than one CLAP window would be rejected outright by the
        # checkpoint's feature extractor (it only truncates on its own default path).
        # Take the first window: that is the length the model was trained on, and it
        # keeps this route identical to how the stored windows were encoded.
        limit_samples = int(round(AUDIO_WINDOW_SECONDS * rate))
        mono = mono[:limit_samples]
        logger.info(
            "reference encode: %d samples @ %d Hz (%.3f s), window limit %.1f s",
            mono.shape[0],
            rate,
            mono.shape[0] / rate if rate else 0.0,
            AUDIO_WINDOW_SECONDS,
        )
        vector = self.encoder.encode_audio_mono(mono[None, :], rate)[0]
        allowed = {
            str(item.asset_id)
            for item in self.catalog.assets
            if not (exclude_self and item.asset_id == ref_uuid)
        } & set(state.validity.valid)
        hits = state.index.search(
            vector,
            space=self.encoder.space,
            limit=max(limit, 1),
            allowed_assets=allowed,
            collapse_per_asset=True,
        )
        return self._outcome(
            state,
            self._build(
                hits,
                query_label=f"reference {asset.name} [{start_seconds:.2f}, {end_seconds:.2f})",
            ),
        )

    def _build(self, hits: list[VectorHit], *, query_label: str) -> list[SearchResult]:
        """One result per asset; hits are already the best window per asset."""
        results: list[SearchResult] = []
        for hit in hits:
            asset = self.catalog.get_asset(UUID(hit.asset_id))
            if asset is None:  # pragma: no cover - validity already checked membership
                continue
            explanation = [
                f"CLAP audio cosine {hit.score:.4f}",
                f"best window [{hit.start_seconds:.2f}, {hit.end_seconds:.2f}) s",
                f"coverage {hit.coverage:.2f}",
                f"query: {query_label}",
            ]
            results.append(
                _to_search_result(
                    rank=len(results) + 1,
                    asset=asset,
                    hit=hit,
                    source=self.source,
                    explanation=explanation,
                )
            )
        return results


__all__ = [
    "AudioVectorRetriever",
    "RetrievalOutcome",
    "TextVectorRetriever",
]
