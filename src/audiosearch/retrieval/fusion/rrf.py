"""Reciprocal Rank Fusion over independent retrieval routes.

Each route sees a different signal (keyword BM25, BGE-M3 text cosine, CLAP audio
cosine) and its scores are not comparable. RRF therefore uses only *ranks*:

    score(asset) = Σ_routes  weight_route / (k + rank_route(asset))

That is the published baseline; the weights stay explicit and un-tuned so a first
run can be reported honestly. The fusion output keeps, per route, the original
hit so the matched source range survives fusion.

Range selection is *not* "first route wins". A keyword or text route can only
report the whole asset ``[0, duration)``, while the audio route knows which window
actually matched. Fusion therefore prefers the audio route's window and only
falls back to a coarser range when no audio route contributed one, so changing
the caller's route order cannot change the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from audiosearch.contracts.models import RetrievalSource

DEFAULT_RRF_K = 60

# Routes that can locate a window inside the file, best first. Keyword and text
# routes encode the record, not the audio event, so they only offer whole assets.
_RANGE_PRIORITY: tuple[RetrievalSource, ...] = (
    RetrievalSource.AUDIO,
    RetrievalSource.KEYWORD,
    RetrievalSource.TEXT,
)


@dataclass(frozen=True, slots=True)
class RouteHit:
    """One route's view of one asset, already collapsed to its best segment."""

    asset_id: str
    rank: int
    score: float | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class RouteResult:
    source: RetrievalSource
    state: Literal["ready", "unavailable", "failed"]
    hits: tuple[RouteHit, ...] = ()
    reason: str | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class FusedHit:
    asset_id: str
    rank: int
    rrf_score: float
    sources: tuple[RetrievalSource, ...]
    per_route: dict[RetrievalSource, RouteHit] = field(default_factory=dict)
    # Original-file range with the best source: audio window over whole asset.
    start_seconds: float | None = None
    end_seconds: float | None = None

    def explanation(self) -> list[str]:
        parts = [
            f"{source.value}: rank {hit.rank}"
            + (f" (score {hit.score:.4f})" if hit.score is not None else "")
            for source, hit in sorted(self.per_route.items(), key=lambda item: item[1].rank)
        ]
        return parts + [f"rrf {self.rrf_score:.6f}"]


def _range_rank(source: RetrievalSource) -> int:
    try:
        return _RANGE_PRIORITY.index(source)
    except ValueError:  # pragma: no cover - every source is listed above
        return len(_RANGE_PRIORITY)


def _pick_range(
    hits: dict[RetrievalSource, RouteHit],
) -> tuple[float, float] | None:
    """Choose the most specific source range among the routes that matched.

    Ordering is by route capability, then by hit rank, and never by the order the
    caller passed the routes in. A window from the audio route therefore always
    wins over a whole-asset range from the text or keyword route.
    """
    candidates = [
        (hit, source)
        for source, hit in hits.items()
        if hit.start_seconds is not None and hit.end_seconds is not None
    ]
    if not candidates:
        return None
    hit, _source = min(candidates, key=lambda item: (_range_rank(item[1]), item[0].rank))
    return (hit.start_seconds, hit.end_seconds)  # type: ignore[return-value]


def fuse(
    routes: list[RouteResult],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int = 50,
) -> list[FusedHit]:
    """Fuse ranked routes with RRF; unavailable routes contribute nothing."""
    if k < 1:
        raise ValueError("k must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    accumulator: dict[str, float] = {}
    per_route: dict[str, dict[RetrievalSource, RouteHit]] = {}
    for route in routes:
        if route.state != "ready" or not route.hits:
            continue
        for hit in route.hits:
            if hit.rank < 1:
                raise ValueError("route hit ranks are 1-based")
            already = per_route.setdefault(hit.asset_id, {}).get(route.source)
            if already is not None:
                # One route can repeat an asset (e.g. several windows). Only its
                # best rank may contribute, so RRF cannot be inflated by repeats
                # and the range picker sees the representative hit.
                if hit.rank >= already.rank:
                    continue
                accumulator[hit.asset_id] -= route.weight / (k + already.rank)
            accumulator[hit.asset_id] = accumulator.get(hit.asset_id, 0.0) + (
                route.weight / (k + hit.rank)
            )
            per_route[hit.asset_id][route.source] = hit
    ordered = sorted(
        accumulator.items(),
        key=lambda item: (-item[1], min(hit.rank for hit in per_route[item[0]].values()), item[0]),
    )
    fused: list[FusedHit] = []
    for rank, (asset_id, score) in enumerate(ordered[:limit], start=1):
        hits = per_route[asset_id]
        # Ordered by rank, so the strongest route is listed first.
        sources_by_rank = sorted(hits.items(), key=lambda item: item[1].rank)
        chosen = _pick_range(hits)
        fused.append(
            FusedHit(
                asset_id=asset_id,
                rank=rank,
                rrf_score=score,
                sources=tuple(source for source, _hit in sources_by_rank),
                per_route=dict(hits),
                start_seconds=chosen[0] if chosen else None,
                end_seconds=chosen[1] if chosen else None,
            )
        )
    return fused


def available_routes(routes: list[RouteResult]) -> list[RetrievalSource]:
    return [route.source for route in routes if route.state == "ready"]


__all__ = [
    "DEFAULT_RRF_K",
    "FusedHit",
    "RouteHit",
    "RouteResult",
    "available_routes",
    "fuse",
]
