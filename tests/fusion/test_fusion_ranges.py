"""Range selection in RRF fusion.

The audio route is the only one that can point at a window inside a file; keyword
and text routes report the whole asset. Fusion must keep the audio window no
matter in which order the caller passed the routes, otherwise a future three-way
merge silently degrades every hit to ``[0, duration)``.
"""

from __future__ import annotations

import pytest

from audiosearch.contracts.models import RetrievalSource
from audiosearch.retrieval.fusion.rrf import RouteHit, RouteResult, fuse

DURATION = 42.0
WINDOW = (7.5, 17.5)


def whole_asset(source: RetrievalSource, rank: int = 1) -> RouteResult:
    return RouteResult(
        source=source,
        state="ready",
        hits=(RouteHit("asset-a", rank, score=0.5, start_seconds=0.0, end_seconds=DURATION),),
    )


def audio_window(rank: int = 3) -> RouteResult:
    return RouteResult(
        source=RetrievalSource.AUDIO,
        state="ready",
        hits=(
            RouteHit("asset-a", rank, score=0.61, start_seconds=WINDOW[0], end_seconds=WINDOW[1]),
        ),
    )


@pytest.mark.parametrize(
    "routes",
    [
        pytest.param([whole_asset(RetrievalSource.TEXT), audio_window()], id="text-then-audio"),
        pytest.param([audio_window(), whole_asset(RetrievalSource.TEXT)], id="audio-then-text"),
        pytest.param(
            [
                whole_asset(RetrievalSource.KEYWORD),
                whole_asset(RetrievalSource.TEXT),
                audio_window(),
            ],
            id="keyword-text-audio",
        ),
        pytest.param(
            [
                audio_window(),
                whole_asset(RetrievalSource.TEXT),
                whole_asset(RetrievalSource.KEYWORD),
            ],
            id="audio-text-keyword",
        ),
    ],
)
def test_audio_window_survives_fusion_in_any_route_order(routes):
    fused = fuse(routes, limit=5)
    assert len(fused) == 1
    hit = fused[0]
    assert (hit.start_seconds, hit.end_seconds) == WINDOW
    assert RetrievalSource.AUDIO in hit.sources


def test_whole_asset_range_is_kept_when_no_audio_window_exists():
    fused = fuse([whole_asset(RetrievalSource.TEXT)], limit=5)
    assert (fused[0].start_seconds, fused[0].end_seconds) == (0.0, DURATION)


def test_hit_without_any_range_stays_empty():
    route = RouteResult(
        source=RetrievalSource.TEXT, state="ready", hits=(RouteHit("asset-a", 1, score=0.2),)
    )
    fused = fuse([route], limit=5)
    assert fused[0].start_seconds is None
    assert fused[0].end_seconds is None


def test_audio_range_wins_even_when_it_ranks_worse():
    # Rank 9 audio vs rank 1 text: specificity must beat rank here.
    fused = fuse([whole_asset(RetrievalSource.TEXT, rank=1), audio_window(rank=9)], limit=5)
    assert (fused[0].start_seconds, fused[0].end_seconds) == WINDOW


def test_two_audio_windows_use_the_better_ranked_one():
    routes = [
        RouteResult(
            source=RetrievalSource.AUDIO,
            state="ready",
            hits=(
                RouteHit("asset-a", 1, score=0.9, start_seconds=0.0, end_seconds=10.0),
                RouteHit("asset-a", 4, score=0.7, start_seconds=30.0, end_seconds=40.0),
            ),
        )
    ]
    fused = fuse(routes, limit=5)
    assert (fused[0].start_seconds, fused[0].end_seconds) == (0.0, 10.0)


def test_unavailable_audio_route_does_not_block_text_range():
    routes = [
        RouteResult(
            source=RetrievalSource.AUDIO,
            state="unavailable",
            reason="CLAP not installed",
            hits=(RouteHit("asset-a", 1, score=0.99, start_seconds=1.0, end_seconds=2.0),),
        ),
        whole_asset(RetrievalSource.TEXT),
    ]
    fused = fuse(routes, limit=5)
    assert (fused[0].start_seconds, fused[0].end_seconds) == (0.0, DURATION)
