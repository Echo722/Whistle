"""Exact vector search with an explicit space guard.

Small-sample work needs *exact* neighbours, not an approximate index, so this is
a full matrix product over float32 rows that are already L2-normalized: the dot
product is the cosine similarity. Two rules are enforced rather than assumed:

- Query and corpus must share one ``space_id``; otherwise ``SpaceMismatchError``.
- Every returned hit carries the original-file segment it came from, so a
  result can be auditioned without re-deriving time positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow
from audiosearch.inference.spaces import EmbeddingSpace, SpaceMismatchError


class DimensionError(ValueError):
    """Query vector width does not match the indexed space."""


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One nearest neighbour before per-asset collapsing."""

    asset_id: str
    score: float
    row_index: int
    start_seconds: float
    end_seconds: float
    coverage: float

    @property
    def is_segment(self) -> bool:
        return self.start_seconds != 0.0 or self.end_seconds != 0.0


@dataclass
class ExactVectorIndex:
    """Immutable exact index over one embedding batch."""

    space: EmbeddingSpace
    vectors: np.ndarray
    rows: list[EmbeddingRow]

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be two-dimensional")
        if self.vectors.shape[1] != self.space.dimension:
            raise DimensionError(
                f"index width {self.vectors.shape[1]} != space dimension {self.space.dimension}"
            )
        if self.vectors.shape[0] != len(self.rows):
            raise ValueError("vectors and rows disagree")

    @classmethod
    def from_batch(cls, batch: EmbeddingBatch) -> ExactVectorIndex:
        return cls(space=batch.space, vectors=batch.vectors, rows=list(batch.rows))

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def asset_count(self) -> int:
        return len({row.asset_id for row in self.rows})

    def search(
        self,
        query: np.ndarray,
        *,
        space: EmbeddingSpace | None = None,
        limit: int = 20,
        allowed_assets: set[str] | None = None,
        collapse_per_asset: bool = True,
    ) -> list[VectorHit]:
        """Return the highest-cosine rows; optionally one best row per asset."""
        if space is not None and space.space_id != self.space.space_id:
            raise SpaceMismatchError(space.space_id, self.space.space_id)
        if limit < 1:
            raise ValueError("limit must be positive")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.space.dimension:
            raise DimensionError(
                f"query width {vector.shape[0]} != space dimension {self.space.dimension}"
            )
        if self.size == 0:
            return []
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("query vector is empty or non-finite")
        scores = self.vectors @ (vector / norm)
        order = np.argsort(-scores, kind="stable")
        hits: list[VectorHit] = []
        seen: set[str] = set()
        for index in order:
            row = self.rows[int(index)]
            if allowed_assets is not None and row.asset_id not in allowed_assets:
                continue
            if collapse_per_asset:
                if row.asset_id in seen:
                    continue
                seen.add(row.asset_id)
            hits.append(
                VectorHit(
                    asset_id=row.asset_id,
                    score=float(scores[int(index)]),
                    row_index=int(index),
                    start_seconds=row.start_seconds,
                    end_seconds=row.end_seconds,
                    coverage=row.coverage,
                )
            )
            if len(hits) >= limit:
                break
        return hits


__all__ = ["DimensionError", "ExactVectorIndex", "VectorHit"]
