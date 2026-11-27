"""Model identity and vector-space bookkeeping.

A vector only means something inside the space that produced it. The space id
joins the model, its immutable revision, the tower (text or audio) and the
preprocessing version, so two vectors can be compared only when their ids are
equal. This is what keeps BGE-M3 text vectors from ever being mixed with CLAP
vectors, even where the dimensions happen to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from audiosearch.contracts.models import ErrorCode

Tower = Literal["text", "audio"]


class ModelUnavailableError(Exception):
    """Model, weights or the inference runtime are not installed/loadable."""

    code = ErrorCode.MODEL_UNAVAILABLE

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SpaceMismatchError(ValueError):
    """Two vectors from different spaces were about to be compared."""

    def __init__(self, left: str, right: str) -> None:
        super().__init__(f"vector spaces differ: {left!r} != {right!r}")
        self.left = left
        self.right = right


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Immutable identity of one embedding space."""

    model_id: str
    revision: str
    tower: Tower
    preprocessing_version: str
    dimension: int
    normalized: bool

    @property
    def space_id(self) -> str:
        return f"{self.model_id}@{self.revision[:12]}/{self.tower}/{self.preprocessing_version}"

    def require_same(self, other: EmbeddingSpace) -> None:
        if self.space_id != other.space_id:
            raise SpaceMismatchError(self.space_id, other.space_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "space_id": self.space_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "tower": self.tower,
            "preprocessing_version": self.preprocessing_version,
            "dimension": self.dimension,
            "normalized": self.normalized,
        }


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """What was loaded, not what was requested."""

    model_id: str
    revision: str | None
    library: str
    library_version: str
    device: str
    dtype: str
    weights_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "library": self.library,
            "library_version": self.library_version,
            "device": self.device,
            "dtype": self.dtype,
            "weights_bytes": self.weights_bytes,
        }


def l2_normalize(matrix, *, eps: float = 1e-12):
    """Row-wise L2 normalization; raises instead of returning a zero row."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("cannot normalize a zero vector")
    return (matrix / norms).astype(np.float32)


__all__ = [
    "EmbeddingSpace",
    "ModelFingerprint",
    "ModelUnavailableError",
    "SpaceMismatchError",
    "Tower",
    "l2_normalize",
]
