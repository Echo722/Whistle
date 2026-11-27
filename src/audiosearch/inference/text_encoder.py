"""BGE-M3 text embedding over existing names and metadata only.

What is encoded is exactly the material the catalog already has: the asset name,
the package name, and existing description/keywords when a source supplied them.
Nothing is invented — an asset without a description simply has fewer fields and
is marked ``sparse_text`` so later ranking can tell the two apart.

Template version ``bge-m3-fields-v1`` is part of the vector space id: changing it
invalidates every vector produced under it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from audiosearch.contracts.models import AssetRecord

from .runtime import plan_device, require_torch, require_transformers
from .spaces import EmbeddingSpace, ModelFingerprint, ModelUnavailableError, l2_normalize

TEMPLATE_VERSION = "bge-m3-fields-v1"
DEFAULT_TEXT_MODEL = "BAAI/bge-m3"
# BGE-M3 dense head output width; read back from the config at load time.
EXPECTED_DIMENSION = 1024


@dataclass(frozen=True, slots=True)
class TextFieldPlan:
    """The text to encode for one asset, plus how much of it was real."""

    asset_id: str
    text: str
    fields_used: tuple[str, ...]

    @property
    def is_sparse(self) -> bool:
        """True when only the file name/package were available."""
        return "name" in self.fields_used and not {"description", "keywords"} & set(
            self.fields_used
        )


def plan_asset_text(asset: AssetRecord) -> TextFieldPlan:
    """Build the short, templated field text for one asset."""
    parts: list[str] = [asset.name]
    fields: list[str] = ["name"]
    if asset.description:
        parts.append(asset.description)
        fields.append("description")
    if asset.keywords:
        parts.append(", ".join(asset.keywords))
        fields.append("keywords")
    if asset.package:
        parts.append(asset.package)
        fields.append("package")
    return TextFieldPlan(
        asset_id=str(asset.asset_id),
        text=" | ".join(part.strip() for part in parts if part and part.strip()),
        fields_used=tuple(fields),
    )


@dataclass
class TextEncoderConfig:
    model_id: str = DEFAULT_TEXT_MODEL
    revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    device: str = "auto"
    batch_size: int = 8
    max_tokens: int = 256
    # BGE-M3 needs no query instruction: its model card documents plain encoding
    # for both queries and documents (unlike the earlier ``bge-*-en`` models, which
    # do want "Represent this sentence..."). Kept configurable so an A/B can be run,
    # but the default stays empty on purpose.
    query_instruction: str = ""


class BgeM3TextEncoder:
    """Dense BGE-M3 text tower with documented preprocessing."""

    def __init__(self, config: TextEncoderConfig | None = None, *, local_only: bool = True):
        self.config = config or TextEncoderConfig()
        self.local_only = local_only
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self.space: EmbeddingSpace | None = None
        self.fingerprint: ModelFingerprint | None = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> EmbeddingSpace:
        torch = require_torch()
        transformers = require_transformers()
        device = plan_device(self.config.device)
        self._torch = torch
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.local_only,
            )
            model = transformers.AutoModel.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.local_only,
            )
        except OSError as exc:
            raise ModelUnavailableError(
                f"text model {self.config.model_id}@{self.config.revision} "
                f"is not available locally: {type(exc).__name__}"
            ) from exc
        model.eval()
        model.to(device.device)
        self._tokenizer = tokenizer
        self._model = model
        dimension = int(model.config.hidden_size)
        self.space = EmbeddingSpace(
            model_id=self.config.model_id,
            revision=self.config.revision,
            tower="text",
            preprocessing_version=f"{TEMPLATE_VERSION}/max{self.config.max_tokens}",
            dimension=dimension,
            normalized=True,
        )
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.fingerprint = ModelFingerprint(
            model_id=self.config.model_id,
            revision=self.config.revision,
            library="transformers",
            library_version=transformers.__version__,
            device=device.device,
            dtype=device.dtype,
            weights_bytes=params_bytes,
        )
        return self.space

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else "unloaded"

    # -- encoding -----------------------------------------------------------
    def encode_texts(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """Encode raw strings; returns L2-normalized float32 rows."""
        if self._model is None or self._tokenizer is None:
            raise ModelUnavailableError("text encoder is not loaded; call load() first")
        if not texts:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        torch = self._torch
        size = batch_size or self.config.batch_size
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens,
                return_tensors="pt",
            )
            tokens = {key: value.to(self._model.device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = self._model(**tokens)
            # BGE-M3 dense embedding is the CLS state; normalize for cosine.
            hidden = output.last_hidden_state[:, 0, :]
            chunks.append(hidden.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode_assets(self, assets: list[AssetRecord]) -> tuple[np.ndarray, list[TextFieldPlan]]:
        plans = [plan_asset_text(asset) for asset in assets]
        return self.encode_texts([plan.text for plan in plans]), plans

    def encode_query(self, query: str) -> np.ndarray:
        if self.config.query_instruction:
            query = f"{self.config.query_instruction}{query}"
        return self.encode_texts([query])[0]


def describe_space(space: EmbeddingSpace) -> dict[str, object]:
    return space.as_dict()


__all__ = [
    "DEFAULT_TEXT_MODEL",
    "EXPECTED_DIMENSION",
    "TEMPLATE_VERSION",
    "BgeM3TextEncoder",
    "TextFieldPlan",
    "TextEncoderConfig",
    "describe_space",
    "plan_asset_text",
]
