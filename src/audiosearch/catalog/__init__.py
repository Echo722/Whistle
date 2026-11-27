"""Embedded WAV metadata parser (BWF bext, LIST/INFO, iXML)."""

from audiosearch.catalog.embedded import (
    Coverage,
    EmbeddedMetadata,
    EmbeddedRead,
    EmbeddedStatus,
    EmbeddedValue,
    read_embedded_metadata,
    read_embedded_verified,
)

__all__ = [
    "Coverage",
    "EmbeddedMetadata",
    "EmbeddedRead",
    "EmbeddedStatus",
    "EmbeddedValue",
    "read_embedded_metadata",
    "read_embedded_verified",
]
