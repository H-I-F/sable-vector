"""Domain primitives shared by every layer above the storage engine.

Two decisions here shape everything downstream:

* **A vector id is an opaque string.** Callers already have stable identifiers
  (chunk hashes, document keys, UUIDs); inventing a parallel numeric id space
  would only add a mapping that can fall out of sync with the payload it names.
* **A metric owns its comparison direction.** Callers ask for ``top-k``, not for
  "the smallest distances" or "the largest scores" depending on a switch they
  have to remember. :attr:`DistanceMetric.higher_is_better` and
  :attr:`DistanceMetric.worst_score` let one comparator serve all three metrics
  without a branch at every call site.

Ordering contract
-----------------
Filters are plain mappings. A scalar value means equality; a mapping value
means an operator expression, so ``{"tenant": "acme"}`` and
``{"tenant": {"$in": ["acme", "globex"]}}`` are the same shape of request and
the executor can decide between an inverted-index probe and a scan.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

import numpy as np

__all__ = [
    "DEFAULT_TOP_K",
    "FILTER_OPERATORS",
    "DistanceMetric",
    "Match",
    "MetadataFilter",
    "VectorId",
    "VectorRecord",
]

VectorId: TypeAlias = str
"""Opaque caller-supplied identifier. Never parsed, never generated here."""

MetadataFilter: TypeAlias = Mapping[str, Any]
"""Predicate over payload fields; scalar for equality, mapping for operators."""

DEFAULT_TOP_K: Final = 10

FILTER_OPERATORS: Final[frozenset[str]] = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}
)


class DistanceMetric(str, enum.Enum):
    """Similarity functions the engine can evaluate without a Python loop.

    ``str``-valued so a metric survives JSON configuration intact, which is what
    lets a persisted collection reopen with the geometry it was created with.
    """

    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"

    @property
    def higher_is_better(self) -> bool:
        """True when a larger number means a closer neighbour."""
        return self is not DistanceMetric.L2

    @property
    def worst_score(self) -> float:
        """Neutral element for a top-k accumulator under this metric."""
        return float("inf") if self is DistanceMetric.L2 else float("-inf")

    @property
    def normalised(self) -> bool:
        """True when inputs are projected onto the unit sphere before scoring."""
        return self is DistanceMetric.COSINE


@dataclass(slots=True)
class VectorRecord:
    """One vector plus the payload that travels with it.

    Attributes:
        id: Caller-supplied identifier, unique within a collection.
        vector: Embedding; coerced to contiguous little-endian float32 on
            construction so every downstream kernel sees the same layout.
        payload: Arbitrary JSON-shaped attributes used for filtering.
    """

    id: VectorId
    vector: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.vector, dtype="<f4")
        if array.ndim != 1:
            raise ValueError(f"vector must be one-dimensional, got shape {array.shape}")
        if array.size == 0:
            raise ValueError("vector must have at least one dimension")
        self.vector = array

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


@dataclass(frozen=True, slots=True)
class Match:
    """One neighbour returned by a search.

    Attributes:
        id: Identifier of the matched record.
        score: Metric score, already oriented so higher means closer: negated
            distance for L2, similarity otherwise.
        distance: Raw metric distance, lower is closer, for callers that need
            the geometric quantity rather than a ranking key.
        payload: Payload stored with the record.
    """

    id: VectorId
    score: float
    distance: float
    payload: dict[str, Any]
