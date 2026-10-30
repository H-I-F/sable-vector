"""Pre-filtering: resolve the predicate first, then score only the survivors.

A hybrid query is two problems wearing one coat — "which records match this
attribute predicate" and "which vectors are nearest" — and the order they are
solved in decides how much of the collection is touched. Pre-filtering answers
the predicate first, through the inverted index, and hands the surviving ids to
the vector kernel as a candidate block.

This is the right ordering when the predicate is selective. Scoring costs one
gemv over the candidate block, so narrowing from 50k rows to 400 costs 99% of
the arithmetic; the predicate itself is a prefix scan whose cost is proportional
to the answer. It is the wrong ordering when the predicate matches almost
everything, because then the candidate block is the whole matrix plus the
overhead of building it — which is the case :mod:`sable_vector.query.postfilter`
exists for.

Correctness does not depend on the ordering: both return the *k* nearest records
that satisfy the predicate. What differs is how much work that costs, which is
why :attr:`PrefilterOutcome.scored` is part of the result rather than a debug
side-channel — an executor that quietly scores everything is silently the other
executor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.storage import VectorStorage
from sable_vector.types import Match, MetadataFilter

__all__ = ["PrefilterExecutor", "PrefilterOutcome"]


@dataclass(frozen=True, slots=True)
class PrefilterOutcome:
    """Result of one pre-filtered query.

    Attributes:
        matches: Up to ``k`` nearest records satisfying the predicate, payloads
            attached, best first.
        candidates: How many ids survived the predicate, or ``None`` when the
            predicate imposed no restriction at all.
        scored: Rows actually handed to the vector kernel. Equal to
            ``candidates`` when restricted and to the corpus size otherwise —
            the number that says whether pre-filtering paid off.
    """

    matches: tuple[Match, ...]
    candidates: int | None
    scored: int


class PrefilterExecutor:
    """Predicate first, geometry second.

    Args:
        storage: Source of payloads for the surviving matches.
        flat: Vector index to score the candidate block with.
        scalar: Inverted index that resolves the predicate.
    """

    def __init__(self, storage: VectorStorage, flat: FlatIndex, scalar: ScalarIndex) -> None:
        self._storage = storage
        self._flat = flat
        self._scalar = scalar

    @property
    def flat(self) -> FlatIndex:
        return self._flat

    @property
    def scalar(self) -> ScalarIndex:
        return self._scalar

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        predicate: MetadataFilter | None = None,
    ) -> PrefilterOutcome:
        """Nearest ``k`` records satisfying ``predicate``.

        Args:
            query: ``(d,)`` embedding.
            k: How many neighbours to return.
            predicate: Attribute filter; ``None`` or empty means unrestricted —
                in which case every row is scored and this executor degenerates
                to a plain Flat search, which is honest but not its purpose.

        Returns:
            A :class:`PrefilterOutcome`.

        Raises:
            ValueError: The predicate names a field that is not indexed, or an
                operator this engine cannot evaluate.
        """
        allowed = self._scalar.matching_ids(predicate)
        candidates = None if allowed is None else len(allowed)

        matches = self._flat.search(query, k, candidate_ids=allowed)
        scored = candidates if candidates is not None else self._flat.size

        return PrefilterOutcome(
            matches=tuple(self._with_payloads(matches)),
            candidates=candidates,
            scored=scored,
        )

    def candidate_ids(self, predicate: MetadataFilter | None) -> set[str] | None:
        """Ids surviving ``predicate``, exposed for planning and for tests."""
        return self._scalar.matching_ids(predicate)

    def _with_payloads(self, matches: list[Match]) -> list[Match]:
        """Attach the stored payload to each match.

        A payload is missing only if the record was deleted between scoring and
        reading, in which case the honest answer is an empty payload rather than
        a stale one.
        """
        return [
            Match(match.id, match.score, match.distance, self._storage.get_payload(match.id) or {})
            for match in matches
        ]
