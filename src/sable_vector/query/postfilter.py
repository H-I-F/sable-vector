"""Post-filtering: rank globally, then interrogate the payloads.

This is the opposite bet to :mod:`sable_vector.query.prefilter`, and the right
one under the opposite condition. Pre-filtering spends a prefix scan to avoid
scoring rows the predicate will reject; post-filtering scores everything and
spends a payload read per candidate *in ranking order*, stopping as soon as it
has *k* survivors. When the predicate matches most of the corpus the survivors
are at the top of the ranking, so it stops after roughly *k* reads and the
inverted index would have been pure overhead. When the predicate matches 2% of
the corpus it walks most of the ranking before filling *k*, and pre-filtering
wins by a wide margin.

Two properties make it more than an alternative ordering:

**It needs no index.** The predicate is evaluated against the payload itself, so
a field that was never declared indexed is still queryable — the escape hatch
for an ad-hoc predicate on a collection built without it.

**It agrees with pre-filtering on the answer.** Both return the *k* nearest
records satisfying the predicate; only the cost differs. That equivalence is the
contract worth testing, because a divergence would mean one of the two paths is
silently wrong about what "hybrid search" means.

Cost is therefore part of the result: :attr:`PostfilterOutcome.examined` counts
payload reads, which is the number that says whether this was the right bet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sable_vector.distance import as_query, score_matrix
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import evaluate_condition
from sable_vector.storage import VectorStorage, render_value
from sable_vector.types import Match, MetadataFilter

__all__ = ["PostfilterExecutor", "PostfilterOutcome"]


@dataclass(frozen=True, slots=True)
class PostfilterOutcome:
    """Result of one post-filtered query.

    Attributes:
        matches: Up to ``k`` nearest records satisfying the predicate, payloads
            attached, best first.
        examined: Payload reads performed — how far down the ranking the walk
            had to go. Roughly ``k`` for a permissive predicate, and the bulk of
            the corpus for a selective one.
        scored: Rows handed to the vector kernel; the whole corpus, which is the
            price of this ordering.
        complete: Whether ``k`` survivors were found before the ranking — or the
            caller's budget — ran out.
    """

    matches: tuple[Match, ...]
    examined: int
    scored: int
    complete: bool


class PostfilterExecutor:
    """Geometry first, predicate second.

    Args:
        storage: Source of payloads for the candidates the walk inspects.
        flat: Vector index to rank.
    """

    def __init__(self, storage: VectorStorage, flat: FlatIndex) -> None:
        self._storage = storage
        self._flat = flat

    @property
    def flat(self) -> FlatIndex:
        return self._flat

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        predicate: MetadataFilter | None = None,
        examine_budget: int | None = None,
    ) -> PostfilterOutcome:
        """Nearest ``k`` records satisfying ``predicate``.

        Args:
            query: ``(d,)`` embedding.
            k: How many neighbours to return.
            predicate: Attribute filter evaluated against each candidate's
                payload. ``None`` or empty means unrestricted.
            examine_budget: Cap on payload reads. ``None`` means "walk the whole
                ranking if that is what it takes", which is the honest default
                for correctness and a poor one for latency under a selective
                predicate — the caller decides.

        Returns:
            A :class:`PostfilterOutcome`.
        """
        flat = self._flat
        if k <= 0 or flat.size == 0:
            return PostfilterOutcome((), 0, 0, k <= 0)

        if not predicate:
            matches = flat.search(query, k)
            return PostfilterOutcome(
                matches=tuple(self._attach(matches)),
                examined=len(matches),
                scored=flat.size,
                complete=len(matches) >= k,
            )

        probe = as_query(query, dim=flat.dim)
        scores = score_matrix(flat.matrix, probe, flat.metric)
        # A full ordering, not a selection: the walk has to consume the ranking
        # from the top, and a partial sort cannot tell it which row is next.
        order = np.argsort(-scores, kind="stable")

        labels = flat.ids
        accepted: list[Match] = []
        examined = 0
        limit = len(order) if examine_budget is None else min(len(order), examine_budget)

        for position in order[:limit]:
            index = int(position)
            vector_id = labels[index]
            payload = self._storage.get_payload(vector_id) or {}
            examined += 1
            if not self._satisfies(payload, predicate):
                continue
            score = float(scores[index])
            accepted.append(Match(vector_id, score, -score, payload))
            if len(accepted) >= k:
                break

        return PostfilterOutcome(
            matches=tuple(accepted),
            examined=examined,
            scored=flat.size,
            complete=len(accepted) >= k,
        )

    def _satisfies(self, payload: dict, predicate: MetadataFilter) -> bool:
        """Evaluate a conjunction of field conditions against one payload.

        A field the payload does not carry cannot satisfy its condition, which
        matches how the inverted index behaves: entries exist only for values
        that were actually present.
        """
        for field, condition in predicate.items():
            if field not in payload:
                return False
            if not evaluate_condition(condition, render_value(payload[field]).decode("utf-8")):
                return False
        return True

    def _attach(self, matches: list[Match]) -> list[Match]:
        return [
            Match(match.id, match.score, match.distance, self._storage.get_payload(match.id) or {})
            for match in matches
        ]
