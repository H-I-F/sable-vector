"""Query side of the scalar inverted index.

The write path maintains ``idx:<field>:<value>:<id>`` entries; this module is
what makes them worth maintaining. A predicate over a payload field becomes a
*range* over the key space, and the whole design follows from that one fact.

**Equality is a probe.** ``{"tenant": "acme"}`` is one prefix scan, cost
proportional to the answer rather than to the collection.

**Negation and ranges are a walk.** ``$ne``, ``$nin``, ``$gt`` … cannot be
expressed as a prefix, so they scan the field's entries and evaluate. That is
still cheaper than reading payloads: the value is carried in the key, so the
predicate is decided without touching a single record.

**Comparison happens on rendered values.** Values are stored as fixed-width,
sign-aware text precisely so that lexical comparison over the key equals numeric
comparison over the value — ``2 < 10`` and ``-1 < 2`` both hold as bytes. The
query value is rendered through the same function, so a predicate never has to
guess which type it is looking at.

**Conjunction is an intersection.** A top-level mapping is an AND; the smallest
set is resolved first and the rest are intersected into it, so a selective
predicate prunes before a broad one is even scanned.

Inconsistency is treated as data rather than as an exception. :meth:`audit`
compares the entries against the committed payloads and reports both directions,
and :meth:`repair` removes what should not exist and writes what should — which
is what lets a crash leave the index merely wrong rather than unrecoverable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sable_vector.storage import VectorStorage, render_value
from sable_vector.types import FILTER_OPERATORS, MetadataFilter, VectorId

__all__ = ["IndexAudit", "ScalarIndex", "evaluate_condition"]


@dataclass(frozen=True, slots=True)
class IndexAudit:
    """Difference between the inverted entries and the committed payloads.

    Attributes:
        missing: ``(field, value, id)`` entries a committed record should own.
        orphaned: ``(field, value, id)`` entries nothing committed owns.
    """

    missing: tuple[tuple[str, Any, VectorId], ...] = ()
    orphaned: tuple[tuple[str, Any, VectorId], ...] = ()

    @property
    def consistent(self) -> bool:
        return not self.missing and not self.orphaned

    @property
    def total(self) -> int:
        return len(self.missing) + len(self.orphaned)


def _is_expression(value: Any) -> bool:
    """True when a filter value is an operator expression rather than a scalar."""
    return isinstance(value, Mapping) and any(
        key.startswith("$") for key in value  # type: ignore[union-attr]
    )


def _probe_values(condition: Any) -> list[Any] | None:
    """Values answerable by a prefix scan, or ``None`` when a walk is needed.

    Args:
        condition: A scalar, or an operator expression.

    Returns:
        The values to probe, or ``None`` when the condition is not a positive
        membership test and the whole field has to be walked.
    """
    if not _is_expression(condition):
        return [condition]
    expression: Mapping[str, Any] = condition  # type: ignore[assignment]
    if set(expression) == {"$eq"}:
        return [expression["$eq"]]
    if set(expression) == {"$in"}:
        return list(expression["$in"])
    return None


def _render(value: Any) -> str:
    """Render a value the way the key space renders it."""
    return render_value(value).decode("utf-8")


def evaluate_condition(condition: Any, rendered_value: str) -> bool:
    """Evaluate one operator expression against an indexed value.

    Args:
        condition: A scalar (equality), or an expression over ``FILTER_OPERATORS``.
        rendered_value: The stored value in its rendered text form.

    Returns:
        Whether the value satisfies the condition.

    Raises:
        ValueError: The expression names an unsupported operator, or combines
            operators this index cannot evaluate together.
    """
    if not _is_expression(condition):
        return rendered_value == _render(condition)
    expression: Mapping[str, Any] = condition  # type: ignore[assignment]
    unknown = set(expression) - FILTER_OPERATORS
    if unknown:
        raise ValueError(f"unsupported filter operators: {sorted(unknown)}")

    for operator, operand in expression.items():
        if operator == "$eq" and rendered_value != _render(operand):
            return False
        if operator == "$ne" and rendered_value == _render(operand):
            return False
        if operator == "$in" and rendered_value not in {_render(v) for v in operand}:
            return False
        if operator == "$nin" and rendered_value in {_render(v) for v in operand}:
            return False
        if operator == "$gt" and not rendered_value > _render(operand):
            return False
        if operator == "$gte" and not rendered_value >= _render(operand):
            return False
        if operator == "$lt" and not rendered_value < _render(operand):
            return False
        if operator == "$lte" and not rendered_value <= _render(operand):
            return False
    return True


class ScalarIndex:
    """Inverted entries over the payload fields named at construction time.

    Args:
        storage: Namespace adapter holding the ``idx:`` namespace.
        fields: Payload attributes to maintain and query.
    """

    def __init__(self, storage: VectorStorage, fields: Sequence[str]) -> None:
        self._storage = storage
        self._fields = tuple(fields)

    @property
    def fields(self) -> tuple[str, ...]:
        return self._fields

    # -- maintenance -------------------------------------------------------

    def owned(self, vector_id: VectorId, payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """The ``(field, value)`` pairs a payload should have inverted."""
        return [
            (field, payload[field])
            for field in self._fields
            if payload.get(field) is not None
        ]

    def index(self, vector_id: VectorId, payload: Mapping[str, Any], *, sync: bool = True) -> int:
        """Write every entry a payload owns. Writing twice changes nothing."""
        written = 0
        for field, value in self.owned(vector_id, payload):
            self._storage.put_index_key(field, value, vector_id, sync=sync)
            written += 1
        return written

    def deindex(
        self,
        vector_id: VectorId,
        payload: Mapping[str, Any] | None = None,
        *,
        sync: bool = True,
    ) -> int:
        """Remove every entry a payload owns.

        Args:
            vector_id: Record to unlink.
            payload: Known payload; when omitted the field is walked, which is
                the only option once the payload is already gone.
            sync: Durability of the removals.
        """
        if payload is None:
            removed = 0
            for field in self._fields:
                for rendered, indexed_id in list(self._storage.scan_index_field(field)):
                    if indexed_id == vector_id:
                        self._storage.delete_index_entry(field, rendered, indexed_id, sync=sync)
                        removed += 1
            return removed
        removed = 0
        for field, value in self.owned(vector_id, payload):
            self._storage.delete_index_key(field, value, vector_id, sync=sync)
            removed += 1
        return removed

    def reindex(
        self,
        vector_id: VectorId,
        previous: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> int:
        """Move the entries from one payload to the next, touching only changes.

        Rewriting every entry would delete and re-add the same key for fields
        whose value did not change; since removal is ordered before insertion,
        that would erase them.
        """
        before = dict(previous or {})
        changed = 0
        for field in self._fields:
            old = before.get(field)
            new = payload.get(field)
            if old == new:
                continue
            if old is not None:
                self._storage.delete_index_key(field, old, vector_id, sync=sync)
                changed += 1
            if new is not None:
                self._storage.put_index_key(field, new, vector_id, sync=sync)
                changed += 1
        return changed

    # -- querying ----------------------------------------------------------

    def matching_ids(self, predicate: MetadataFilter | None) -> set[VectorId] | None:
        """Ids satisfying a predicate, or ``None`` when there is nothing to apply.

        Args:
            predicate: Conjunction of per-field conditions. Empty or ``None``
                means "everything", which callers treat as "do not restrict".

        Returns:
            The matching ids, or ``None`` when the predicate imposes no
            restriction at all — distinguishable from an empty match, which is
            an empty set.

        Raises:
            ValueError: A condition names an unsupported operator or a field
                that this index does not maintain.
        """
        if not predicate:
            return None

        for field in predicate:
            if field not in self._fields:
                raise ValueError(
                    f"field {field!r} is not indexed; indexed fields are {list(self._fields)}"
                )

        # Resolve the most selective field first so the intersection starts small.
        ordered = sorted(
            predicate.items(), key=lambda item: len(_probe_values(item[1]) or [])
        )
        result: set[VectorId] | None = None
        for field, condition in ordered:
            ids = self.field_ids(field, condition)
            result = ids if result is None else (result & ids)
            if not result:
                return set()
        return result

    def field_ids(self, field: str, condition: Any) -> set[VectorId]:
        """Ids of one field's entries satisfying one condition.

        Args:
            field: Indexed attribute.
            condition: A scalar, or an operator expression.

        Returns:
            The matching ids.

        Raises:
            ValueError: The expression names an operator this index cannot
                evaluate. Raised before any data is read, so a malformed query
                is rejected even against an empty field.
        """
        if _is_expression(condition):
            unknown = set(condition) - FILTER_OPERATORS
            if unknown:
                raise ValueError(f"unsupported filter operators: {sorted(unknown)}")
        probes = _probe_values(condition)
        if probes is not None:
            found: set[VectorId] = set()
            for value in probes:
                found.update(self._storage.scan_index(field, value))
            return found
        return {
            vector_id
            for value, vector_id in self._storage.scan_index_field(field)
            if evaluate_condition(condition, value)
        }

    def values_of(self, field: str) -> list[str]:
        """Every rendered value indexed under ``field``."""
        return [value for value, _ in self._storage.scan_index_field(field)]

    # -- consistency -------------------------------------------------------

    def audit(self) -> IndexAudit:
        """Compare the entries against the committed payloads, both directions.

        Entries are identified the way the key space identifies them — by their
        rendered value — while :attr:`IndexAudit.missing` keeps the *domain*
        value so a repair can rewrite it through the normal builder.

        Returns:
            An :class:`IndexAudit`; :attr:`IndexAudit.consistent` is the answer.
        """
        expected: dict[tuple[str, str, VectorId], Any] = {}
        for vector_id, payload in self._storage.iter_payloads():
            for field, value in self.owned(vector_id, payload):
                expected[(field, _render(value), vector_id)] = value

        actual = {
            (field, rendered, vector_id)
            for field in self._fields
            for rendered, vector_id in self._storage.scan_index_field(field)
        }

        missing: list[tuple[str, Any, VectorId]] = []
        for key in sorted(expected, key=repr):
            if key not in actual:
                field, _, vector_id = key
                missing.append((field, expected[key], vector_id))
        orphaned = tuple(sorted(actual - set(expected), key=repr))
        return IndexAudit(missing=tuple(missing), orphaned=orphaned)

    def repair(self) -> IndexAudit:
        """Bring the entries back in line with the payloads.

        Returns:
            The audit taken *before* the repair, so a caller can report what was
            wrong rather than only that it is now right.
        """
        report = self.audit()
        for field, rendered, vector_id in report.orphaned:
            self._storage.delete_index_entry(field, rendered, vector_id)
        for field, value, vector_id in report.missing:
            self._storage.put_index_key(field, value, vector_id)
        return report

        report = self.audit()
        for field, rendered, vector_id in report.orphaned:
            self._storage.delete_index_entry(field, rendered, vector_id)
        for field, value, vector_id in report.missing:
            self._storage.put_index_key(field, value, vector_id)
        return report
