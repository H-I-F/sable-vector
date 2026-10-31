"""Physical namespace adapter over one ``sable`` database.

A single LSM tree holds every concern of the vector engine, separated only by
key prefix. That is a deliberate choice: co-locating them inside one tree means
one write-ahead log, one manifest, one set of SSTables and one crash-recovery
story, instead of four subsystems that can each recover to a different point in
time. The price is that the key space has to be designed rather than inherited:

    v:<id>                      raw float32 image
    p:<id>                      canonical JSON payload (the commit record)
    idx:<field>:<value>:<id>    scalar inverted index entry, empty value
    c:<cluster>                 IVF centroid
    b:<cluster>:<id>            IVF posting-list member
    m:<name>                    collection metadata

Three properties make the layout work.

* **Prefixes are designed for scans.** A predicate is a range, not a filter:
  ``idx:tenant:acme:`` is a contiguous run of keys, so answering it is one
  ordered read rather than a walk over every record with a test per record.
* **Segments are escaped, not assumed clean.** ``:`` separates segments and is
  legal inside a field name, a value or an id; unescaped separators would make
  ``idx:a:b:c`` ambiguous. Escaping happens once, here, so no caller has to
  remember to do it.
* **The prefixes are disjoint by construction.** ``v:`` is not a prefix of
  ``p:``, so a namespace scan can never leak another namespace's keys — which is
  what keeps a full rebuild from having to filter by key shape after the fact.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from sable.engine import Engine, Snapshot
from sable.types import EngineOptions, EngineStats, WriteOptions

from sable_vector.codec import decode_payload, encode_payload
from sable_vector.types import VectorId

__all__ = [
    "CENTROID_PREFIX",
    "INDEX_PREFIX",
    "META_PREFIX",
    "PAYLOAD_PREFIX",
    "POSTING_PREFIX",
    "VECTOR_PREFIX",
    "VectorStorage",
    "decode_id",
    "escape_segment",
    "index_key_rendered",
    "index_key",
    "index_prefix",
    "payload_key",
    "prefix_upper_bound",
    "render_value",
    "vector_key",
]

VECTOR_PREFIX = b"v:"
PAYLOAD_PREFIX = b"p:"
INDEX_PREFIX = b"idx:"
CENTROID_PREFIX = b"c:"
POSTING_PREFIX = b"b:"
META_PREFIX = b"m:"

_SEPARATOR = b":"
_ESCAPE = b"\\"

_NAMESPACES = (
    VECTOR_PREFIX,
    PAYLOAD_PREFIX,
    INDEX_PREFIX,
    CENTROID_PREFIX,
    POSTING_PREFIX,
    META_PREFIX,
)


def escape_segment(segment: bytes) -> bytes:
    """Escape ``\\`` then ``:`` so a segment can never forge a separator."""
    return segment.replace(_ESCAPE, _ESCAPE + _ESCAPE).replace(
        _SEPARATOR, _ESCAPE + _SEPARATOR
    )


def _split_escaped(key: bytes, *, expected: int) -> list[bytes]:
    """Split ``key`` on unescaped separators.

    Args:
        key: Segment run, without its namespace prefix.
        expected: How many segments the caller requires.

    Returns:
        The unescaped segments.

    Raises:
        ValueError: The key does not hold ``expected`` segments.
    """
    parts: list[bytearray] = [bytearray()]
    index = 0
    while index < len(key):
        current = key[index : index + 1]
        if current == _ESCAPE and index + 1 < len(key):
            parts[-1].append(key[index + 1])
            index += 2
        elif current == _SEPARATOR:
            parts.append(bytearray())
            index += 1
        else:
            parts[-1].append(key[index])
            index += 1
    if len(parts) != expected:
        raise ValueError(f"malformed key {bytes(key)!r}: expected {expected} segments")
    return [bytes(part) for part in parts]


def prefix_upper_bound(prefix: bytes) -> bytes | None:
    """Smallest key that is still greater than every key under ``prefix``.

    Args:
        prefix: Exclusive lower bound of a scan.

    Returns:
        The incremented prefix, or ``None`` when every byte is ``0xff`` and no
        upper bound exists — in which case the scan runs to the end of the tree.
    """
    data = bytearray(prefix)
    while data and data[-1] == 0xFF:
        del data[-1]
    if not data:
        return None
    data[-1] += 1
    return bytes(data)


def vector_key(vector_id: VectorId) -> bytes:
    return VECTOR_PREFIX + escape_segment(vector_id.encode("utf-8"))


def payload_key(vector_id: VectorId) -> bytes:
    return PAYLOAD_PREFIX + escape_segment(vector_id.encode("utf-8"))


def index_key(field: str, value: Any, vector_id: VectorId) -> bytes:
    """Build ``idx:<field>:<value>:<id>``.

    Args:
        field: Payload attribute name.
        value: Attribute value; rendered as text so the index stays sortable.
        vector_id: Referenced record.
    """
    return index_key_rendered(field, render_value(value).decode("utf-8"), vector_id)


def index_key_rendered(field: str, rendered: str, vector_id: VectorId) -> bytes:
    """Build ``idx:<field>:<value>:<id>`` from an already-rendered value.

    An entry read back out of the key space carries its value in rendered form.
    Handing that text back to :func:`index_key` would render it a second time
    and miss the entry it came from; this is the builder for that direction,
    which is what lets an audit delete exactly the key it just read.

    Args:
        field: Payload attribute name.
        rendered: Value exactly as it is stored.
        vector_id: Referenced record.
    """
    return INDEX_PREFIX + _SEPARATOR.join(
        (
            escape_segment(field.encode("utf-8")),
            escape_segment(rendered.encode("utf-8")),
            escape_segment(vector_id.encode("utf-8")),
        )
    )


def index_prefix(field: str, value: Any | None = None) -> bytes:
    """Prefix covering every id indexed under ``field`` (and optionally ``value``).

    Args:
        field: Payload attribute name.
        value: When given, narrow the prefix to one indexed value.
    """
    head = INDEX_PREFIX + escape_segment(field.encode("utf-8")) + _SEPARATOR
    if value is None:
        return head
    return head + escape_segment(render_value(value)) + _SEPARATOR


def render_value(value: Any) -> bytes:
    """Render an indexable value as sortable text.

    Booleans render as ``true``/``false`` and numbers as fixed-width text so
    that lexical order matches numeric order over any range the index supports.
    """
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, (int, float)):
        return f"{value:020.6f}".encode()
    return str(value).encode("utf-8")


class VectorStorage:
    """Prefix-aware facade over one :class:`sable.engine.Engine`.

    Everything above this class talks in ids, fields and payloads; nothing above
    it sees a key. Keeping the byte layout in one type is what lets the layout
    change — a new namespace, a different escape scheme — without touching the
    query path.
    """

    def __init__(self, engine: Engine, home: Path) -> None:
        self._engine = engine
        self._home = Path(home)

    @classmethod
    def open(
        cls, home: Path, options: EngineOptions | None = None
    ) -> VectorStorage:
        """Open or recover the database directory ``home``.

        Args:
            home: Directory holding the WAL, manifest and SSTables.
            options: Storage-engine tuning; defaults are used when omitted.
        """
        engine = Engine.open(Path(home), options)
        return cls(engine, Path(home))

    @property
    def home(self) -> Path:
        return self._home

    @property
    def engine(self) -> Engine:
        """The underlying LSM engine, for callers that need a raw snapshot."""
        return self._engine

    @property
    def stats(self) -> EngineStats:
        return self._engine.stats

    def close(self) -> None:
        self._engine.close()

    def flush(self) -> None:
        """Publish the memtable so the data is readable without a WAL replay."""
        self._engine.flush()

    def compact(self) -> bool:
        return self._engine.compact()

    # -- raw key space ----------------------------------------------------

    def put_raw(self, key: bytes, value: bytes, *, sync: bool = True) -> None:
        """Write one key. ``sync=False`` defers the fsync to a later commit."""
        self._engine.put(key, value, WriteOptions(sync=sync))

    def get_raw(self, key: bytes) -> bytes | None:
        return self._engine.get(key)

    def delete_raw(self, key: bytes, *, sync: bool = True) -> None:
        self._engine.delete(key, WriteOptions(sync=sync))

    def scan_prefix(
        self, prefix: bytes, *, limit: int | None = None
    ) -> Iterator[tuple[bytes, bytes]]:
        """Every ``(key, value)`` under ``prefix``, in key order.

        Args:
            prefix: Namespace prefix, already escaped by a key builder.
            limit: Optional cap on emitted pairs.
        """
        yield from self._engine.scan(
            start=prefix, end=prefix_upper_bound(prefix), limit=limit
        )

    def snapshot(self) -> Snapshot:
        """Cut a frozen view of the tree that outlives the writes that follow it.

        The view is frozen by pinning, not by copying, so a read holds no engine
        lock and a compaction can run to completion underneath it. That is what
        makes the concurrency contract of this engine a property rather than a
        hope: a writer stages a record's derived keys and then commits its
        payload, and a reader without a snapshot can walk straight into the gap
        between the two. A reader *with* one cannot, because the gap is not
        visible at any single sequence number.

        The view is a resource: use it as a context manager or release it
        explicitly. Never releasing one defers file reclamation, which leaks
        space rather than correctness.
        """
        return self._engine.snapshot()

    def scan_namespace(
        self, prefix: bytes, *, limit: int | None = None
    ) -> Iterator[tuple[VectorId, bytes]]:
        """Scan a ``v:``/``p:`` namespace and decode the id out of each key."""
        for key, value in self.scan_prefix(prefix, limit=limit):
            yield key[len(prefix) :].decode("utf-8"), value

    # -- vectors -----------------------------------------------------------

    def put_vector(self, vector_id: VectorId, blob: bytes, *, sync: bool = True) -> None:
        self.put_raw(vector_key(vector_id), blob, sync=sync)

    def get_vector_blob(self, vector_id: VectorId) -> bytes | None:
        return self.get_raw(vector_key(vector_id))

    def delete_vector(self, vector_id: VectorId, *, sync: bool = True) -> None:
        self.delete_raw(vector_key(vector_id), sync=sync)

    def iter_vector_ids(self) -> Iterator[VectorId]:
        for raw_id, _ in self.scan_namespace(VECTOR_PREFIX):
            yield decode_id(raw_id)

    # -- payloads ----------------------------------------------------------

    def put_payload(
        self, vector_id: VectorId, payload: Mapping[str, Any], *, sync: bool = True
    ) -> None:
        self.put_raw(payload_key(vector_id), encode_payload(payload), sync=sync)

    def get_payload(self, vector_id: VectorId) -> dict[str, Any] | None:
        blob = self.get_raw(payload_key(vector_id))
        return None if blob is None else decode_payload(blob)

    def delete_payload(self, vector_id: VectorId, *, sync: bool = True) -> None:
        self.delete_raw(payload_key(vector_id), sync=sync)

    def iter_payloads(self) -> Iterator[tuple[VectorId, dict[str, Any]]]:
        for raw_id, blob in self.scan_namespace(PAYLOAD_PREFIX):
            yield decode_id(raw_id), decode_payload(blob)

    # -- scalar inverted index ---------------------------------------------

    def put_index_key(
        self, field: str, value: Any, vector_id: VectorId, *, sync: bool = True
    ) -> None:
        """Add one inverted entry. Writing it twice is idempotent."""
        self.put_raw(index_key(field, value, vector_id), b"", sync=sync)

    def delete_index_key(
        self, field: str, value: Any, vector_id: VectorId, *, sync: bool = True
    ) -> None:
        self.delete_raw(index_key(field, value, vector_id), sync=sync)

    def delete_index_entry(
        self, field: str, rendered: str, vector_id: VectorId, *, sync: bool = True
    ) -> None:
        """Remove an entry identified by its rendered value, as stored."""
        self.delete_raw(index_key_rendered(field, rendered, vector_id), sync=sync)

    def scan_index(self, field: str, value: Any) -> Iterator[VectorId]:
        """Every id whose ``field`` is indexed with ``value``."""
        for raw_id, _ in self.scan_namespace(index_prefix(field, value)):
            yield decode_id(raw_id)

    def scan_index_field(self, field: str) -> Iterator[tuple[str, VectorId]]:
        """Every ``(value, id)`` pair indexed under ``field``, in value order.

        Values are returned unescaped but still in their rendered (sortable)
        text form, which is what makes a range predicate a contiguous read.
        """
        prefix = index_prefix(field)
        for key, _ in self.scan_prefix(prefix):
            segments = _split_escaped(key[len(INDEX_PREFIX) :], expected=3)
            yield segments[1].decode("utf-8"), segments[2].decode("utf-8")


def decode_id(segment: str) -> VectorId:
    """Undo :func:`_escape` for a single id segment.

    Args:
        segment: A key segment as it appears on the wire, still escaped.

    Returns:
        The original id.
    """
    if "\\" not in segment:
        return segment
    out: list[str] = []
    index = 0
    while index < len(segment):
        if segment[index] == "\\" and index + 1 < len(segment):
            out.append(segment[index + 1])
            index += 2
        else:
            out.append(segment[index])
            index += 1
    return "".join(out)


def namespace_prefixes() -> tuple[bytes, ...]:
    """Every namespace prefix, for tooling that has to enumerate them."""
    return _NAMESPACES
