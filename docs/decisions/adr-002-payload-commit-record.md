[English](adr-002-payload-commit-record.md) | [简体中文](adr-002-payload-commit-record.zh-CN.md)

# ADR-002: The payload is the commit record

**Status:** Accepted — implemented in `src/sable_vector/engine.py`.

## Context

`sable` has no atomic multi-key batch: **every `put` gets its own sequence number**.
A single sable-vector record is several keys (see ADR-001), and a reader must never
see half of one. So the engine has to manufacture a commit boundary out of the only
thing the storage engine does offer — a per-write `sync` flag that decides whether
the WAL frame is flushed and `fsync`-ed.

## Decision

**The payload is the commit record.** A record is committed exactly when `p:<id>`
exists. The write path is:

```
v:<id>       float32 image     (sync=False)
idx:...      inverted entries  (sync=False)
p:<id>       payload           (sync=True)   <- the record becomes visible here
```

`sync=False` leaves the frame in the WAL buffer; the payload's `sync=True` flushes
and `fsync`s the whole run in one `fsync(2)`, regardless of how many keys it spans.
`upsert_many()` does the same at batch scale: stage every key of every record, then
commit the payloads, with only the last one forcing the log to stable storage.

`delete()` is the mirror image: it removes the payload and inverted entries unsynced
and closes with `delete_vector(sync=True)`.

`reclaim()` uses the same rule to define a torn write — a vector without a payload,
or an inverted entry without a committed id, is orphaned derived data.

## Consequences

A kill during an upsert leaves either nothing or a complete record. Readers can treat
"payload present" as the single visibility predicate, and recovery in `recovery.py`
rebuilds derived indexes as a pure function of the committed payloads.

The boundary is per record (or per batch), not per key. A caller that wants different
durability for different records must issue separate calls. Anything that writes a
payload without `sync=True`, or that makes another key authoritative for visibility,
silently breaks the invariant — which is why `tests/test_upsert.py` and the SIGKILL
chaos suite assert it directly rather than checking a return value.
