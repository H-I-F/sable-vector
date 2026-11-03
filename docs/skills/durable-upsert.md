[English](durable-upsert.md) | [简体中文](durable-upsert.zh-CN.md)

# Durable upsert: staging, commit records and crash rebuild

> The write path of this engine is three lines of ordering plus one visibility
> rule. Everything else — `reclaim()`, `recover()`, the chaos suite, the
> directional invariant the concurrency tests assert — follows from that rule.
> This guide reads `engine.py` and `recovery.py` in that order.

## 1. One invariant: a record is its payload

A record is committed exactly when `p:<id>` exists. [ADR-002](../decisions/adr-002-payload-commit-record.md)
makes this the single visibility predicate. It is worth stating what that buys
before looking at the code: a reader never has to reconcile a vector key, an
inverted entry and a payload to decide whether a row is real; recovery can treat
every other key as derived data that is a pure function of the committed
payloads.

The invariant has a **direction**. Vectors are written before payloads and
deleted after them, so there is no point in the sequence at which a payload
exists without its vector. A vector without a payload is the opposite: it is
exactly what a torn write looks like, and it is the direction the reaper exists
to clean up.

## 2. `_write_batch`: stage everything, commit once

```python
def _write_batch(self, items):
    if not items:
        return
    prepared = [(vid, self._admit(vec), payload) for vid, vec, payload in items]

    # 1. large, cheap, derived: the float32 images.
    for vector_id, vector, _ in prepared:
        self._storage.put_vector(vector_id, encode_vector(vector), sync=False)

    # 2. inverted entries, computed against the payload each one replaces.
    for vector_id, _, payload in prepared:
        previous = self._storage.get_payload(vector_id)
        self._apply_index_delta(vector_id, previous or {}, payload, sync=False)

    # 3. the commit records. Only the last one forces the log to stable storage.
    last = len(prepared) - 1
    for index, (vector_id, _, payload) in enumerate(prepared):
        self._storage.put_payload(vector_id, payload, sync=index == last)
```

`sync=False` leaves the frame in the WAL buffer. The final payload's `sync=True`
flushes and `fsync`s the whole run in one `fsync(2)`, regardless of how many keys
or records it spans. Both `upsert()` and `upsert_many()` go through this function;
a batch of 200 records is one durable unit, not 600 fsyncs.

The consequence is that durability boundaries are **per call**, not per record. A
caller that wants different durability for two records must issue two calls. It
also means nothing is visible mid-batch: frames are buffered, and no `p:` has
landed yet.

### The dimension is part of the batch too

```python
def _admit(self, vector):
    query = as_query(vector)
    width = int(query.shape[0])
    if self._dim is None:
        self._dim = width
        self._storage.put_raw(_DIM_KEY, str(width).encode("utf-8"), sync=False)
        return query
    if self._dim != width:
        raise ValueError(
            f"vector of width {width} does not match the collection width {self._dim}"
        )
    return query
```

The first vector in a collection fixes its width, and `_DIM_KEY` is written
unsynced as part of that same batch. If the batch never commits, the width was
never established either — the in-memory `_dim` and the on-disk metadata cannot
disagree. Every later vector must match it.

## 3. `delete()` is the mirror image

```python
def delete(self, vector_id):
    payload = self._storage.get_payload(vector_id)
    if payload is None and self._storage.get_vector_blob(vector_id) is None:
        return False
    if payload is not None:
        self._storage.delete_payload(vector_id, sync=False)
        self._drop_index_keys(vector_id, payload, sync=False)
    else:
        self._drop_unknown_index_keys(vector_id, sync=False)
    self._storage.delete_vector(vector_id, sync=True)
    return True
```

Payload first, inverted entries next, vector last with `sync=True`. The ordering
is chosen so an interrupted delete leaves a **vector with no payload** — a known
torn state that `reclaim()` recognises — rather than a payload with no vector,
which the write contract says cannot happen. When the payload is already gone,
`_drop_unknown_index_keys` walks the field instead of trying to reconstruct the
value segment from a missing payload.

`_apply_index_delta` → `ScalarIndex.reindex()` diffs old and new payloads, so an
update to one attribute drops exactly the old `idx:` entries and writes exactly
the new ones, all unsynced.

## 4. `reclaim()`: bring the tree back to "a record is its payload"

```python
def reclaim(self) -> RepairReport:
    vector_ids = set(self._storage.iter_vector_ids())
    payloads = dict(self._storage.iter_payloads())
    payload_ids = set(payloads)

    report = RepairReport(
        orphan_vectors=len(vector_ids - payload_ids),
        orphan_payloads=len(payload_ids - vector_ids),
    )
    for orphan in sorted(vector_ids - payload_ids):
        self._storage.delete_vector(orphan)
        report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))
    for orphan in sorted(payload_ids - vector_ids):
        self._storage.delete_payload(orphan)
        report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))

    audit = self._scalar.repair()
    return _bump(report,
                 stale_index_keys=len(audit.orphaned),
                 rebuilt_index_keys=len(audit.missing))
```

`RepairReport` has four counters and a `total`:

| Counter | Meaning |
| --- | --- |
| `orphan_vectors` | torn writes — `v:` with no `p:` |
| `orphan_payloads` | `p:` with no `v:`; impossible under the write order, reported anyway |
| `stale_index_keys` | `idx:` entries pointing at ids no longer committed |
| `rebuilt_index_keys` | entries a committed record was missing |

The inverted index is a pure function of the committed payloads, so once the torn
records are gone, `ScalarIndex.repair()` diffs it in **both** directions: orphaned
entries are dropped, missing ones rebuilt. There is no "trust whatever is on disk"
branch. `VectorEngine.open()` calls `reclaim()` before returning the handle, so a
caller never observes a torn record.

## 5. `recover()`: reclaim, then decide whether the IVF index is worth keeping

`recover()` starts with `repair = engine.reclaim()` and carries its counters into
a `RecoveryReport`. Then it measures the IVF index without writing anything:

```python
health = inspect_ivf(storage, metric=engine.metric, dim=dim)
if health.needs_retrain(rebuild_ratio):
    if health.vectors == 0:
        clear_namespace(storage, POSTING_PREFIX)
        clear_namespace(storage, CENTROID_PREFIX)
        storage.delete_raw(IVF_META_KEY, sync=False)
        return _bump(report, retrained=True)
    rebuild_ivf(storage, metric=engine.metric, dim=dim,
                partitions=min(partitions, health.vectors), seed=seed)
    ...
```

`IndexHealth` reports what is wrong without modifying anything:

| Field / property | Meaning |
| --- | --- |
| `trained`, `partitions`, `vectors`, `postings` | what the index and corpus contain |
| `missing` | committed ids with no posting anywhere |
| `stale` | posting ids that are not committed — survivors of an interrupted delete or rebuild |
| `duplicated` | ids in more than one partition; always a rebuild trigger |
| `malformed` | a stored image is not `dim` floats wide, or metadata will not decode |
| `drift` | `missing + stale + duplicated` |
| `healthy` | no damage, no drift; an empty corpus is healthy only with *no* index |
| `needs_retrain(ratio)` | drift is big enough to recompute rather than splice |

`needs_retrain` is deliberately strict: `malformed`, missing metadata, a `dim` or
`k` mismatch, any duplication, or drift above `rebuild_ratio * vectors` all force
a full retrain. Otherwise `recover()` loads the index and splices just the
difference — `remove()` for each `stale` id, `add()` for each `missing` one.

Two details make this safe to call on every open. A healthy store yields an
**all-zero** `RecoveryReport`, and `train=False` leaves an untrained or badly
drifted index alone so that a read-only inspection never triggers a rebuild.

## 6. How the tests pin the invariant

A test that runs threads until something breaks proves nothing about which
intermediate states are legal. `tests/test_concurrent.py` states the invariant
first, then forces the interleaving:

- **Barriers, not sleeps.** `threading.Barrier` releases every participant at the
  same instant, so "the writer started before the reader looked" is a property of
  the test rather than a hope about the scheduler.
- **Bounded work.** Readers run a fixed number of passes; writers a fixed number
  of mutations; a reader needing the post-write state waits on an `Event`, which
  is synchronisation rather than a delay.
- **A thread that raises fails the test.** Python reports an exception in a
  thread as a warning and lets the test pass for the wrong reason, so every
  participant runs under a guard that records the failure and breaks the barrier.

`tests/test_chaos.py` kills the process in five structurally different places —
mid-ingest, mid-batch (inside one `upsert_many` sharing a single fsync boundary),
mid-training, mid-delete, and mid-recovery — and then kills it **five times in a
row** on the same database, because a store that recovers from the first crash
and not the fourth is not a store. Every kill is a real `SIGKILL`: no handler, no
`finally`, no runtime flush. What is asserted is narrow and strong: the store
reopens, nothing acknowledged before the kill is lost, every surviving vector has
a payload and every payload has a vector, the inverted index agrees with the
payloads, and the derived index converges. Nothing asserts a particular record
count — how many survive is up to the storage engine; that all of them are whole
is not.

[ADR-002](../decisions/adr-002-payload-commit-record.md) notes why this matters:
anything that writes a payload without `sync=True`, or makes another key
authoritative for visibility, silently breaks the invariant. A return value would
not catch that; these tests do.

## 7. Failure modes

| Mistake | What actually happens |
| --- | --- |
| `sync=True` on every staged key | Correct but 3× the fsyncs; the batch stops being one durable unit. |
| `sync=False` on the final payload | Nothing is durable when the call returns; a crash loses acknowledged writes. |
| Make `v:` authoritative for visibility | Torn writes become visible rows; `reclaim()` can no longer tell them apart. |
| Delete vector before payload | An interrupted delete leaves `p:` without `v:`, the one state the contract forbids. |
| Skip `reclaim()` on open | A caller can observe a torn record; `RepairReport` never gets a chance to converge. |
| Assume a healthy report means a trained index | An empty corpus is healthy precisely when there is no index. |

## 8. Checklist for a new durable write path

1. Decide what single key means "committed", and make everyone use it.
2. Stage derived keys unsynced; flush exactly once, on the commit record.
3. Put metadata established by the first write in the same batch, unsynced.
4. Mirror the order on delete so a torn delete is a known orphan, not a new state.
5. Rebuild derived indexes as a pure function of the committed set, diffing both ways.
6. Report impossible states rather than assuming them away.
7. Test with barriers and real `SIGKILL`, asserting wholeness rather than counts.

## Where to read next

| Concern | Source |
| --- | --- |
| The write and delete paths | [`engine.py`](../../src/sable_vector/engine.py) |
| Commit-boundary decision | [ADR-002](../decisions/adr-002-payload-commit-record.md) |
| Rebuild and index health | [`recovery.py`](../../src/sable_vector/recovery.py) |
| Inverted-index two-way diff | [`index/scalar.py`](../../src/sable_vector/index/scalar.py) |
| Directional invariant under threads | [`tests/test_concurrent.py`](../../tests/test_concurrent.py) |
| Five kill points, five crashes in a row | [`tests/test_chaos.py`](../../tests/test_chaos.py) |
| Upstream WAL and torn-tail recovery | [`sable`](https://github.com/H-I-F/sable) `docs/skills/wal-and-recovery.md` |
