# Cross-layer bug hunt: when a snapshot read silently lost a vector

A debugging story about [`sable`](https://github.com/H-I-F/sable), this project's
storage engine, and how two failing tests in `sable-vector` ended up being a
one-paragraph fix in someone else's memtable.

## Summary

| | |
| --- | --- |
| **Symptom** | Two tests in `tests/test_concurrent.py` failed deterministically (100% of runs, not flaky) |
| **Root cause** | `sable`'s memtable point lookup returned only the *newest* version of a key, so a snapshot-pinned read could not fall back to an older, still-visible version |
| **Fix** | `MemTable.get_visible()` — walk the version run and stop at the first version at or below the view's sequence, matching what range scans already did |
| **Shipped as** | [`sable` v0.1.1](https://github.com/H-I-F/sable/releases/tag/v0.1.1), pinned from `sable-vector` |
| **Why it mattered** | It was not a test bug and could not be worked around downstream — a durable engine that sometimes fails to find a committed key is not durable |

## The symptom

Two of sable-vector's concurrency tests failed every single time:

- `test_readers_never_see_a_payload_without_its_vector` — a concurrent reader
  observed a committed record's payload while its vector had gone missing.
- `test_snapshot_pins_are_returned_on_release` — referenced an observability
  attribute that did not exist.

The second was cosmetic (the engine had to expose it). The first was real, and it
was the kind of failure worth stopping everything for: the whole premise of this
project is that a record's payload and its vector commit as one unit, so a reader
should either see nothing or see something complete. Seeing *half* a record means
the durability claim is false.

The tests were deterministic, which was the useful part — a flaky test would have
been tempting to quarantine. A 100% failure is a reproducible bug.

## Narrowing it down

The first question to settle is always "whose bug is it?" Two projects are in
play, and guessing wrong wastes days. So the failing behaviour was reproduced in
**pure sable**, with no sable-vector code involved:

```python
# put K=A  -> allocated sequence 1
# put K=B  -> allocated sequence 4  (same key, newer version)
# snapshot pinned at sequence 2
view.get(b"K")   # expected b"A"  (newest version <= 2)
                 # actual: None
```

That is the whole reproduction, and it is unambiguous: this is storage-engine
behaviour with no vector code anywhere nearby.

Two further observations pinned the blast radius:

1. **Only the memtable path was wrong.** `SSTableReader.get_entry` already
   returned the first version whose `sequence <= visible_sequence`. After a flush
   or compaction the same read resolved correctly. So the bug lived purely in the
   in-memory table.
2. **It only fired when versions coexisted.** A single-version key resolved fine.
   You need a newer version of the same key sitting beside an older one the view
   is still entitled to — precisely what a concurrent writer produces.

## Root cause

`sable`'s memtable orders entries by `(user_key, -sequence)`, so the newest
version of a key sits first in its version run. That makes "newest wins" a
structural property rather than a convention. But there are two different read
paths, and only one of them honoured visibility:

- **Range scans** (`scan` / `_versioned_scan`) walk *every* version and filter by
  `sequence <= visible_sequence`, then take the first survivor. Correct.
- **Point lookups** (`MemTable.get`) went straight to the head of the run and
  returned it if visible — and returned `None` otherwise, giving up instead of
  continuing down the run.

So with a snapshot pinned at sequence 2 and a newer version at sequence 4 in front
of the visible sequence-1 version, the point lookup saw the sequence-4 entry,
found it invisible, stopped, and reported the key as absent. A scan over the same
key would have happily returned `A`. **The two read paths disagreed about what a
pinned sequence number resolves to**, which is the actual definition of broken.

This is why the sable-vector symptom looked the way it did: the engine writes each
record's parts under separate sequence numbers, and a concurrent reader holding a
pinned view would land between them.

## The fix

Teach the memtable the same visibility rule the SSTable reader already had. In
[`sable`](https://github.com/H-I-F/sable), `MemTable.get_visible`:

```python
def get_visible(self, user_key, visible_sequence):
    with self._mutation:
        node = self._descend_to(user_key)
        while node is not None:
            candidate = node.entry
            if candidate.user_key != user_key:
                break
            if visible_sequence is None or candidate.sequence <= visible_sequence:
                return candidate
            node = node.next[0]
    return None
```

Walk the run newest-first, return the first entry at or below the view's sequence.
`visible_sequence is None` keeps meaning "latest wins", so ordinary un-snapshotted
reads are unchanged.

Then point the lookup at it — `Engine._lookup_entry` now calls
`table.get_visible(user_key, visible_sequence)` for the memtable instead of
`table.get(user_key)`.

And add the observability the other test wanted, `Engine.open_snapshots`:

```python
@property
def open_snapshots(self) -> int:
    """Number of unreleased views, so callers can assert acquire/release balance."""
```

That one is small but not cosmetic: a snapshot is a resource that pins files and
retains versions. Being able to assert that a acquire/release pair nets to zero,
without reaching into engine internals, is what stops a leak from becoming a
mystery six weeks later.

**Verification.** Both originally-failing sable-vector tests turned green, the
pure-sable reproduction passes, `sable`'s own suite stayed at 183 tests passing
with all five CI checks green, and `sable-vector` stayed at 374 passing / 1
xpassed — the `xpassed` entry being the concurrency regression that used to be
expected to fail.

## Why it could not be fixed downstream

Worth stating explicitly, because "just handle it in your code" is the tempting
answer. `sable` has no atomic multi-key batch: **every `put` gets its own sequence
number**. sable-vector stages a vector, then its inverted index entries, then
commits the payload, relying on the storage engine's visibility rule to make that
look atomic to a reader. There is no sequence-number arithmetic the caller could
do to sidestep the memtable's behaviour. The only honest options were to fix the
engine or to document that concurrent snapshot reads may lose committed keys and
ship anyway.

## Shipping it across two repositories

This is the part that is easy to get wrong, because the two projects can drift:

1. Land the fix in [`sable`](https://github.com/H-I-F/sable) (PR #37), with two
   regression tests written at engine level so they guard the invariant even if
   sable-vector disappears.
2. Bump `sable` to **v0.1.1** and tag it — behaviour changes need a version, and
   the CHANGELOG entry names the downstream project that surfaced it.
3. In `sable-vector`, move the dependency pin from `@main` to the released tag:

   ```toml
   sable = ["sable @ git+https://github.com/H-I-F/sable.git@v0.1.1"]
   ```

   Building against `@main` would have meant "whichever commit happened to exist",
   which is how you end up unable to reproduce a release.
4. Add a `## Ecosystem` section to `sable`'s READMEs pointing back here, so the
   relationship is discoverable from either side.

The full write-up of the moment is also linked from the v0.1.0 announcement in
[`docs/RELEASE_NOTES_v0.1.0.md`](../RELEASE_NOTES_v0.1.0.md).

## Lessons

- **Two read paths that implement the same predicate will eventually disagree.**
  Here they were `get` and `scan`; the invariant both were supposed to honour is
  "the highest sequence at or below the view's sequence". If only one implements
  it, you have built two answers to one question.
- **Structure is not semantics.** Ordering by `(user_key, -sequence)` makes the
  newest version *first*, which is not the same as making it *visible*. The
  shortcut was fine right up until someone asked for a version that was neither
  newest nor absent.
- **Reproduce in the dependency, not in your project.** Three lines of pure sable
  settled the ownership question. Without that, the natural assumption would have
  been a bug in the vector engine's staged writes, and the investigation would
  have started in the wrong repository.
- **A 100%-failing test is a gift.** It is a deterministic reproduction someone
  already wrote for you. Fixing it removes a permanent tax; suppressing it does
  not.

## Links

- [`sable`](https://github.com/H-I-F/sable) — the storage engine this project is
  built on
- [`sable` v0.1.1 release](https://github.com/H-I-F/sable/releases/tag/v0.1.1)
- [`sable-vector`](https://github.com/H-I-F/sable-vector) — this project
