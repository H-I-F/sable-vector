[English](ivf-nprobe-tuning.md) | [简体中文](ivf-nprobe-tuning.zh-CN.md)

# Recipe: choosing `nprobe` against measured numbers

> The most useful question about IVF in this engine is not "which `nprobe`
> should I use" but "should I be using IVF at all". This recipe works from the
> [November 2026 benchmark report](../reports/benchmark-2026-11.md) — the numbers
> below are quoted from that run, not produced for this page — and turns them into
> a decision procedure. No benchmark was rerun here.

## Step 0: know what the index costs before it scores anything

`IvfIndex.search()` ranks the partitions with one gemm over the centroids, then
reads only the winning posting lists:

```python
distances = squared_distances(probe.reshape(1, self._dim), self._centroids)[0]
order = np.argsort(distances, kind="stable")[: min(nprobe, self.partitions)]
```

That centroid scoring and the posting-list reads happen **before a single vector
is scored**. On a small corpus the resident flat index has no such overhead — it
is one gemv over a contiguous `(n, d)` block. So approximate search is not a
free faster mode; it is a bet that skipping most of the corpus is worth paying
for the routing.

Training also has a precondition: `IvfIndex.train()` rejects an empty corpus and
a corpus smaller than `partitions`. `tests/test_ivf_scan.py` asserts both, plus
that training commits with one fsync.

## Step 1: the decision table

| Corpus | What the numbers say | Use |
| --- | --- | --- |
| ~1,000 vectors | flat p50 **0.144 ms**, recall 1.00, faster than IVF at any `nprobe` | resident `FlatIndex` |
| ~5,000 vectors, memory is tight | flat p50 3.773 ms; `flat-stream` 4.993 ms, recall 1.00, ~8× the resident path | `search_streaming()` |
| ~5,000 vectors, recall can be traded | IVF `nprobe=8` gives p50 5.860 ms, recall 0.920 | IVF, if recall 0.920 is acceptable |
| Large enough that skipping most rows dominates the routing cost | — | IVF, tuned on your own data |

The first row is the one people skip. At 1,000 vectors with 16 clusters,
`nprobe=8` is half the corpus — the curve flattens to recall 1.000 while p50
rises to 1.489 ms, six times the flat answer. Exact search is the right default
at this scale, and the report says so plainly.

`search_streaming()` exists for a different constraint, not a different speed:
it keeps only `block_rows × d` floats plus `n` scores resident, so it is the mode
for collections that do not fit in memory. It keeps recall 1.00 and is
disk-bound, which is why it costs roughly 8× the resident path at 1,000 vectors.

## Step 2: read the recall / latency curve, not one number

Both tables below are copied from the report (128 dimensions, 16 clusters, 30
queries, `k=10`, cosine, seed 42, Linux x86-64 / Python 3.11 / NumPy 2.3.5).

**1,000 vectors** — ingest 21,108 vec/s:

| index | nprobe | p50 (ms) | p95 (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | — | 0.144 | 0.157 | 6,894 | 1.00 |
| flat-stream | — | 1.148 | 1.194 | 871 | 1.00 |
| ivf | 1 | 0.257 | 0.316 | 4,022 | 0.523 |
| ivf | 2 | 0.466 | 0.553 | 2,166 | 0.680 |
| ivf | 4 | 0.819 | 1.202 | 1,129 | 0.877 |
| ivf | 8 | 1.489 | 1.609 | 663 | 1.000 |

**5,000 vectors** — ingest 26,596 vec/s:

| index | nprobe | p50 (ms) | p95 (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | — | 3.773 | 95.074 | 34 | 1.00 |
| flat-stream | — | 4.993 | 5.024 | 200 | 1.00 |
| ivf | 1 | 0.897 | 1.179 | 1,124 | 0.623 |
| ivf | 2 | 1.534 | 1.933 | 639 | 0.727 |
| ivf | 4 | 3.087 | 3.613 | 322 | 0.810 |
| ivf | 8 | 5.860 | 6.756 | 167 | 0.920 |

`nprobe` buys recall at a predictable price. On 5,000 vectors, 1 → 8 partitions
lifts recall 0.623 → 0.920 while p50 rises 0.897 → 5.860 ms and qps falls
1,124 → 167. Anchor the choice on the recall your application actually needs,
then read the latency off the same row; do not start from a latency budget and
hope the recall is fine.

`nprobe == partitions` degenerates to a full scan with exact recall. That is a
useful ceiling to know and a useless operating point to run — at that point you
have paid for centroid scoring on top of the whole corpus.

### One outlier is published, not smoothed away

The 5,000-vector resident flat row shows p50 3.773 ms but **p95 95.074 ms** and
mean 29.779 ms. That is a warm-up / allocation stall in a handful of the 30
queries, not a property of the kernel. The p50 is the representative figure; the
p95 is the reason not to quote a single number. A rerun will move those tails;
the recall column will not. Treat the recall curve and the ratios as the result,
and the absolute latencies as a snapshot of one container.

Ingest is not a durable write benchmark either: 21–27k vectors/s at 128
dimensions is an in-memory staging rate and does not include a per-record
`fsync`. [ADR-002](../decisions/adr-002-payload-commit-record.md) is where that
boundary actually sits.

## Step 3: reproduce on your own shape

The harness is self-asserting — the resident flat index is exhaustive, so recall
is 1.0 by construction and every approximate figure is computed against it with
`_recall_at_k`. The published run was:

```bash
python -m benchmarks.bench_vector \
  --dim 128 --sizes 1000 5000 --queries 30 --ks 10 \
  --nprobes 1 2 4 8 --clusters 16 --stream-reps 2 \
  --metric cosine --seed 42
```

For a production-shaped corpus, change the shape rather than trusting an
extrapolation:

```bash
python -m benchmarks.bench_vector --dim 1536 --sizes 10000 --queries 50
```

What the published run does **not** cover: 1536 dimensions, `l2` and `dot`
metrics, IVF training time, concurrent readers, and durable write costs with real
`fsync`. Recall here is measured against an exhaustive index over the *same*
corpus, so it says nothing about the quality of the upstream embedding model. If
your operating point depends on any of those, measure it.

## Step 4: keep the index healthy after a crash

An approximate index is derived data. `recover()` calls `inspect_ivf()` and asks
`IndexHealth.needs_retrain(rebuild_ratio)` whether to recompute the partitions or
just splice the difference. `drift` is `missing + stale + duplicated`; a
`malformed` image, missing metadata, a `dim`/`k` mismatch, any duplication, or
drift above the ratio all force a retrain. Otherwise the stale postings are
removed and the missing ones added in place. Calling `recover()` on a healthy
store yields an all-zero report, which is what makes it safe to run on every open.

This matters for tuning because a drifted index does not fail loudly — it
returns plausible neighbours with quietly degraded recall. Check
`IndexHealth.healthy` before treating a recall number as a property of `nprobe`.

## Checklist

1. Ask whether exact search already meets the latency budget. At 1,000 vectors
   it does, by a wide margin.
2. If the corpus does not fit in memory, use `search_streaming()` — it keeps
   recall 1.00.
3. Pick the recall you need, then read p50 / qps from that row; watch the p95
   tail.
4. Keep `nprobe` well below `partitions`; at the ceiling you are just paying for
   routing on top of a full scan.
5. Reproduce on your own dimension and metric before committing to a number.
6. After a crash, check index health before trusting a recall measurement.

## Where to read next

| Concern | Source |
| --- | --- |
| The measured report | [benchmark-2026-11.md](../reports/benchmark-2026-11.md) |
| The runnable harness | [`benchmarks/bench_vector.py`](../../benchmarks/bench_vector.py) |
| `nprobe` pruning and posting lists | [`index/ivf_storage.py`](../../src/sable_vector/index/ivf_storage.py) |
| Centroid training | [`index/ivf_cluster.py`](../../src/sable_vector/index/ivf_cluster.py) |
| Crash rebuild and index health | [`recovery.py`](../../src/sable_vector/recovery.py), [durable upsert](../skills/durable-upsert.md) |
| Score orientation behind the ground truth | [zero-copy codec](../skills/zero-copy-codec.md) |
