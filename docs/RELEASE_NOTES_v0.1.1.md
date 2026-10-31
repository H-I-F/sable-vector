# sable-vector v0.1.1

**Embedded hybrid vector search for local AI agent runtimes — now speakable over
Model Context Protocol.** 31 October 2026.

v0.1.0 made the memory durable: a vector, its payload and its inverted entries
commit as one unit, survive a `SIGKILL`, and reopen without a rebuild. What it
did not do is make that memory *addressable by anything that isn't Python*. If
your agent runtime is not your process — if it is a host that speaks MCP to its
tools — then every sable-vector collection was still behind a glue script you
had to write and maintain.

v0.1.1 removes that glue. A collection is now a first-class MCP resource: over
stdio it publishes three tools, and any MCP-capable host can read and write it
without importing this library at all.

## What ships in v0.1.1

### Model Context Protocol server

```bash
pip install "sable-vector[sable,mcp]"
sable-vector serve-mcp
```

The command blocks while the transport is open and returns when the client
disconnects — which is why it belongs in a client's stdio server configuration
rather than a shell you are waiting on.

| Tool | Purpose |
| --- | --- |
| `sable_vector_search` | Query by embedding, optionally narrowed to records whose payload fields equal given values |
| `sable_vector_upsert` | Write one record — embedding plus JSON attributes — as one durable unit |
| `sable_vector_describe` | Report record count, vector width and distance metric for a collection |

Two details worth stating plainly, because they are the ones people get wrong:

**Embeddings come from the caller.** The server never computes a vector, never
calls a model, and never opens a socket. It is an index with a protocol in front
of it; whoever owns the embedding model still owns the embedding.

**Filters are real filters.** A `search` carrying a `filters` payload is routed
through `PrefilterExecutor`, so candidates are pruned by the inverted index
*before* scoring rather than retrieved globally and discarded afterwards. You can
observe the difference in the response: the pre-filter path reports how many
candidates it considered and how many it actually scored. A predicate that
matches nothing returns zero candidates, not a top-N of near-misses.

### A note on the `mcp` pin

The server uses the low-level `Server.list_tools` / `Server.call_tool`
**decorator** API. Version 2.x of the `mcp` package replaced those decorators
with constructor callbacks, so the extra is pinned to `<2`. This is a deliberate
compatibility choice, not an oversight: the pinned line is what the documented
code was written against, and unpinning silently would trade a working server
for an `AttributeError` on the first handshake.

### Documentation fixes carried in this release

- The quickstart in both READMEs and in the v0.1.0 notes called
  `FlatIndex.materialise(engine, metric=...)`. `materialise` takes the
  **storage**, not the engine, and requires `dim` — the snippet raised
  `TypeError` exactly as written. All three occurrences are corrected.
- Unquoted Mermaid node labels containing parentheses broke the architecture
  diagrams on GitHub; both READMEs now quote them.

## Upgrading from v0.1.0

Storage layout is unchanged. `0.1.0` was prefixed as pre-stable and no migration
step exists: reopening an existing collection with v0.1.1 just works. The only
behavioural difference is additive — a new `serve-mcp` command and an optional
`mcp` extra nothing else depends on.

```bash
pip install --upgrade "sable-vector[sable,mcp]"
```

## Known limits

Unchanged from v0.1.0, and still the honest list: a single writer process per
collection; the on-disk layout is pre-1.0 and not yet frozen; pre-filtering only
sees fields explicitly declared as indexed; no embedding model is bundled.

MCP-specific additions to that list: the server is single-client stdio (no HTTP
transport, no multi-tenant auth layer — this is a local library, and the trust
boundary is the process boundary of whoever launched it), and each tool call
opens and materialises state for its collection, so throughput-sensitive loops
should use the library API directly rather than going over the protocol.

## Thanks

This release also records the cross-layer bug hunt from v0.1.0's development —
a concurrency failure that turned out to be a storage-engine defect, fixed
upstream in [`sable`](https://github.com/H-I-F/sable) v0.1.1. The full write-up
is in [the case study](./case-studies/cross-layer-bugfix.md); both READMEs carry
*"Built on sable — with critical upstream bugfixes contributed"*.

## Links

- Repository: <https://github.com/H-I-F/sable-vector>
- Release: <https://github.com/H-I-F/sable-vector/releases/tag/v0.1.1>
- Full history: [CHANGELOG.md](../CHANGELOG.md)
- Upstream storage engine: [`sable`](https://github.com/H-I-F/sable)
