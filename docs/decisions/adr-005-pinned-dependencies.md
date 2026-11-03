[English](adr-005-pinned-dependencies.md) | [简体中文](adr-005-pinned-dependencies.zh-CN.md)

# ADR-005: Pin the dependencies that define the API

**Status:** Accepted — implemented in `pyproject.toml` and `.github/dependabot.yml`.

## Context

Two dependencies are part of the public contract, not just the environment:

- `adapter/mcp_server.py` is built on the low-level `Server.list_tools` /
  `Server.call_tool` **decorator** API. `mcp` 2.x replaced those decorators with
  constructor callbacks. A Dependabot PR that relaxed the constraint to `<3` turned
  all three CI jobs red with `AttributeError: 'Server' object has no attribute
  'list_tools'`; the PR was closed and the rule made permanent.
- `sable` is not published to PyPI, and the snapshot-read consistency fix this
  project depends on only exists from `v0.1.1` (see the
  [case study](../case-studies/cross-layer-bugfix.md)). Building against `@main`
  would mean "whichever commit happened to exist", which makes a release
  irreproducible.

## Decision

```toml
mcp   = ["mcp>=1.9,<2"]
sable = ["sable @ git+https://github.com/H-I-F/sable.git@v0.1.1"]
```

`.github/dependabot.yml` carries a permanent `ignore` rule for `mcp` versions `>=2`.
The PEP 508 direct reference is not managed by Dependabot in the first place, so the
upstream pin is upgraded deliberately, by hand, with a release — never by a bot.

## Consequences

Unpinning `mcp` silently would trade a working stdio server for an exception on the
first handshake; it now requires migrating the server to the constructor-callback API
and shipping that as its own change. Upstream fixes arrive as a tag bump plus a new
`0.1.x` of sable-vector, which is also how `SECURITY.md` says they are consumed.

The price is age: both pins will drift behind their upstreams. That is accepted
because the alternative — an automatic PR that quietly changes the API the code was
written against — already failed once in CI. The PR review report checks the
`mcp<2` rule and the direct reference on every pull request so the constraint cannot
be edited away by accident.
