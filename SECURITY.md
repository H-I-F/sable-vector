# Security Policy

## Supported versions

sable-vector is pre-1.0. Only the latest `0.1.x` line receives security fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately to the maintainer:

- Email: security reports to the project owner via GitHub
  (use the "Report a vulnerability" option on the
  [Security tab](https://github.com/MKA930/sable-vector/security/advisories/new))

Acknowledge receipt is normally within 72 hours. Include a minimal reproduction
where possible: the affected version, the embedding/scalar configuration, and the
observed vs. expected behaviour.

## Scope

sable-vector is an **embedded** engine: it runs inside the caller's process and
trusts its host. The following are out of scope unless explicitly stated:

- Authentication / authorization of clients (it has none — it is a library).
- Network transport security (the host is responsible for any RPC it builds on
  top).
- Supply-chain of the build host.

In scope:

- Correctness and integrity of stored vectors, payloads and scalar metadata.
- Snapshot isolation and crash/recovery guarantees of persisted data.
- The storage engine it depends on, [`sable`](https://github.com/MKA930/sable);
  if a vulnerability is rooted there, it should be reported against `sable` and
  this project will bump the pinned tag once a fix ships.

## Dependency handling

`sable` is pinned to a released tag in `pyproject.toml` (currently `v0.1.1`).
Security updates to the storage engine are consumed by bumping that pin and
releasing a new `0.1.x` of sable-vector.
