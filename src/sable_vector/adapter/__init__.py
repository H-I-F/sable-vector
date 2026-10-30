"""Protocol adapters that make this engine usable from an agent framework.

An adapter exists because the framework's interface and this engine's interface
disagree about who owns the hard part. A generic ``VectorStore`` treats metadata
filtering as a post-processing step: embed, search, then drop whatever does not
match. This engine treats it as a *prefix scan*, and the whole point of the
scalar inverted index is that a predicate is answered before any vector is
scored.

So the adapter is not a wrapper that renames methods. It is the place where a
framework-shaped request is translated into an engine-shaped one, and where the
one thing that cannot be translated is handled honestly rather than silently —
a filter on a field that was never indexed becomes a post-filter, not an empty
result and not an exception.
"""

from __future__ import annotations

__all__: list[str] = []
