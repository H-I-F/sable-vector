"""Query executors: how a predicate and a similarity ranking are combined.

The two orderings answer the same question with different cost profiles, and
which one is right is a property of the *selectivity* of the predicate rather
than of taste. Both live here so the choice is explicit at the call site
instead of being buried in a search method.
"""

from __future__ import annotations

__all__ = ["prefilter"]
