"""Index structures that turn a scan into a candidate set.

Every index here answers the same question — which rows are worth scoring — and
they differ only in how much work they are willing to do to shrink that set.
``flat`` scores everything and therefore cannot be wrong; the IVF structures
trade a controllable amount of recall for reading a fraction of the bytes.
"""

from __future__ import annotations

__all__ = ["flat"]
