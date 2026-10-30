"""sable-vector: embedded hybrid vector search on top of the sable LSM engine.

The engine keeps three things apart that are usually conflated:

* **Durability** belongs to ``sable``. Every mutation reaches a write-ahead log
  before it is visible, so a killed process loses acknowledgements, never
  acknowledged data.
* **Geometry** belongs to NumPy. Similarity is a matrix-vector product; the
  kernels here never loop over vectors in Python.
* **Filtering** belongs to the scalar inverted index, which is materialised as
  ordinary sorted keys so that a predicate becomes a prefix scan instead of a
  table walk.

Nothing in this package opens a socket, spawns a thread behind the caller's
back, or embeds a model. It is a library for a process that already owns its
own embedding model and wants its vectors to survive that process.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
