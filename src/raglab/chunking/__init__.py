"""Chunking strategies (§10).

Import for side effects: each module registers itself via `@register`, and
`base._load_builtins()` performs the imports. `get_chunker("structural")` and
friends therefore work without a caller knowing which module defines what.
"""

from raglab.chunking.base import (
    CHUNKER_VERSION,
    BaseChunker,
    Chunker,
    ChunkerSpec,
    Span,
    available,
    get_chunker,
    register,
)

__all__ = [
    "CHUNKER_VERSION",
    "BaseChunker",
    "Chunker",
    "ChunkerSpec",
    "Span",
    "available",
    "get_chunker",
    "register",
]
