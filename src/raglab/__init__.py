"""kaggle-local-rag: citation-backed RAG on local models, inside a Kaggle T4 x2 session.

Public surface is deliberately small. The modules are the API; re-exporting them
all here would create two import paths for the same thing and they would drift.
"""

from raglab.config import Profile, Settings, from_env
from raglab.errors import ComponentMismatch, ErrorCode, ModelUnavailable, RagError
from raglab.metrics import MetricsRegistry
from raglab.types import Answer, Chunk, Citation, Document, Hit

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "ComponentMismatch",
    "Document",
    "ErrorCode",
    "Hit",
    "MetricsRegistry",
    "ModelUnavailable",
    "Profile",
    "RagError",
    "Settings",
    "__version__",
    "from_env",
]
