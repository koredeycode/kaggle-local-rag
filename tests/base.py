"""Shared test helpers.

Tests are `unittest.TestCase` rather than bare pytest functions for a reason
specific to this box: there is no pip here, so pytest cannot be installed or run
locally. `unittest` is stdlib, so `python3 -m unittest discover -s tests` works
today *and* `pytest` collects the same classes unchanged on Kaggle, satisfying the
§22 `pytest` gate where pip is available. `docs/development.md` explains this.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture(name: str) -> str:
    """Read a text fixture by name."""
    p = FIXTURES / name
    if not p.exists():
        msg = f"fixture not found: {p}"
        raise FileNotFoundError(msg)
    return p.read_text(encoding="utf-8")


def make_document(
    text: str,
    *,
    doc_id: str = "sha256:test",
    path: str = "corpus/test.md",
    heading_path_at: tuple[tuple[int, str], ...] = (),
    pages: tuple = (),
    raw_len: int = 0,
):
    """Build a `Document` with sensible defaults for tests."""
    from raglab.types import Document, PageSpan

    return Document(
        doc_id=doc_id,
        path=path,
        text=text,
        pages=tuple(PageSpan(*p) for p in pages),  # type: ignore[misc]
        heading_path_at=heading_path_at,
        raw_len=raw_len or len(text),
    )


class RagTestCase(unittest.TestCase):
    """Base with assertion helpers for the properties this codebase cares about."""

    def assertOffsetsInRange(self, text: str, start: int, end: int, msg: str = "") -> None:
        self.assertGreaterEqual(start, 0, msg)
        self.assertLess(start, end, msg)
        self.assertLessEqual(end, len(text), msg)

    def assertMonotonic(self, values: list[int], msg: str = "") -> None:
        for a, b in zip(values, values[1:], strict=False):
            self.assertLessEqual(a, b, msg)

    def assertWithin(self, value: float, low: float, high: float, msg: str = "") -> None:
        self.assertGreaterEqual(value, low, msg)
        self.assertLessEqual(value, high, msg)
