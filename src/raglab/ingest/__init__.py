"""Ingestion: loaders, cleaning, structure, manifest (§9).

Phase 1. The loaders normalise every format into a `Document` with per-page
spans and a heading breadcrumb, `clean.py` emits the clean-to-original offset
map that the citation contract depends on, and `manifest.py` makes re-ingest
cheap by content hash.
"""
