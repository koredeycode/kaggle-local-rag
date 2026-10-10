# Ingestion

Companion to §9. Phase 1 scope; this document records the design now, at design
time, rather than reconstructing it later (§18).

## Loaders normalise once, up front

Every loader returns normalised `Document` objects, so every downstream stage sees
the same shape and normalisation never leaks into retrieval.

| Format | Library | Notes |
|---|---|---|
| PDF | `pymupdf` | Per-page extraction; page numbers retained for citation |
| Markdown | in-repo | Native; richest heading structure |
| DOCX | `python-docx` | Preserve heading styles |
| HTML | in-repo `html.parser` | Strip nav, header, footer, script, style |
| TXT | in-repo | Paragraph-break heuristic for pseudo-headings |

PDF is the hardest and is handled explicitly rather than pretended away:
two-column layouts interleave reading order, and tables shred into meaningless
fragments.

`doc_id` is `sha256:` over the raw bytes, and `path` is always **relative to the
session's `files/` directory** — never absolute, because a citation's `path` is
shown to the operator and must not leak the VM's filesystem layout.

## Structural context

Every chunk carries its heading path, discovered during parsing and stored on the
document as sorted `(offset, heading)` pairs:

```text
# Lagos State
## Education
### Fee Structure 2026
→ heading_path = ["Lagos State", "Education", "Fee Structure 2026"]
```

The breadcrumb is prepended to the **embedded** text. This costs nothing at
inference time and reliably outperforms larger chunk sizes in practice: a chunk
reading only "Term 1 is ₦45,000" is unretrievable, and one carrying its section
path is not.

Offsets rather than a per-character lookup table: `heading_path_at_offset` walks
the sorted list backwards so the innermost heading wins, and returns `[]` for flat
prose, which is the correct answer rather than a placeholder.

## Two coordinate systems, and the map between them

This is the section that matters most, because conflating the two is a silent
correctness bug rather than a visible one.

Cleaning mutates text. Chunks are cut from the **cleaned** text, retrieval matches
against the **cleaned** text, and the generator only ever sees **cleaned** text.
But a citation is supposed to point into the **original** file, because that is
the thing an operator can open and check.

| Space | Used by | Meaning |
|---|---|---|
| **clean** | chunking, retrieval, prompt assembly, span verification | offset into normalised document text |
| **original** | citations, the evidence panel, gold spans | offset into the file on disk |

So every chunk carries both, plus the map that produced them:

```python
class Chunk:
    clean_start: int          # what retrieval ranks
    clean_end: int
    orig_start: int | None    # None where not mappable — honest, not guessed
    orig_end: int | None
    page: int | None          # PDFs only
    heading_path: tuple[str, ...]
```

`ingest/clean.py` emits a `cleaned -> original` position map per document: a list
of `(clean_offset, orig_offset)` breakpoints, one per retained run of text.
Lookups are exact for text the cleaner kept verbatim and `None` where it did not.
Boilerplate removal and Unicode normalisation create those gaps honestly rather
than guessing at them.

### The rule that makes this load-bearing

> **Verification happens in clean space. Display happens in original space.**

Checking "does this span support this claim?" must run against the exact string
the model was shown, so it uses `clean_start`/`clean_end`. Pointing the operator at
the source file uses `orig_start`/`orig_end`. When the two are unavailable the
citation is still valid, and the evidence panel falls back to page + breadcrumb +
the retrieved clean text.

This is also why §15's gold spans are **clean-space** offsets: recall@k compares
gold against what retrieval actually ranked, so gold must live in the same space.

A citation response carries both pairs, because "which span supports this" and
"where is that in my file" are different questions with different answers:

```json
{
  "marker": "S1",
  "char_start": 8120, "char_end": 8467,
  "char_start_clean": 7944, "char_end_clean": 8291,
  "page": 4,
  "heading_path": ["Lagos State", "Education", "Fee Structure 2026"]
}
```

## Cleaning

- Normalise whitespace, Unicode, and line endings.
- Detect and drop repeated boilerplate across pages — headers, footers, page
  numbers — by frequency across the document.
- Drop near-duplicate chunks by similarity threshold, so boilerplate does not
  crowd the top-k.

All three rewrite text, which is exactly why the offset map is an artefact rather
than an afterthought. Dropping a page footer is what creates a gap between clean
and original space.

## Manifest

Content-hash based, per document:

```json
{
  "doc_id": "sha256:ab12...",
  "path": "corpus/fees-2026.pdf",
  "content_hash": "sha256:cd34...",
  "parser_version": "1.0.0",
  "cleaner_version": "1.0.0",
  "chunk_count": 412,
  "ingested_at": "2026-10-07T12:00:00Z"
}
```

On re-ingest, unchanged `content_hash` **and** unchanged `parser_version` **and**
unchanged `cleaner_version` means skip. That is what makes re-runs cheap enough to
iterate on inside a 12-hour session.

The session-level `manifest.json` carries what applies to the index as a whole:
`embed_model`, `embed_dim`, `vector_dtype`, `chunker`, `chunk_count` (physical
rows), and `live_chunk_count`.

### Version bumps force re-ingest

Changing the cleaner changes the clean-space coordinates of everything. So a
`cleaner_version` bump forces re-ingest of every affected document — the same path
as a `parser_version` bump, because the citation contract depends on both.

This is worth stating plainly: a cleaner change is not a refactor, it is a data
migration. Silently accepting new chunks alongside old ones would produce a corpus
whose offsets mean two different things, and every citation into the older half
would be quietly wrong.

## Current limitations

None of this is implemented yet — Phase 1. What exists today is the target shape:
`Document` and `Chunk` in `types.py` carry both offset spaces, `page_for_offset`
and `heading_path_at_offset` implement the lookups, `content_hash`/`text_hash` are
in place, and tests pin the semantics described above.

The offset map itself, the cleaners, and the loaders are Phase 1 deliverables, and
§22 requires a round-trip test on a fixture containing text the cleaner rewrites,
with unmappable regions reporting `None` rather than a wrong number.
