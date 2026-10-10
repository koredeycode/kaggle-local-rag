# Development

How to work on this codebase on a machine with no GPU, no model downloads, and
possibly no network.

## The unittest/pytest split

This project was built on a box with **no network and no pip**: Python 3.14, zero
third-party packages, `pypi.org` unresolvable. `pytest` cannot be installed or
run there.

So tests are written as `unittest.TestCase` classes:

```bash
make test          # stdlib unittest — works with nothing installed
make test-kaggle   # pytest — the §22 gate, where pip exists
```

Both run the same 225 test classes. pytest collects `unittest.TestCase`
subclasses natively, so nothing is lost when it is available; §22's "pytest
green" gate is satisfied unchanged on Kaggle.

This is a deliberate trade, not a preference. The alternative — pytest-style bare
asserts and fixtures — would match the plan's wording exactly and leave every test
unrunnable until a Kaggle session with pip. Verifying logic today is worth more
than matching a tool name.

`tests/base.py` documents the reasoning next to the code so it is not mistaken
for an accident.

## What needs no dependencies

Deliberately, the Phase 0 core is pure stdlib plus optional numpy:

- `types.py` — value types, JSONL helpers, hashing
- `errors.py` — the closed error-code set and single envelope
- `config.py` — frozen settings, env loading, the two config signatures
- `metrics.py` — p50/p95, counters, spans, Prometheus exposition
- `logging.py` — redaction filter and structured events
- `profiles.py` — protocols plus the `mock` implementations
- `tokens.py` — regex and identity tokenizers
- `chunking/` — all four strategies
- `index/bm25.py`, `index/rrf.py`
- `eval/` — retrieval metrics, citation metrics, abstention pair, reporting

§8 rule 1 is that a run must never require a GPU download to be testable, and
this is that rule taken seriously enough to keep the *whole* Phase 0 surface
dependency-free. torch, sentence-transformers, FastAPI, pymupdf, and python-docx
are all in optional extras.

## numpy is optional at runtime

`pyproject.toml` lists numpy in the core dependencies because the vector store
needs it, but nothing in the Phase 0 test suite imports it. Tests that would
exercise numpy paths skip cleanly when it is absent. If you are on a box like
this one, `make test` passes with numpy missing.

## Running the Phase 0 measurements

Two of the four §20 gates need no GPU and no models:

```bash
make measure-bm25     # §6.7 BM25 build cost        (~10s at 10k chunks)
make measure-cache    # §20 embedding-cache hit rate
make preflight        # §19.1 environment report
```

Both write to `eval/runs/` and both are seeded, so a later run is comparable with
an earlier one. Results and interpretation are in
[`docs/measurements.md`](docs/measurements.md).

`make measure-precision` (§5.5 fp16 vs fp32) needs `bge-m3` and realistically a
GPU. It ships; it has not been run.

## Linting and typing

```bash
make lint       # ruff check + ruff format --check
make typecheck  # mypy --strict
make check      # test + lint + typecheck
```

Both tools are in the `dev` extra and neither can be installed here, so they have
**not been run** on this code. The configuration is written to the same bar the
plan sets (strict mypy, ruff with a broad rule set including `ANN` and `D`), and
it is expected to surface findings on the first real run. Treat the first
`make lint` output as work to do, not as a regression.

## Style, in brief

- Every module has a docstring explaining what it is for and **why**, not what the
  code obviously does.
- Comments explain *decisions and their consequences*, especially the failure a
  line prevents. A comment restating the code is deleted.
- Where the plan states a rule that is easy to violate, it is enforced in code and
  tested: the index signature, the abstention pair, span redaction, chunk
  ordering. Conventions are what get quietly broken.
- `docstring`s cite the plan section they implement (`§9`), so a reader can tell
  which decisions are settled and which are still open.

## Adding a chunker

1. Subclass `BaseChunker`, implement `spans(doc) -> list[Span]`.
2. Decorate with `@register("name")`.
3. Declare `allows_overlap` honestly — `validate` enforces it.
4. Override `_effective_params()` with the parameters that actually move a
   boundary. These form the §6.5 index signature, so a knob that cannot change a
   chunk must not appear there.
5. Add a case to `TestDeterminism.test_same_input_same_output`.

Do not resolve offsets into text inside `spans`. The base class does that, so no
strategy can forget `clean_start`/`clean_end` and break the citation contract.
