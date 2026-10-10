# Kaggle Local RAG — Plan

A comprehensive, citation-backed RAG system that runs on local models inside a free Kaggle
T4 x2 session, and doubles as a structured RAG curriculum.

This is a standalone project. It defines its own architecture, its own service contract, and
its own browser UI flow, and carries no dependency on any other repository.

---

## 1. Scope



### In scope

- Local-only inference. No third-party AI API key, ever. Cloud credentials are not required.
- Document ingest for PDF, Markdown, TXT, HTML, DOCX.
- Four interchangeable chunking strategies.
- Hybrid retrieval (dense + BM25) fused with reciprocal rank fusion.
- Cross-encoder reranking.
- Query transformation (rewrite, HyDE, decomposition, multi-query).
- A relevance gate that can abstain instead of hallucinating.
- Contextual compression and token-budget packing.
- Answers with citations requested in the prompt and validated after generation, resolved to
  exact source spans.
- An evaluation harness with ablations, using a local LLM judge.
- FastAPI service, browser client, observability, full `docs/` set.
- Incremental, resumable indexing that persists to `/kaggle/working`.

### Out of scope

- Production hosting, TLS, billing, multi-tenancy, SLA.
- Serving beyond a single operator on a temporary public URL.
- Model training, fine-tuning, LoRA.
- Distributed or multi-node anything.
- Images, audio, or video in the corpus (text-only extract).

---

## 2. Key terms to learn



This project is a curriculum first. These are the terms it teaches, in dependency order: you
cannot read `retrieval.md` without 2.1, and 2.4 is meaningless until you understand why 2.2
exists. Each entry says what it is in plain language, and why it matters *here* rather than in
general.

### 2.1 Corpus, chunks, and embedding

**Corpus** — the collection of documents the system answers over. The thing you index. Here it
is a directory of files on Kaggle disk. A RAG system is strictly limited by what is in its
corpus: a perfect model asked about a document it was never given will either refuse (correct)
or hallucinate (a bug). Knowing your corpus boundary is the first thing a user needs.

**Document** — one source file after parsing, normalised into text plus structure. Not the raw
file. A PDF becomes a `Document` with per-page text and heading paths.

**Chunk** — a retrievable slice of a document, typically 200 to 1000 tokens. The unit of
retrieval. Chunking is the single highest-leverage decision in naive RAG, and the main subject
of §10.

**Chunking strategy** — the rule that decides where to cut. Fixed-size cuts at N tokens;
recursive cuts at the best natural separator; structural cuts never cross a heading; semantic
cuts where topic meaning shifts. Different strategies suit different corpora, which is why this
project makes it swappable and ablates it.

**Overlap** — tokens repeated between adjacent chunks so a fact spanning a boundary survives in
one piece. Costs index size and creates near-duplicate candidates.

**Embedding** — a list of floats representing meaning, so that similar meanings land near each
other. "Term 1 is ₦45,000" and "first-term school fee costs forty-five thousand naira" land
close; a name and a phone number do not.

**Embedding model** — the network that produces embeddings. `bge-m3` here, producing 1024
dimensions. Chosen once and recorded in the manifest, because swapping it invalidates every
stored vector.

**Dimension** — the length of an embedding vector. `bge-m3` gives 1024. Determines raw storage
size: 100,000 chunks at 1024 dims in fp16 is about 200MB. Dimension is a capacity limit, not a
quality setting; two models with the same dimension are not interchangeable.

**Bi-encoder (dense retrieval)** — embeds the query and each document *separately*, then
compares vectors by cosine similarity. Because document vectors are precomputed at index time,
query time is a single fast matrix multiply, which is why this is the only affordable first
stage at scale. The weakness: it never sees the query and the document together, so a perfectly
relevant passage can embed poorly in isolation.

**Lexical retrieval (BM25)** — ranks by literal term overlap, with two refinements: term
frequency saturates (the tenth occurrence of "fee" adds far less than the first), and long
documents are penalised. Catches exactly what dense retrieval blurs — statute numbers, dates,
product codes, error strings, proper nouns.

**Hybrid retrieval** — running both and combining. Dense handles paraphrase; BM25 handles exact
strings. Their errors are largely uncorrelated, which is what makes combining them worth the
extra work.

### 2.2 Ranking and fusion

**Top-k** — how many candidates to retrieve. Too few and the right answer is never considered;
too many and the generator gets distracted, plus you pay to rerank and generate over noise.
`retrieve_top_k` and `context_top_n` are separate knobs on purpose: retrieve wide, then narrow.

**Reciprocal Rank Fusion (RRF)** — combines ranked lists without needing their scores to be
comparable. For each document, sum `1 / (60 + rank)` across every list that contained it. Uses
*positions*, not scores, so you never have to min-max normalise a cosine similarity against a
BM25 score and get it subtly wrong.

**Reranking** — a second, more expensive scoring pass over the top candidates. This is the
highest-return component in most RAG systems.

**Cross-encoder** — takes the query and a candidate *together* and outputs a relevance score
directly. It reads them jointly, so it can tell that a passage is relevant for *this* question
even when the passage alone looks like noise. The cost is that it cannot precompute document
scores, so it can only ever run over a short candidate list. This is the bi-encoder /
cross-encoder trade-off in one sentence.

**Reranker score** — an unbounded logit, not a probability. `bge-reranker-v2-m3` returns raw
values in roughly -8 to +5. Apply a sigmoid to map to [0,1] before display or thresholding, and
never compare scores across different queries.

**Maximal Marginal Relevance (MMR)** — selects candidates that are relevant *and* not redundant
with each other. Useful when the top-k are near-duplicates; a possible addition once the
baseline ablations exist.

### 2.3 Query transformation and context

**Query transformation** — rewriting the user's question before searching, because the query
and the document are often phrased differently. A conversation-scoped question like "what about
the second one?" is unsearchable until it is made standalone.

**Query rewriting** — normalise into a self-contained query. Cheapest transform.

**HyDE (hypothetical document embeddings)** — ask the model to write a plausible fake answer,
then embed *that* and search with it. Works because a fake answer is closer in style and
vocabulary to the real answer than a question is.

**Multi-query** — generate several paraphrases, retrieve each, fuse the lists. Reduces the
chance that one phrasing misses.

**Query decomposition** — split a multi-hop question into sub-questions, retrieve for each, then
merge. The standard route to questions that need evidence stitched from two places.

**Context window** — the maximum tokens a model can consider at once. `qwen3:8b` has 32,768
natively (Ollama's library reports 40K for the GGUF metadata; the HF base model is 32,768). It
is a hard ceiling, not a target.

**Token budget** — deliberately spending less than the window, to leave room for the question
and the answer and to reduce distraction. Packing more context is not obviously better; models
attend less reliably to what sits in the middle.

**Contextual compression** — dropping sentences from retrieved chunks that do not bear on the
query. Cheaper than better chunking and stacks with it.

**Small-to-big / parent-child retrieval** — index small chunks for precise matching, then
return their larger parent region to the generator. Small chunks match sharply, large chunks read
coherently, and you get both.

**Grounding** — attributing output to specific source material. A grounded answer names its
sources; an ungrounded one asserts the same fact from model memory.

**Faithfulness** — whether every claim is actually supported by the retrieved context.
Distinct from correctness: an answer can be factually right and still unfaithful, which is
exactly the failure mode a RAG system is supposed to prevent.

**Hallucination** — fluent output with no basis in the input. In RAG the specific worry is the
model ignoring the context and answering from its training data, then citing a span that does
not say that.

**Abstention** — declining to answer because the corpus does not cover the question. A feature,
not a failure. Measured by including unanswerable questions in the eval set — but only ever
alongside the overall abstention **rate**, since a system that refuses everything would
otherwise score perfectly (§15).

**Citation / span / provenance** — a citation pointing at an exact, verifiable location in a
source document: character offsets, page number, and heading path. The difference between "the
docs say so" and something you can check in one click.

**Clean space vs original space** — cleaning a document rewrites its text, so the same sentence
exists at two different character offsets. This project keeps both, because they answer
different questions: *verification* runs in clean space, since that is the text the model was
actually shown, while *navigation* runs in original space, since that is the file on disk. The
map between them is an artefact, not an afterthought (§9). Conflating the two produces
citations that point at the wrong text and a recall metric that is undefined.

### 2.4 Evaluation

**Recall@k** — of the gold chunks for a question, what fraction appear in the top k. The primary
retrieval health metric. For single-gold questions this equals hit-rate@k.

**Hit-rate@k** — fraction of questions where *at least one* gold item appears in the top k.

**MRR@10** — mean reciprocal rank: for each question, `1 / position of the first correct
result`, averaged. Rewards putting the right answer first, not merely in the list.

**nDCG@10** — normalised discounted cumulative gain. Credits graded relevance and discounts by
log position, so a relevant hit at rank 1 outweighs the same hit at rank 8.

**Ablation** — re-running the same eval with one component changed, to attribute an outcome to
a specific decision. The mechanism that turns "it feels better" into a number.

**Judge (LLM-as-judge)** — using a model to score output quality against a rubric, because human
scoring does not scale to an ablation matrix. Has real weaknesses, chiefly that a small local
judge is a weak judge, and that here the judge is also the generator, which biases it toward
its own output. Only valid as a self-consistent yardstick within one fixed setup.

**Faithfulness / answer relevancy** — the two generation-side metrics: is the answer supported by
its sources, and does it address the question.

**p95 latency** — the 95th percentile, not the average. Averages hide the slow tail, which is
what a user actually experiences.

### 2.5 Runtime and infrastructure

**Ollama** — a local model server exposing an OpenAI-compatible HTTP API. Lets one model serve
both the generator and the LLM-as-judge role, with no API key anywhere.

**GGUF** — the quantised model file format Ollama serves. Loading a quantised model is
inference; you never load the training weights.

**Quantisation** — storing weights at reduced precision to fit more of a model into limited VRAM.
`qwen3:8b` is 8.2B parameters stored at 5.2GB via Q4_K_M. It is the reason an 8B model fits on
a 16GB T4 at all.

**fp16 / bf16 / fp32** — floating point widths. T4 is Turing architecture and has no bf16, so
bf16 is unavailable. `bge-m3` is published at fp32; loading at fp16 saves VRAM at some cost to
embedding quality, which §5 requires you to measure rather than assume.

**Inference** — running a trained model to produce output, as opposed to training it. Everything
in this project is inference. No training anywhere.

**`CUDA_VISIBLE_DEVICES`** — an environment variable that decides which GPUs a process can see at
all. Used here as a hard boundary: Ollama sees only GPU 0, the embedder and reranker see only
GPU 1. Cheaper and more reliable than hoping two libraries share one device politely.

**Ephemeral session** — the Kaggle VM is destroyed when the session ends, including RAM. The
20GB auto-saved volume in `/kaggle/working` is the only thing that survives, which is why index
persistence is an architectural requirement rather than an optimisation.

**Quota** — 30 GPU-hours per week, 12 hours per session. The real budget of this project. Every
design decision that trades compute for latency has to be justified against it.

---

## 3. Hard constraints



### Kaggle (verified 2026-10-07)

| Fact | Value | Consequence |
|---|---|---|
| GPU options | **T4 x2** is the target | P100 is still listed on Kaggle and is the CLI default, but it is **sm_60** and silently breaks modern PyTorch (needs sm_70+), producing dead kernels. Design for T4. If pushing via CLI, set `machine_shape: NvidiaTeslaT4` or you land on P100 (§19.2). |
| GPU memory | 16GB per T4 (Turing) | No bf16. fp16 / int8 only. One 8B model per GPU. |
| CPU RAM | 29GB | BM25 index, corpus, and rerank batches all fit in memory. |
| CPU cores | 4 | Keep CPU work bounded. GIL-bound Python is the limit, not threads. |
| Persistent disk | 20GB auto-saved in `/kaggle/working` | Index persists. **This is what makes the system real.** Returns next session via the Input pane, **not** at the same path (§6.2). |
| Session cap | **12 hours** (GPU/CPU) | 30h/week ÷ 12h ≈ **2.5 sessions/week**. Indexing must fit in one session, or be resumable. Note the 9-hour figure in Kaggle's docs is the **TPU** limit. |
| Quota | 30 GPU-hours per week | About 4h/day. Budget deliberately. |
| Python | 3.12 on the image | Pin `requires-python = ">=3.10,<3.14"`. |
| Scratchpad | not saved | Anything outside `/kaggle/working` dies with the session |
| Auto-save | session end, needs Internet ON | A convenience, not a durability guarantee. Write-through instead (§6.2). |

### Local development box

| Fact | Value | Consequence |
|---|---|---|
| GPU | none | All GPU work is Kaggle-only. |
| RAM | 3GB | `bge-m3` will not fit alongside torch. Local runs need the `tiny` profile. |
| Disk | 35GB free | Fine. |
| Python | 3.14 | Outside the pin. Local is not fidelity-equivalent to Kaggle. |

**Design consequence:** local development validates logic, types, and API contracts. It does
not validate model quality or latency. Quality claims come only from Kaggle runs.

### Accuracy note

Model VRAM figures in this document are approximate. `scripts/preflight.py` must measure them,
and the plan must be corrected against measurements rather than trusted. The KV-cache figure in
§5.3 has been independently checked against Qwen3-8B's geometry (8 KV heads × 128 dim × 36 layers
= 144 KiB/token fp16, so 8192 tokens ≈ 1.15GB) and holds; everything else in §5.3 is estimated.

---

## 4. Architecture



### GPU split by workload, not by model

`T4 x2` is one 16GB device seen twice. Ollama will not split a sub-16GB model across both, so
without deliberate separation the second T4 sits idle. Split by *access pattern* instead:

| | GPU 0 — generation | GPU 1 — retrieval |
|---|---|---|
| Process | Ollama server | `sentence-transformers` worker |
| Models | `qwen3:8b` generator, and the same model as judge | `bge-m3` embedder, `bge-reranker-v2-m3` cross-encoder |
| Loaded | Persistent for the session | Per request, batched |
| Isolation | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |
| Access | Sequential, low concurrency | Batched, high throughput |

Why this is the right cut:

- Reranking is the highest-ROI component in RAG and it is throughput-bound, not latency-bound.
  Batching it on a dedicated device costs the generator nothing.
- The generator is latency-bound and wants its VRAM undivided.
- `CUDA_VISIBLE_DEVICES` is a cheap, hard isolation boundary. Neither process can see the
  other's memory, so neither can OOM the other.

### Component flow

```text
                    ┌──────────────── GPU 0 ────────────────┐
                    │  Ollama :11434                        │
                    │  qwen3:8b  ├─ generate               │
                    │  (same)    └─ judge (JSON mode)       │
                    └───────────────────▲────────────────────┘
                                        │ loopback HTTP
┌──────────┐   ┌──────────┐   ┌────────┴─────────┐   ┌──────────┐
│  ingest  │──▶│  chunk   │──▶│  retrieve/transform│──▶│ generate │
│ loaders  │   │ strategies│   │  rerank / gate     │   │  + cite  │
└──────────┘   └──────────┘   └────────▲─────────┘   └──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │  dense store (exact) + BM25     │
                    │  /kaggle/working/index (20GB)   │
                    └────────────────▲────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │  GPU 1 — sentence-transformers  │
                    │  bge-m3 embedder                │
                    │  bge-reranker-v2-m3 reranker    │
                    │  CUDA_VISIBLE_DEVICES=1         │
                    └─────────────────────────────────┘
```

### Exact search, not approximate

**Decision: brute-force exact vector search by default. No FAISS, no HNSW.**

At 1024 dimensions and fp16, a 100,000-chunk index is about 200MB. Exact search returns true
recall@k, which matters enormously in Phase 2 when you are comparing chunking strategies and
attributing recall changes to the chunker rather than to an approximate index's error.

**Where the matmul actually runs.** Decided here so §6.4, §6.7 and §11 do not each guess
differently:

| Index size | Search path |
|---|---|
| ≤ `dense_gpu_threshold` chunks (default 50k) | `np.load(mmap_mode="r")`, **upcast fp16→fp32**, matmul on the 4 CPU cores. Zero resident RAM, zero GPU contention. |
| above the threshold | copy to GPU 1 as one fp32 matrix once per session, reuse across queries |

The upcast is not optional. `numpy` has no fast `float16 @ float16` BLAS path — it falls back to
a slow reference loop, so a naive fp16 matmul turns "effectively instant" into seconds and reads
as "exact search is slow" rather than as a dtype bug. Upcasting also costs 2× peak RAM during
the multiply (400MB→800MB at 100k chunks), which is affordable against 29GB. The corpus is
memory-resident either way if it has been queried recently; mmap means it does not have to *be*
resident to be searched.

Adding an ANN index later would silently change your recall floor and invalidate the ablation
table. Ship it only when a measured corpus exceeds roughly 250k chunks, and re-baseline
everything when you do. Note this explicitly: approximate retrieval is a scale concession,
not a default.

---

## 5. Models and hardware budget



Verified against the Hugging Face model cards and the Ollama library on 2026-10-07. Sizes are
what will be downloaded; in-VRAM figures are estimates that `scripts/preflight.py` must
confirm in practice.

### 5.1 Models

| Role | Model | Params | Download | Notes |
|---|---|---|---|---|
| Embedder | `BAAI/bge-m3` | 568M | ~2.3GB fp32 | 1024 dims, 8192 seq, 100+ languages |
| Reranker | `BAAI/bge-reranker-v2-m3` | 0.6B | ~2.3GB fp32 | cross-encoder, xlm-roberta base, max_len 512 |
| Generator | `qwen3:8b` via Ollama | 8.2B | 5.2GB Q4_K_M | 32,768 ctx native, thinking toggle |
| Tiny embedder | `all-MiniLM-L6-v2` | 22M | ~90MB | 384 dims, for the `tiny` profile |
| Tiny reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ~90MB | for the `tiny` profile |

### 5.2 Why these

**`bge-m3` for embeddings.** Three reasons that matter here. It is **multilingual** across 100+
languages, which is insurance if any corpus text is not English. It handles **8192 tokens**, so
long passages do not have to be truncated into meaninglessness. And it uniquely emits **dense,
sparse, and multi-vector representations from one forward pass** — see 5.4.

**It needs no query instruction.** Earlier BGE models required prefixing queries with
`"Represent this sentence for searching relevant passages: "`. `bge-m3` does not, and the model
card states this explicitly. Getting this wrong is a silent quality bug, so it is pinned in a
test.

**`bge-reranker-v2-m3` for reranking.** It shares `bge-m3`'s backbone and language coverage, so
reranking scores stay comparable with the embeddings that produced the candidates. The model
card's own recommendation for a RAG pipeline is exactly the architecture in §11: hybrid retrieval
followed by reranking.

**`qwen3:8b` for generation and judging.** 5.2GB leaves ample room on a 16GB T4 for an 8K
context. Two properties matter: it is Apache 2.0, and it has a **thinking toggle**, which is
disabled by default (`gen_think=False`). Reasoning mode inflates latency and token use sharply,
and for grounded extraction it rarely earns its cost. Enable it only if ablation shows it helps.

**Both knobs need Ollama's native API, not the OpenAI-compat one.** `/v1/chat/completions`
accepts neither `num_ctx` nor `think`. `gen/ollama.py` therefore talks to `/api/chat`, where
`options.num_ctx` and `think` are real fields. If it used the OpenAI-compat route, `gen_ctx`
would silently do nothing and Ollama would size the KV cache from its own default instead —
which breaks the §5.3 budget without any error. Set the context in all three places to be
safe: `OLLAMA_CONTEXT_LENGTH` on the server, `options.num_ctx` per request, and
`PARAMETER num_ctx` in the Modelfile.

Known trap: `"think": false` has been reported as ignored on some Qwen3 tags, where reasoning
output appears regardless. `preflight.py` must assert that a trivial prompt returns no reasoning
block under `think=false`, and fall back to a `-no-think` model variant if it does.

### 5.3 Device and VRAM budget

```text
GPU 0  ── Ollama: qwen3:8b Q4_K_M          5.2GB weights
         + KV cache at 8192 ctx             ~1.0GB
         + CUDA context overhead             ~0.5GB
                                             ────────
                                              ~6.7GB of 16GB

GPU 1  ── bge-m3 fp16                       ~1.15GB
         + bge-reranker-v2-m3 fp16          ~1.15GB
         + batch activations (batch 32)      ~2GB
         + dense index resident (100k)       ~0.2GB
                                             ────────
                                              ~4.5GB of 16GB
```

Both fit with wide margins, which is the point of splitting by workload: the 16GB on each card
is enough that neither side is pressured. `embed_batch` is the first knob to lower if
`preflight.py` ever reports a tight card.

### 5.4 A capability worth exploiting

`bge-m3` produces three retrieval representations from a single forward pass:

| Output | What it is | Replaces or complements |
|---|---|---|
| `dense_vecs` | one 1024-dim vector per text | the dense stage |
| `lexical_weights` | per-token weights, BM25-like | **can replace** in-repo BM25 |
| `colbert_vecs` | per-token vectors, late interaction | an optional stronger stage |

The model card notes sparse weights come "without any additional cost when generating dense
embeddings". That means a learned lexical stage is available almost free.

**But do not remove in-repo BM25 on principle.** Learned sparse weights and BM25 are different
kinds of lexical signal: the learned one is tuned to the embedding model's notion of relevance,
while BM25 is an independent, inspectable, hand-verifiable function. Keeping both gives the
ablation three lexical variants to compare — BM25 alone, `bge-m3` sparse alone, and both fused —
which is better learning than picking one on faith.

ColBERT-style late interaction is genuinely stronger, and it is **infeasible on this store** —
not "deferred for now". Doing the arithmetic: 512 tokens × 1024 dims × 2 bytes ≈ **1MB per
chunk**. That is 100GB at 100k chunks against a 20GB store, and still 10GB at 10k chunks.
Supporting it would need a separate store, a different serving model, and its own re-baseline.
Revisit only if the corpus outgrows this design, and treat it as a new architecture rather than
a config flag.

### 5.5 Precision is an open measurement

`bge-m3` is published at fp32 and its card says fp16 "speeds up computation with a slight
performance degradation". "Slight" is not a number, and silent embedding degradation would
corrupt every retrieval metric downstream.

Phase 0 must therefore measure it: embed a fixed 200-chunk sample at fp32 and fp16, compute
cosine similarity between the two matrices, and report mean and minimum deviation. If the
agreement is poor, load the embedder at fp32 (it fits comfortably on GPU 1 per 5.3) and note the
VRAM cost. Do not guess this.

### 5.6 If the corpus turns out to need something else

| Situation | Change |
|---|---|
| Corpus is English-only and multilingual is unwanted | `bge-base-en-v1.5` (768 dims) is lighter; re-baseline everything |
| Reranking too slow | `bge-reranker-base`, or raise `rerank_top_n` down from 20 to 10 |
| Generator too weak for grounded extraction | `qwen3:14b` Q4_K_M ~9.3GB, still fits GPU 0 alone, slower |
| Corpus is mostly tabular | Embedding a table row-wise is poor; this needs a different retrieval design |

Any change here re-baselines the entire ablation table. Record model identity and revision with
every run (§15).
## 6. Session model and storage


Two things were under-specified until now: **where state actually lives**, and **what a session
is**. Both are answered here, because the storage layout is a consequence of the session model,
not the other way round.

### 6.1 What a session is

A session is a **titled, isolated workspace** owning its own documents, chunks, embeddings, and
evaluation runs.

```text
Session (title: "Lagos school fees 2026")
├── documents   PDFs, MD, TXT, DOCX, HTML uploaded by the operator
├── chunks      derived, per-document, with heading paths and offsets
├── embeddings  one fp16 row per chunk, 1024 dims
├── eval runs   question sets and results for this corpus
└── config      chunker, embedder, top-k, transform flags
```

A session is a **data boundary, not a compute boundary.** This distinction is the single most
important thing to get right, so it is stated explicitly:

| Shared across all sessions | Scoped to one session |
|---|---|
| `bge-m3` embedder (GPU 1) | documents |
| `bge-reranker-v2-m3` (GPU 1) | chunks |
| `qwen3:8b` via Ollama (GPU 0) | embeddings |
| the FastAPI process | evaluation runs |
| the 20GB store root | session title and config |

Ten sessions do not cost ten models. They cost ten indexes and share one set of loaded weights.
The alternative, one process per session, is impossible here: 16GB of VRAM per T4 cannot hold
ten copies of an embedder, and there is one 4-core CPU.

Two properties follow directly and are enforced rather than assumed:

1. **Isolation.** A query in session A can never read session B's data. Every store call takes a
   `session_id` and no store method has an "all sessions" default. A test asserts that a
   document id from one session is unreachable from another.
2. **Isolation is a feature, not a limitation.** An untrusted or exploratory corpus cannot
   contaminate the corpus you rely on. If a cross-session query is ever wanted, it is a separate
   explicit operation over a chosen set of sessions, never an implicit default.

Session identity is a generated `session_id`, not the title. Titles must be human-readable,
must be editable, and may contain spaces and punctuation, so they cannot be directory names.
Renaming a session therefore never moves files.

### 6.2 Where things live on Kaggle

Three distinct storage tiers, and confusing them is how people lose work:

| Tier | Path | Survives session end | Notes |
|---|---|---|---|
| Durable store | `/kaggle/working/ragstore/` | **Yes**, up to 20GB | The canonical home. Auto-saved. |
| Scratch | `/kaggle/working/ragstore/tmp/` | Yes, but disposable | Sorting, temp batch files |
| Session memory | RAM | **No** | Anything not written is gone |
| System scratch | anything outside `/kaggle/working` | **No** | Explicitly not saved |

The critical, non-obvious consequence: **saved output does not come back at the same path.**
Kaggle's documented flow is to re-attach it through the notebook editor's Input pane, which
mounts the previous run's output under `/kaggle/input/<notebook-slug>/`. So the next session
starts with the store *absent* from `/kaggle/working` and *present* read-only under
`/kaggle/input/`.

That drives the next decision, which is the one that actually protects the work:

**Write-through, never write-on-shutdown.** The index is written to disk as soon as it is built,
and every mutation is flushed immediately. Nothing is held in RAM waiting for a clean exit.
Session-end auto-save is treated as a convenience, not as the durability mechanism.

The failure this avoids is real and documented: users have lost full pipelines when a session
ended and auto-save did not complete in time. A 12-hour session killed by timeout is exactly the
case where deferred writes evaporate, and it is also the case where the quota was already spent.

### 6.3 On-disk layout

```text
/kaggle/working/ragstore/
  sessions/
    <session_id>/                 # generated id, never the title
      session.json                # id, title, created/updated, config, doc count
      documents.jsonl             # one record per document
      chunks.jsonl                # one record per chunk
      vectors.npy                 # (n_live_chunks, 1024) float16|float32, memory-mappable
      manifest.json               # content hashes + component versions + dtype/dim
      embedcache/                 # content-addressed chunk-text -> vector, §20
      files/                      # the uploaded originals
      eval/                       # question sets and run results
  tmp/                            # safe to delete at any time
```

### 6.4 The one invariant everything else depends on

```text
row i of vectors.npy  ==  line i of chunks.jsonl  ==  chunk.ordinal
```

Embeddings are stored as a plain `.npy` precisely so this holds and so the array can be opened
with `np.load(path, mmap_mode="r")`. The file is named `vectors.npy`, not `vectors.f16.npy`,
because §5.5 may switch the dtype to fp32 — the dtype lives in `manifest.json` and in the
`.npy` header, never in the filename.

`chunk.ordinal` is **session-global and monotonically increasing**. It is not per-document and
it is never reused, so a deleted chunk's ordinal stays burned.

That single choice buys three things at once:

- **Exact search without loading the index.** The matmul runs over memory-mapped pages (see §4
  for the CPU/GPU split), so opening a 400MB index costs no RAM.
- **Incremental appends.** A new document's chunks append to the end of both files, so an
  incremental ingest rewrites nothing already indexed.
- **No database.** No SQLite, no FAISS, no server. A directory of files that `numpy` can read is
  the entire storage engine, which is also the easiest thing to inspect when something is wrong.

A test asserts the invariant after every append: line count equals row count equals manifest
chunk count. Corruption here is silent and would poison every retrieval metric downstream.

### Deletion, and why it is the one place the invariant bends

`DELETE /api/sessions/{id}/corpus/{doc_id}` must remove a document's chunks, which are
scattered through the middle of both files. Deleting rows would break the ordinal invariant and
force a rewrite, so deletion is **logical, not physical**:

- Chunks are never removed from `chunks.jsonl` or `vectors.npy`. Rows are only ever appended.
- A `live` boolean on the chunk record marks rows as dead. `doc_id` presence in
  `documents.jsonl` is the authoritative membership test.
- A session's *effective* chunk count is the number of live rows, not the line count. Both are
  reported: `chunk_count` (live) and `chunk_rows` (physical), because their divergence is
  otherwise invisible and looks like storage drift.

Dense search multiplies against all rows including dead ones and filters after, which is cheap
at this scale and keeps the vector file append-only forever. BM25 filters to live rows before
building its postings, since that index is derived anyway.

The invariant test therefore asserts `rows == physical_chunk_lines == manifest.chunk_rows`, and
a *separate* test asserts `live_count <= rows` and that deletion preserves both counts in step.

### 6.5 What is persisted and what is derived

| Artefact | Persisted | Reason |
|---|---|---|
| Uploaded originals | Yes | Needed to re-chunk and to show source documents |
| Chunks with offsets | Yes | Expensive to recompute, and offsets are the citation contract |
| Embeddings | Yes | The expensive artefact; recomputing is the main cost to avoid |
| BM25 index | **No** | Derived from live `chunks.jsonl` on load; a persisted copy can only ever be stale |
| Chunk-text → vector cache | **Yes**, shared | Content-addressed by chunk text hash, so chunking ablations reuse embeddings instead of re-embedding (§20). Read-only and keyed by content, so it cannot leak across sessions. |
| `bge-m3` sparse / ColBERT weights | **No** | `lexical_weights` is a sparse vocab-sized vector per text; `colbert_vecs` is corpus-token-sized (≈1MB per 512-token chunk, see §5.4 — infeasible at this scale). Derived when enabled. |
| Query history, eval results | Yes | Cheap, and they are the record of what was learned |
| Model weights | No | Re-downloaded each session; nothing to gain from caching them |

Storing only irreducible artefacts means there is exactly one source of truth per fact and no
class of "stale derived index" bug. Everything derived is rebuilt from the manifest's recorded
component versions, so a `chunker` or `embedder` version change is detectable rather than
silently wrong.

### Detectability is not enough — mismatched components are rejected

Detecting a component change and then appending anyway is worse than not detecting it, because
the §6.4 invariant still passes while the index silently becomes a mixture. Concretely: build
with `fixed/512`, `PATCH` the chunker to `structural`, index again, and you now have two
chunking regimes in one matrix — invariant green, recall degraded, no error anywhere.

So `POST /api/sessions/{id}/index` compares the session's current `(chunker.name,
chunker.version, embed_model, embed_dim, vector_dtype)` against what the manifest recorded:

- **all equal** → append, as usual.
- **any differ** → **refuse with a 409**, naming the differing component. The response offers
  `?rebuild=true`, which is a destructive full re-chunk and re-embed.

Changing retrieval-only settings (`retrieve_top_k`, `context_top_n`, transform flags) does not
touch the index and is always allowed. The distinction is: settings that shape *stored data*
are pinned to the index; settings that shape *queries* are free.

`manifest.json` records `chunk_count` for physical rows plus `live_chunk_count`, plus
`embed_dim` and `vector_dtype`, so this check and the invariant test read the same fields.

### 6.6 Size arithmetic

Storage is deliberately the least binding constraint, so it is worth proving rather than hoping.

| Quantity | Figure |
|---|---|
| One fp16 vector, 1024 dims | 2,048 bytes exactly |
| 10,000 chunks | ~20MB vectors + ~4MB text |
| 100,000 chunks | ~205MB vectors + ~40MB text |
| 1,000,000 chunks | ~2.0GB vectors + ~400MB text |
| Sessions of 50k chunks each that fill 20GB | ~100 |

If §5.5 measures that fp16 embedding degrades retrieval, vectors become fp32 at 4,096 bytes
each, doubling every figure above and still leaving roughly 50 sessions at that corpus size.
Storage is not the constraint; VRAM and CPU time are.

`GET /api/sessions/{id}/index/status` therefore reports index bytes per session and total bytes
used, so growth is visible before the ceiling is hit rather than after. It also reports both
`chunk_count` (live) and `chunk_rows` (physical) per §6.4, so the two diverging is visible rather
than looking like storage drift.

### 6.7 Concurrency with one embedder

Many sessions share one GPU 1 embedder and one reranker, so concurrent queries must contend for
them deliberately:

- Embedding and reranking run behind an `asyncio.Semaphore`, with the limit a setting. Without
  it, several concurrent queries on 4 CPU cores and one GPU will thrash and every request slows
  down, including the first.
- Reranking is **batched across requests** rather than per-request, because the cross-encoder is
  throughput-bound and batching is where its throughput comes from.
- Dense search follows the §4 split: CPU matmul for corpora under
  `dense_gpu_threshold`, GPU 1 above it. Because most sessions stay on CPU, dense search does
  not contend with the embedder for GPU 1 in the common case.
- Ollama handles concurrent requests itself, but `keep_alive` must be set, or a model reloads
  between queries and the first request after an idle gap pays a multi-second penalty.

**The LRU evicts BM25 before it evicts handles.** `open_session_limit: 8` bounds memory-mapped
handles, but handles are nearly free — the real cost is each session's BM25 postings. Eviction
order is: BM25 index → dense GPU-resident matrix → memory-mapped handle, so the cheap-to-rebuild,
expensive-to-keep structure goes first. BM25 build time is **measured in Phase 0**, not assumed:
pure-Python tokenisation, document frequencies and length normalisation over 100k chunks on 4
cores is plausibly seconds, not "well under a second". If it is slow enough to be felt, session
open happens outside the query path (on `POST /index`, or on session selection in the UI) and
the semaphore in §6.7 covers index-open as well, so the first query after switching sessions
never reads as a hang.

### 6.8 Backup, because auto-save is not a backup

Auto-save is convenient, not guaranteed. A 20GB store whose only copy is inside a notebook's
output is one hard kill away from gone.

`scripts/export_index.py` exists for this and is the manual counterpart to auto-save:

- Package one session, or all sessions, into a `.zip` under `/kaggle/working` for download.
- Optionally publish to a **private** Kaggle dataset using the Kaggle CLI, which gives a named,
  versioned artefact independent of any notebook version.
- Print a `FileLink` so the archive is downloadable directly from notebook output.

The store is rebuildable from `files/` plus the manifest, so the irreplaceable assets are the
uploaded originals and the evaluation question sets. Both are small enough that exporting them
is cheap, and both are worth exporting deliberately rather than assuming.

### 6.9 Mobile access

A mobile-responsive UI is a requirement, not a polish item, and §17.6 specifies it. One platform
detail belongs here because it is about reaching the system rather than using it:

**The Quick Tunnel URL is long, and typing it on a phone is miserable.** The tunnel URL only
ever exists inside the notebook — the client is served *through* that tunnel and is never told
its own address. So the QR code is rendered by **cell 27 into the notebook output**, not by the
browser client: a `qrcode` PNG, or ASCII straight into the output cell.

That puts the QR where the operator actually is when they reach for the phone — looking at
Kaggle, not at the app on their laptop — and it removes a client dependency plus the unsolved
question of how the URL reaches the client. The runtime token is printed beside it so a phone
scan lands authenticated in one step. The client still displays the URL when the operator is on
a desktop, but it does not need to generate a QR at all.

## 7. Repository layout



```text
kaggle-local-rag/
  README.md
  PLAN.md                      # this document
  pyproject.toml
  Makefile
  .env.example
  .gitignore
  docs/
    glossary.md                # every term in §2, expanded
    architecture.md            # components, data flow, GPU split rationale
    sessions-and-storage.md    # session model, on-disk layout, durability rules
    models.md                  # model specs, device split, precision results
    ingestion.md               # loaders, cleaning, heading extraction
    chunking.md                # four strategies, trade-offs, when to use which
    retrieval.md               # dense, BM25, RRF, reranking, gating
    evaluation.md              # metrics, judge protocol, ablation methodology
    api.md                      # endpoint contracts, SSE event schema
    ui.md                      # views, Setup and Ask flows, every renderable state
    notebook-runbook.md        # cell-by-cell notebook structure, workaround catalogue
    security.md                # threat model, loopback discipline, tunnel policy
    observability.md           # metric names, latency semantics, redaction rules
    troubleshooting.md         # OOM, wrong device, tunnel failure, stale index
    production-migration.md    # what must change before real hosting
    kaggle-operations.md       # session runbook, quota budget, persistence rules
  src/raglab/
    __init__.py
    config.py                  # frozen Settings + from_env(), profile switch
    logging.py
    metrics.py                 # dependency-free p50/p95 counters and spans
    types.py                   # Document, Chunk, Hit, Citation, Answer
    errors.py
    profiles.py                # mock | tiny | local model resolution
    ingest/
      __init__.py
      base.py                  # Loader protocol
      pdf.py  markdown.py  text.py  html.py  docx.py
      clean.py                 # whitespace, boilerplate, dedupe
      structure.py             # heading path extraction
      manifest.py              # content-hash manifest
    chunking/
      __init__.py
      base.py                  # Chunker protocol + registry
      fixed.py  recursive.py  structural.py  semantic.py
      small_to_big.py          # child->parent index
    sessions/
      __init__.py
      model.py                 # Session, DocumentRef value types
      store.py                 # session CRUD, titles, listing
      scope.py                 # per-session resolution + isolation guarantees
      registry.py              # in-process session cache, LRU open handles
    index/
      __init__.py
      store.py                 # chunk store + dense matrix, exact search
      vectors.py               # .npy memory-map load/save, append + live-mask invariant
      embcache.py              # content-addressed chunk-text -> vector, for ablations
      bm25.py                  # in-repo BM25, derived not persisted
      rrf.py                   # reciprocal rank fusion
      persist.py               # write-through save/load under the store root
      export.py                # zip + optional private Kaggle dataset
    retrieval/
      __init__.py
      dense.py  lexical.py  hybrid.py
      rerank.py                # cross-encoder
      filters.py               # metadata predicates
      parent_expand.py
    transforms/
      __init__.py
      rewrite.py  hyde.py  decompose.py  multiquery.py
      gate.py                  # CRAG-style relevance gate, may abstain
    assemble/
      __init__.py
      compress.py              # contextual compression
      pack.py                  # token-budget packing
      number.py                # S1/S2 source labelling
    generate/
      __init__.py
      ollama.py                # client + streaming
      prompts.py
      cite.py                  # citation validation and resolution
    eval/
      __init__.py
      dataset.py               # question set with gold doc + gold span
      author.py                # authoring helper: propose + locate gold spans
      retrieval_metrics.py     # recall@k, MRR, nDCG, hit-rate
      gen_metrics.py           # faithfulness, relevancy, citation P/R
      judge.py                 # local LLM judge, JSON mode
      ablate.py                # ablation runner
      report.py                # markdown + json tables
    service/
      __init__.py
      app.py                   # FastAPI factory
      routes_query.py  routes_index.py  routes_corpus.py  routes_search.py
      routes_sessions.py       # session CRUD, rename, delete
      routes_health.py  routes_profile.py
      auth.py                  # single-operator runtime token
      concurrency.py           # embed/rerank semaphore, batched rerank queue
      sse.py                   # SSE event bus and ordering, heartbeats
      static.py                # static client mount
    bootstrap/
      __init__.py
      models.py                # pinned model manifest + download
      preflight.py             # moved to scripts/ in practice
  scripts/
    preflight.py               # JSON env + VRAM + device report
    smoke_test.py              # assert the pipeline answers, end to end
    cleanup.py                 # PID file, killpg, SIGTERM then SIGKILL
    export_index.py            # package index as a Kaggle dataset
  kaggle/
    README.md
    00_rag_system.ipynb        # launcher: clone, bootstrap, index, tunnel, serve
    01_ablation.ipynb          # evaluation runs
  static/
    index.html  app.js  citations.js  styles.css
  eval/
    questions.sample.jsonl     # seeded eval set, committed
    runs/                      # gitignored results
  tests/
    test_chunking.py  test_bm25.py  test_rrf.py
    test_citations.py  test_manifest.py  test_metrics.py
    test_api.py  test_gate.py
```

---

## 8. Configuration and profiles



Extends the `MODEL_MODE` pattern from `kaggle-avatar-prototype` into three explicit tiers.

| Profile | Embedder | Reranker | Generator | Where |
|---|---|---|---|---|
| `mock` | deterministic hash stub | stub scorer | scripted text | unit tests, CI |
| `tiny` | `all-MiniLM-L6-v2` (90MB) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | none, or remote/local stub | local integration tests |
| `local` | `bge-m3` (568M, 1024-dim) | `bge-reranker-v2-m3` (568M) | `qwen3:8b` via Ollama | Kaggle |

Rules carried over from the avatar prototype:

- `mock` is the default. A run must never require a GPU download to be testable.
- Failures degrade to a **visible, recoverable error**, never a silent model swap. Silent
  substitution is how you end up with eval numbers that do not mean anything.
- Profile is one setting, resolved once, exposed via `GET /api/profile`.

### Settings shape

```python
@dataclass(frozen=True)
class Settings:
    profile: str = "mock"                 # mock | tiny | local

    # Storage root is global; everything below it is per session (§6).
    store_root: str = "/kaggle/working/ragstore"
    vector_dtype: str = "float16"         # float32 if §5.5 measures fp16 degradation

    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_device: str = "cuda:1"          # hard boundary with Ollama on cuda:0
    embed_batch: int = 32

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cuda:1"
    rerank_top_n: int = 20                # candidates handed to the cross-encoder
    rerank_max_len: int = 512           # bge-reranker-v2-m3 hard limit; see §10

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_api: str = "native"          # /api/chat, not /v1/chat/completions — see §5.2
    gen_model: str = "qwen3:8b"
    gen_ctx: int = 8192                 # requires the native API to take effect
    gen_predict: int = 512
    gen_think: bool = False             # reasoning off for latency and cost

    retrieve_top_k: int = 40            # fused candidates
    context_top_n: int = 6              # spans passed to the generator
    context_token_budget: int = 3000

    gate_enabled: bool = True           # relevance gate is NOT a transform (§11)
    gate_max_spans: int = 6             # grades the post-rerank survivors, not all hits

    # One embedder and reranker are shared by every session (§6.7).
    embed_concurrency: int = 2            # simultaneous GPU 1 callers
    rerank_batch: int = 16                # pairs per cross-encoder forward pass
    open_session_limit: int = 8           # resident memory-mapped handles
    dense_gpu_threshold: int = 50_000     # above this, dense matmul moves to GPU 1 (§4)
    ollama_keep_alive: str = "30m"        # avoid a model reload between queries

    runtime_token: str = ""
    public_mode: bool = False
```

`gen_think` matters: Qwen3 reasons by default, which inflates latency and token use
substantially. Enable it only for hard multi-hop queries, if at all — and note both it and
`gen_ctx` only work against Ollama's native `/api/chat` (§5.2).

`gate_enabled` is separated from the §12 transforms deliberately. Transforms change *what is
searched* and are off by default because they need ablation deltas to justify their cost. The
gate changes *whether the system speaks at all*, which is a correctness property, so it
defaults on. Its cost is real and is accounted for in §11.

---

## 9. Ingestion



### Loaders

Each loader returns normalised `Document` objects. Normalisation happens once, up front, so
every downstream stage sees the same shape.

| Format | Library | Notes |
|---|---|---|
| PDF | `pymupdf` | Extract per page, keep page numbers for citation |
| Markdown | in-repo | Native; richest heading structure |
| DOCX | `python-docx` | Preserve heading styles |
| HTML | in-repo `html.parser` | Strip nav, header, footer, script, style |
| TXT | in-repo | Paragraph-break heuristic for pseudo-headings |

PDF is the hardest: two-column layouts interleave reading order, and tables shred into
meaningless fragments. Handle explicitly rather than pretending it is not a problem.

### Structural context

Every chunk carries its heading path, discovered during parsing:

```text
# Lagos State
## Education
### Fee Structure 2026
→ heading_path = ["Lagos State", "Education", "Fee Structure 2026"]
```

The breadcrumb is prepended to the embedded text. This is cheap, has no inference cost, and
consistently outperforms larger chunk sizes in practice, because a chunk that says only
"Term 1 is ₦45,000" is unretrievable while one carrying its section path is not.

### Cleaning

- Normalise whitespace, Unicode, and line endings.
- Detect and drop repeated boilerplate across pages (headers, footers, page numbers) by
  frequency across the document.
- Drop near-duplicate chunks, using a similarity threshold, so boilerplate does not crowd the
  top-k.

### Two coordinate systems, and the map between them

Cleaning mutates text. Chunks are cut from the **cleaned** text, retrieval matches against the
**cleaned** text, and the generator only ever sees **cleaned** text. But a citation is supposed
to point into the **original** file, because that is the thing the operator can open and check.

So there are two coordinate spaces, and conflating them is a silent correctness bug:

| Space | Used by | Meaning |
|---|---|---|
| **clean** | chunking, retrieval, prompt assembly, span verification | character offset into the normalised document text |
| **original** | citations, the evidence panel, gold spans | character offset into the file on disk |

Every chunk therefore carries **both**, plus the map that produced them:

```python
class Chunk:
    ...
    clean_start: int          # offset into normalised text — what retrieval ranks
    clean_end: int
    orig_start: int | None    # offset into the original file, None if not mappable
    orig_end: int | None
    page: int | None          # PDFs only
    heading_path: list[str]
```

`ingest/clean.py` emits a `cleaned -> original` position map per document (a list of
`(clean_offset, orig_offset)` breakpoints, one per retained run of text). Lookups are exact for
text the cleaner kept verbatim and `None` where it did not — boilerplate removal and unicode
normalisation create those gaps honestly rather than guessing.

The rule that makes this load-bearing: **verification happens in clean space, display in
original space.** Checking "does this span support this claim?" must run against the exact
string the model was shown, so it uses `clean_start`/`clean_end`. Pointing the operator at the
source file uses `orig_start`/`orig_end`. When the two are unavailable, the citation is still
valid and the evidence panel falls back to page + heading path + the retrieved clean text.

This is also why §15's gold spans are **clean-space** offsets: recall@k compares against what
retrieval actually ranked, so gold must live in the same space.

### Manifest

Content-hash based, extending the `models-manifest.example.json` idea:

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

Per-document records carry the parser and cleaner versions. The session-level `manifest.json`
carries what applies to the index as a whole: `embed_model`, `embed_dim`, `vector_dtype`,
`chunker`, `chunk_count` (physical rows) and `live_chunk_count`.

On re-ingest, unchanged `content_hash` + unchanged `parser_version` + unchanged `cleaner_version`
means skip. This is what makes re-runs cheap enough to iterate on, given 12-hour sessions.

Changing the cleaner changes the clean-space coordinates of everything, so a `cleaner_version`
bump forces re-ingest of every affected document — the same path as a parser bump, because the
citation contract depends on both.

---

## 10. Chunking



Four strategies behind one interface, so ablations can swap them freely.

```python
class Chunker(Protocol):
    name: str
    def chunk(self, doc: Document) -> list[Chunk]: ...
```

| Strategy | Idea | Strengths | Weaknesses |
|---|---|---|---|
| `fixed` | N tokens, overlap | Predictable, no model needed | Splits mid-thought |
| `recursive` | Split on the best separator, recurse | Respects natural boundaries | Still ignores meaning |
| `structural` | Never cross a heading boundary | Best for MD/DOCX/HTML | Coarse; useless for flat prose |
| `semantic` | Break where embedding similarity dips | Tracks topic shifts | Costs an embedding pass; slow on big docs |

Defaults to start: `fixed` at 512 with 64 overlap, `context_top_n = 6`, then `structural`.
Ablate over `{256, 512, 1024, 2048}` and all four strategies. The context fraction is
`context_top_n`, not a percentage of `retrieve_top_k` — one knob, one place (§2.2).

### The reranker's 512-token ceiling constrains this grid

`bge-reranker-v2-m3` has `max_len 512`. Any chunk longer than that is **silently truncated**
inside the cross-encoder, which is the component this project expects to produce its single
largest quality jump. Ablating 1024 and 2048 without handling this measures truncation, not
chunking.

Three options, and the plan takes the third:

1. **Cap chunk size at 512.** Simple, but gives up the long-chunk arm of the grid entirely.
2. **Window and max-pool.** Split each chunk into ≤512-token windows, score each, take the max.
   Faithful to "is any part of this relevant", at several times the rerank cost.
3. **Window and score the head**, but make the head *representative*: `heading_path` prepended
   plus a **query-aware** selection of the most relevant sentences within the chunk. Reuses the
   §13 compression filter before reranking, so what the cross-encoder sees is the part that
   matters.

Option 3 costs one embedding pass per candidate rather than several cross-encoder passes, and it
composes with compression instead of duplicating it. It is also an ablation result worth having
on its own: "compress-then-rerank" vs "rerank-then-compress".

Whichever is implemented, `docs/chunking.md` states the ceiling, and the §15 ablation table
records `rerank_window` per row so a 2048-token row is never compared against a 512-token row as
if the two pipelines were the same.

### Small-to-big

Index small chunks (about 200 tokens) for precise retrieval, then return the parent region
(about 1000 tokens) to the generator. Small chunks retrieve sharply; large chunks read
coherently. This resolves the precision-versus-recall tension that otherwise forces you to
oversize chunks and lose precision.

---

## 11. Retrieval



Pipeline, in order:

1. **Query embedding** via `bge-m3`.
2. **Dense search** — exact; CPU matmul under `dense_gpu_threshold`, GPU 1 above it (§4).
3. **BM25** — in-repo, Okapi BM25 over the live chunk store.
4. **Metadata filters** applied before both, not after.
5. **RRF fusion.** `score = Σ 1/(k + rank)`, k = 60. Rank-based, so no score normalisation
   between incomparable scales is needed, which is the usual source of hybrid bugs.
6. **Cross-encoder rerank** on the top ~20 fused candidates, windowed to `rerank_max_len` (§10).
7. **Parent expansion** on the survivors.
8. **Relevance gate** on the post-rerank survivors, not on every hit (§11).

Why hybrid matters: dense retrieval misses exact identifiers — statute numbers, dates,
proper nouns, error codes — because embeddings blur near-identical strings. BM25 catches
those; BM25 alone flails on paraphrase. RRF exploits the complementarity.

Why reranking matters: bi-encoders embed query and passage independently, so a relevant
passage can embed poorly in isolation. A cross-encoder reads them jointly. Expect this to
be the single largest quality jump in the whole project.

### BM25, in-repo

Roughly 60 lines with tokenisation, document frequencies, and length normalisation, plus
tests. Chosen deliberately: it is core RAG curriculum, it has no dependency, and it matches
the dependency-light ethos already in the avatar prototype. `rank_bm25` remains an
acceptable substitute if it saves time; the ablation table does not care which.

### The relevance gate

A CRAG-style step, and the one component that is **on by default** — see the `gate_enabled`
note in §8 for why it is not treated as a §12 transform.

The LLM grades spans as `relevant`, `ambiguous`, or `irrelevant`, with a reason.

- Confidently relevant: proceed.
- Mostly irrelevant: re-retrieve with an adjusted query, once.
- Still nothing: **abstain explicitly** and say what is missing.

**Cost, which has to be stated because it is easy to forget.** One LLM call over the assembled
candidates is not free on a T4: several seconds of prefill before generation even starts. That
makes the gate plausibly **the slowest pre-generation stage**, not reranking — which corrects
the assumption in §17.3 and is why `gate_ms` is a reported stage rather than being folded into
`retrieve_ms`.

Three consequences, all specified:

- The gate runs **after** reranking, over `gate_max_spans` (default 6) survivors. Grading every
  one of `retrieve_top_k` would multiply this cost by ~7 for no benefit — rerank has already
  removed the obviously-bad candidates.
- It is **one batched JSON-mode call** returning a verdict per span, not one call per span.
- It emits its own `gate_ms` in `stages`, and appears as its own row in the §17.3 stage
  readout. A user waiting on an unexplained pause is worse served than one told it is grading
  evidence.

Abstention is a feature. A system that answers confidently from bad retrieval is worse than
one that says "the corpus does not cover this", and the eval harness scores refusals as correct
when gold evidence is absent — **but only alongside the abstention rate**, because a system that
abstains on everything would otherwise score 15/15 (§15).

---

## 12. Query transforms



Each is a flag, each is measured, none is on by default. They cost latency and tokens, so
they earn their place through ablation deltas or they get removed. The §11 relevance gate is
**not** one of these — it defaults on, because it governs whether the system speaks at all.

| Transform | What it does | Cost |
|---|---|---|
| `rewrite` | Normalise conversation into a standalone query | One small call |
| `hyde` | Hypothetical answer, embed that instead | One generation + one embed |
| `decompose` | Split multi-hop into sub-queries, retrieve each, merge | One generation + N searches |
| `multiquery` | Paraphrase fan-out, fuse results | One generation + N searches |

Multi-query retrieval fuses several candidate lists and reduces the chance that one phrasing
misses the relevant chunk. Decomposition is what makes genuinely multi-hop questions possible.
HyDE helps most when the query is short or uses vocabulary unlike the corpus.

---

## 13. Context assembly



1. **Number** surviving spans `[S1]`, `[S2]`, ...
2. **Compress** each against the query — drop sentences with no lexical or semantic overlap.
   Cheap and dependency-light via an embedding-similarity sentence filter.
3. **Pack** to the token budget using the generator's own tokenizer
   (`AutoTokenizer.from_pretrained` for the Qwen family), never a character-count guess.
   Character heuristics drift silently as soon as text is not English prose.
4. Order spans most-relevant-first, and place the single best span last. Recency bias in
   decoder models means the tail of the prompt is disproportionately influential.

Exceeding the context window should be an explicit, reported truncation, not an overflow.

---

## 14. Generation and citation



### Prompt contract

The system prompt states: answer only from the provided sources, cite every claim, and if the
sources do not contain the answer, say so plainly. Sources are labelled `[S1]`..`[Sn]`.

### Citation resolution

After generation, each `[Sn]` marker is resolved back to a `Chunk` and surfaced with:

```json
{
  "answer": "Term 1 tuition is ₦45,000 [S1].",
  "citations": [
    {
      "marker": "S1",
      "doc_id": "sha256:ab12...",
      "path": "corpus/fees-2026.pdf",
      "heading_path": ["Lagos State", "Education", "Fee Structure 2026"],
      "page": 4,
      "char_start": 8120,
      "char_end": 8467,
      "char_start_clean": 7944,
      "char_end_clean": 8291,
      "score": 0.87
    }
  ],
  "abstained": false,
  "unverified_claims": [],
  "stages": {
    "retrieve_ms": 41, "rerank_ms": 180, "gate_ms": 940, "generate_ms": 920
  }
}
```

`char_start`/`char_end` are **original-file** offsets for display and navigation;
`char_start_clean`/`char_end_clean` are the offsets into the text the model actually saw, which
is what span verification runs against (§9). They differ wherever cleaning moved anything, and
they are both present because "which span supports this" and "where is that in my file" are
different questions.

Validating citations is not optional decoration. Three failure modes are common and all are
caught by post-generation resolution:

1. **Hallucinated markers** — `[S7]` emitted when only 6 sources existed.
2. **Over-citation** — every sentence citing everything, which looks like rigor and is noise.
3. **Unsupported claims** — uncited assertions that happen to be true from the model's
   parametric memory. These are the dangerous ones, because they read as correct.

Report citation precision, citation recall, and an `unverified_claims` flag.

**"Forced" is too strong a word, and the plan should not use it.** Citations are *requested* in
the prompt and *validated* afterwards; nothing constrains decoding, so an uncited sentence can
still be emitted. That is precisely why validation exists. There is a real alternative worth
ablation: a **repair pass** — send back the answer plus its unverified sentences and ask for a
citation or a retraction. It doubles generation cost on exactly the answers that need it, so it
is a Phase 6 experiment, not a default.

---

## 15. Evaluation



This is the deliverable that turns the project from a demo into an engineering artifact.

### Eval set

Evaluation is **per session**, because every metric is only meaningful against a specific
corpus. An eval set lives inside its session's `eval/` directory, is versioned, and travels with
the corpus it describes. Results from two sessions are never placed in one table, even when the
configuration is identical, because the corpora differ.

`eval/questions.sample.jsonl` is the seeded starter set, copied into a session on creation:

```json
{"qid": "q001", "question": "What is Term 1 tuition for 2026?",
 "gold_docs": ["corpus/fees-2026.pdf"], "gold_spans": [[7944, 8291]],
 "answerable": true, "tags": ["factual", "single-hop"]}
```

`gold_spans` are **clean-space** offsets (§9), because recall@k compares gold against what
retrieval actually ranked. Authoring them by hand means counting characters in a PDF, which
nobody does correctly twice — so `eval/author.py` is a Phase 2 deliverable, not a nicety:

1. The operator types a question.
2. It retrieves against the warm index and shows the top candidates.
3. The operator clicks the supporting chunk.
4. The tool emits the exact `gold_spans` from that chunk's `clean_start`/`clean_end`, plus a
   draft `question`.

Human judges relevance by reading, the tool handles coordinates. It also doubles as the
click-to-evidence path §17.5 wants.

Seed 60 questions across the corpus: 45 answerable spanning single-hop, multi-hop, and
paraphrased phrasing, plus **15 unanswerable**. Tag each with the retrieval property it
probes, because a single blended recall number hides which case is failing.

**This set is the critical path and no phase owns it**, which was the gap. It gates Phase 2's
baseline recall and every gate after it, so it is Phase 2 scope (§20), seeded earlier if
possible.

### Metrics

**Retrieval** — recall@k, hit-rate@k, MRR@10, nDCG@10. Computed against gold spans.

**Generation** — faithfulness (is every claim supported by its cited span?), answer relevancy
(did it actually answer?), citation precision, citation recall, and correct abstention on
unanswerable questions.

**Correct abstention is reported with the abstention rate, always.** Scoring refusals as
correct when gold evidence is absent is the right behaviour to reward, but on its own it is
trivially gameable: a system that abstains on every single question scores 15/15 on the
unanswerable set while being useless. So the eval reports, as a pair:

| Metric | Reads as |
|---|---|
| `correct_abstention` | of the 15 unanswerable questions, how many were correctly refused |
| `abstention_rate` | of **all 60**, how many were refused at all |

A good system sits high on the first and low on the second. A degenerate one shows up as
`correct_abstention` at ceiling with `abstention_rate` climbing toward 1.0. Reporting either
number alone is worse than reporting neither, because it invites exactly that conclusion.
`report.py` renders both columns for every ablation row that generates.

### Judge

`qwen3:8b` on GPU 0, JSON mode, thinking disabled. No external judge, no API key. Three
acknowledged consequences:

- An 8B judge is weaker than a frontier judge, so absolute faithfulness numbers are
  pessimistic and **only comparable against themselves**.
- **The judge and the generator are the same model**, which is textbook self-preference bias:
  the model is most likely to rate its own output as faithful. This cannot be fixed by model
  choice — a `qwen3:14b` judge does not fit beside an 8B generator on one 16GB T4
  (9.3 + 5.2 + KV > 16). Two mitigations that do fit, both free:
  - **Blind the judge** — never tell it which pipeline variant produced an answer.
  - **Randomise presentation order** across variants within a run.
  The remaining mitigation is the human sample below, which is doing more work here than the
  plan currently credits it for.
- For that reason, keep a fixed human-rated sample of ~20 answers as a calibration anchor, and
  report the gap between human and judge periodically. Without this, judge drift silently
  corrupts every phase-to-phase comparison.

**Rule: never compare numbers across different corpora, embedders, or judges.** Comparisons are
valid only within a fixed setup.

### Ablations

The core output. One command produces a table like:

| Chunking | Strategy | Retrieval | Recall@5 | MRR@10 | nDCG@10 | P95 ms |
|---|---|---|---|---|---|---|
| 512/64 | fixed | dense | 0.62 | 0.55 | 0.58 | 120 |
| 512/64 | fixed | hybrid | 0.68 | 0.61 | 0.63 | 135 |
| 512/64 | fixed | hybrid+rerank | 0.81 | 0.77 | 0.79 | 300 |
| 512/64 | structural | hybrid+rerank | 0.86 | 0.83 | 0.84 | 310 |

Whether reranking moves recall from 0.68 to 0.81, or structural chunking adds another 5 points,
is an empirical question, and the table is the answer. Run identical question sets, identical
corpus, identical judge, and log every config hash alongside results. Every row also records
`rerank_window` (§10), so a 2048-token row is never read as comparable to a 512-token row.

### Two matrices, because one grid costs more than the quota

The table above shows **retrieval-only** metrics, which run through `/api/sessions/{id}/search`
and generate nothing. That is not an incidental simplification — it is the difference between a
feasible ablation programme and an infeasible one, and conflating the two is how §21's estimate
got to 1.5–3.0h per run.

| | Retrieval matrix | Generation matrix |
|---|---|---|
| Metrics | recall@k, hit-rate@k, MRR@10, nDCG@10, p95 | + faithfulness, relevancy, citation P/R, correct abstention, abstention rate |
| Cost per config | embed once, then seconds per question | plus one 8B generation per question |
| Grid | large — the whole chunking × retrieval space | small — only the 3–4 configs worth a faithfulness claim |
| Corpus | subset (100–200 docs) is fine; deltas are what matter | full corpus, fewer questions |

Phase gates that only concern retrieval (baseline recall, hybrid delta, rerank delta) run the
cheap matrix and can therefore be large. Phase gates about grounding (compression delta,
citation P/R) run the generation matrix on a question subset.

**The chunking grid is an embedding cost, not just a search cost.** 4 strategies × 4 sizes = 16
index builds, at 0.5–1.0h per 1k documents (§21) — 8–16 GPU-hours, which is more than half the
weekly quota for one phase. Two mitigations, both specified in §20: a content-addressed
embedding cache shared across chunker settings, and running the chunking grid on a corpus
subset. Note that changing chunker settings against a warm index requires a rebuild, per §6.5.

### Determinism

Record seeds, model revisions, and library versions with every run. Without this, a 3-point
"improvement" is uninterpretable. Every run also records the full retrieval config hash,
`rerank_window`, and the live/physical chunk counts (§6.4).

---

## 15a. What the ablation programme costs

Worth stating separately from §21 because it is the single largest quota risk in the project,
and it was previously unbudgeted.

| Matrix | Configs | Per-config cost | Total |
|---|---|---|---|
| Chunking grid (4 strategies × 4 sizes) | 16 | one rebuild each | **8-16 GPU-hours** |
| Retrieval variants (dense / hybrid / +rerank / +gate) | 4 | warm index, seconds per question | < 0.5h |
| Query transforms (rewrite, hyde, decompose, multiquery) | 4 | warm index, 1 generation per question | 1-2h |
| Generation matrix (faithfulness, citation P/R) | 3-4 | full generation per question | 3-6h |

The chunking grid dominates because **every cell is a fresh index build**, not a search. Two
levers, both specified earlier:

- **Embedding cache.** Keyed by chunk text hash, shared across chunker settings. Heavily
  overlapping outputs across strategies mean most second-and-later builds are mostly cache
  hits. Phase 0 measures the real hit rate rather than assuming it.
- **Corpus subset for the chunking grid.** 100-200 docs. Deltas between chunkers hold; absolute
  recall does not, and the table reports deltas.

If the measured cache hit rate is low and the corpus is large, the honest move is to narrow the
grid — drop `semantic` (it needs an embedding pass per document anyway, on top of everything
else) and ablate sizes only on `fixed` and `structural`. Sixteen cells is a wish, not a
requirement.

---

## 16. Service, security, observability



### Endpoints

**Unauthenticated**, because the browser has to load the client and log in before it can send
anything:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/`, `/static/*` | the client itself |
| `GET` | `/healthz` | readiness, used by the §19.1 cell-25 tunnel check |
| `POST` | `/api/auth` | exchange the operator token for a session cookie |

**Authenticated** (bearer token or the cookie from `/api/auth`) — everything else, without
exception:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/sessions` | list sessions: id, title, doc count, chunk count, size, updated |
| `POST` | `/api/sessions` | create a titled session; title required, id generated |
| `GET` | `/api/sessions/{id}` | one session's metadata and config |
| `PATCH` | `/api/sessions/{id}` | rename title, or change query-only config (§6.5) |
| `DELETE` | `/api/sessions/{id}` | delete session and its index; confirm required in the UI |
| `GET` | `/api/sessions/{id}/corpus` | list documents and per-document status |
| `POST` | `/api/sessions/{id}/corpus` | add a document, queues ingest |
| `DELETE` | `/api/sessions/{id}/corpus/{doc_id}` | remove a document and its chunks |
| `POST` | `/api/sessions/{id}/index` | build or incrementally update this session's index |
| `GET` | `/api/sessions/{id}/index/status` | doc/chunk counts, index bytes, last run |
| `POST` | `/api/sessions/{id}/query` | answer with citations; `stream=true` streams |
| `GET` | `/api/sessions/{id}/search` | retrieval only, no generation, for tuning |
| `GET` | `/api/profile` | active profile, model identifiers, capabilities |
| `GET` | `/metrics` | requires auth, like everything else |

`POST /api/auth` takes the operator token the notebook printed and returns a short-lived cookie
the client then uses. It is an *exchange*, not a mint: sending a valid token and receiving a
token back is a no-op loop that tells the client nothing it did not already have.

**One error envelope, everywhere.** The §17.4 state table has fifteen distinct states and every
one of them branches on machine-readable data, so the shape is specified once here rather than
left to `docs/api.md`:

```json
{ "error": { "code": "index_component_mismatch", "detail": "chunker fixed/1.0.0 != structural/1.0.0", "session_id": "..." } }
```

Clients branch on `code`, never on `detail`. `detail` is for humans and may be reworded freely.
Codes are a closed set in `errors.py`, and the §17.4 table names one per state.

**Streaming is POST-only, deliberately.** Cloudflare Quick Tunnel buffers `GET` responses and
flushes them only when the connection closes ([cloudflared#1449](https://github.com/cloudflare/cloudflared/issues/1449),
observed, and still open), and the browser's `EventSource` API can only issue `GET`. `POST`
responses on the same route stream correctly. So `stream=true` returns standard
`text/event-stream` framing on a POST, consumed by the client via `fetch` and `ReadableStream`
rather than `EventSource`. Full rationale in §19.3. The framing stays standard, so a real
deployment reverts to `EventSource` by changing the method and nothing else.

### No unscoped corpus, index, query, or search route exists

Every route that touches corpus, index, query, or search state is nested under a session id, and
the session id is a path segment rather than a query parameter. That makes the scope visible in
the URL, makes an un-scoped call impossible to write by accident, and makes the isolation
guarantee checkable by reading the routes instead of auditing the store.

This is worth stating explicitly because the plan previously referenced unscoped
`/api/index/status`, `POST /api/index`, `/api/corpus`, and `/api/search` in the UI and risk
tables — each of which contradicts the guarantee above. There is no `/api/index` and no
`/api/query`. `/healthz`, `/api/profile`, `/api/auth`, and the static mount are the only paths
that are not session-scoped, and none of them read corpus state.

Every route that touches corpus or index state is nested under a session id, and the session id
is a path segment rather than a query parameter. That makes the scope visible in the URL, makes
an un-scoped call impossible to write by accident, and makes the isolation guarantee checkable by
reading the routes instead of auditing the store.

`/api/sessions/{id}/corpus` is a first-class surface, not an internal detail. Ingest progress,
per-document state, and removal all need to be visible and controllable from the client, and
having them as first-class routes means the browser flow in §17 needs no ad-hoc endpoints.

`/api/sessions/{id}/search` matters more than it looks. Tuning retrieval without paying for
generation is how you iterate in minutes instead of hours, and on a 30-hour weekly quota that
difference is the whole game. It is also what makes the large retrieval ablation matrix in §15
affordable.

### Security model

Carried over from the avatar prototype, unchanged in spirit:

- Everything binds `127.0.0.1` by default. This is the real boundary and it holds: the tunnel is
  the only ingress.
- The Cloudflare Quick Tunnel exposes **only** FastAPI. Never Ollama, never Jupyter, never a
  model dashboard. Ollama on `0.0.0.0` is an unauthenticated arbitrary-weight-execution
  endpoint on the public internet, and `nomic-embed-text` in their old README is proof this is
  an easy mistake to make.
- A locally generated runtime token. It is never written to disk, never logged, and never
  persisted client-side beyond `sessionStorage`. It **is** printed once, in notebook cell 27,
  because the operator has to be able to reach the system they just built; the resolved rule is
  "printed once to the operator, never stored, never logged", not "never displayed".
- **Every route requires auth except the three unauthenticated ones listed above** — static
  client, `/healthz`, and the token exchange. No exceptions for `/metrics`.
- **A loopback check on `/metrics` is not a control and is not used.** cloudflared runs *on the
  VM* and connects to `127.0.0.1:<port>`, so every tunnelled request appears to arrive from
  loopback. `client.host == "127.0.0.1"` passes for all public traffic and can never fail once
  the tunnel is up, which makes it a check that looks like protection and is not. Auth is the
  control; "binds 127.0.0.1" is the boundary. If a loopback restriction is ever genuinely
  wanted, it has to key on a forwarded header that cloudflared sets, not on socket peer address.
- No prompts, no retrieved text, no tokens, no secrets in metrics or logs.
- Quick Tunnels are public, random, temporary, and have no SLA. Cloudflare describes them as a
  demo product with deliberate guardrails, and recommends **named tunnels** for anything real.
  Say so in the README rather than implying otherwise, and treat a named tunnel as the first
  item in `production-migration.md`.
- Private clones use `GIT_ASKPASS` with a Kaggle Secret, never a token in the URL.

### Observability

Dependency-free p50/p95 counters, matching the avatar prototype:

- `retrieve_ms`, `rerank_ms`, `gate_ms`, `generate_ms`, `total_ms`, `embed_ms`
- `first_token_ms` for streaming
- `chunk_count` (live) vs `chunk_rows` (physical) per session, so §6.4 divergence is visible
- counts of: queries, abstentions, empty retrievals, citation failures, OOM-adjacent retries
- bounded ring buffer of recent spans with stage names and durations

Redaction rules live in `docs/observability.md` and are enforced by a test asserting that no
`Document`, `Chunk`, or prompt text reaches a log call.

---

## 17. Browser UI flow



Static HTML, CSS, ES modules. No Node build step, no bundler, no framework. Served by FastAPI
from the same loopback port as the API. The client is deliberately plain, because it must run
inside a Kaggle session reached over a Quick Tunnel, where a build step is a liability and a
CDN dependency is a hard failure.

The flow below is the contract. `docs/api.md` carries the matching endpoint and SSE schemas.
Layout is responsive and phone-first in behaviour; §17.6 specifies the breakpoint behaviour.

### 17.1 Session shell, then views

The app is scoped to **one active session** at a time. The session is the root of all state:
switching sessions swaps the entire document list, index, and question history, and never
leaves one session's chunks reachable from another.

```text
┌─ Session bar (always visible) ──────────────────────────────┐
│ [Session ▾]  Lagos school fees 2026 · 412 chunks · 24 MB     │
│                                          [+ New] [Rename]   │
├──────────────────────────────────────────────────────────────┤
│  View content: Setup or Ask, for the active session only     │
└──────────────────────────────────────────────────────────────┘
```

- **Session picker** opens a sheet listing every session with its title, document count, chunk
  count, size, and last updated. The active one is marked.
- **New session** asks for a title, required and free text. The generated `session_id` is never
  shown and never typed by hand.
- **Rename** changes the title only. It does not move files, because the directory is keyed by id.
- **Delete** requires a typed confirmation of the title, since it destroys the index and is not
  recoverable from inside the app.
- Switching sessions while a query is streaming cancels that stream first. Leaving a stream
  writing into another session's view is a cross-session leak through the UI even when the store
  is correctly scoped.

Below the bar, two views, no router library. The view is a function of whether the active
session's index is empty.

| View | Condition | Purpose |
|---|---|---|
| **Setup** | active session's index is empty or unbuilt | add documents, build index, inspect it |
| **Ask** | active session's index has chunks | query, read answers, inspect evidence |

Entering Setup when the session is empty is not a fallback state. It is the correct first-run
experience, because a query interface with nothing to retrieve is a dead end. Note the
condition is per session, not global: a fresh session shows Setup even when another session is
fully indexed.

### 17.2 Setup view

Drives `/api/sessions/{id}/corpus` and `/api/sessions/{id}/index`. Everything below is scoped to
the active session.

1. **Authenticate.** Browser prompts for the runtime token once; held in `sessionStorage`,
   never `localStorage`, so it does not outlive the tab.
2. **Add documents.** Multi-file picker or drag and drop, but **the browse button is always
   present and never hidden behind drag and drop**, because touch devices have no drag and drop.
   One `<input type="file" multiple>` plus an optional drop zone. Each file posts to
   `POST /api/sessions/{id}/corpus` and returns a `doc_id`.
   On a phone, `accept="application/pdf,.md,.txt,.docx,.html"` filters the picker, and
   `capture` is deliberately **not** used so photo-of-a-document does not silently become a
   low-quality image upload.
3. **Per-document status list.** Each row shows filename, format, byte size, a state, and a
   progress hint. States are exactly `queued`, `parsing`, `chunking`, `embedding`, `indexed`,
   `failed`. The state name is shown literally rather than as a percentage, because ingest
   progress is not honestly measurable per document and a fake percentage is worse than none.
4. **Build index.** `POST /api/sessions/{id}/index`. The manifest makes unchanged documents a
   no-op, so this is safe to press repeatedly and cheap on a warm index. Writes are flushed to
   disk as the index builds, so an interrupted build keeps completed work (§6.2).
5. **Index status strip.** Document count, chunk count, index bytes, last run timestamp, active
   profile. Sourced from `/api/sessions/{id}/index/status`.
6. **Remove.** `DELETE /api/sessions/{id}/corpus/{doc_id}` per row, with confirm. Removes the
   document and every chunk it contributed.
7. **Leave Setup.** Enabled only once at least one document in *this* session is `indexed`.

### 17.3 Ask view

Drives `/api/sessions/{id}/query`.

On a wide screen, layout is two panes: question and answer on the left, evidence on the right.
The split is the point of the product: the answer and its proof are visible at the same time.
On a narrow screen the same two panes become a stack with evidence in a bottom sheet (§17.6),
because losing the answer/proof adjacency is not an acceptable trade for screen width.

1. **Ask.** Text input, submit on Enter, or the send button (touch has no Enter key worth
   relying on). Question history is per session, so switching sessions shows that session's
   history and nothing else.
2. **Stages appear as they happen.** A compact stage readout, fed by streamed events:

   | Stage | Meaning |
   |---|---|
   | retrieving | dense, BM25, fusion in progress |
   | reranking | cross-encoder scoring candidates |
   | checking | the relevance gate is grading the survivors (§11) |
   | reading | generation has begun |

   Read via `fetch` + `ReadableStream`, never `EventSource`, for the tunnel reason in §19.3.
   The gate is likely the **slowest** pre-generation stage on a T4 — it is a full LLM prefill
   over the assembled candidates, not a matrix multiply — so it gets its own row rather than
   being folded into `retrieving`. This readout is what keeps a multi-second pause from looking
   like a hang.

   Latency without explanation is the usual complaint about local-model UIs. Showing where the
   time went is honest and also teaches the user what the system is doing.
3. **Answer streams in**, token by token.
4. **Citations resolve into chips.** Each `[Sn]` becomes a clickable chip. Markers referencing
   a source that does not exist, or an out-of-range span, render as a visibly broken chip
   rather than being dropped. A citation failure is a signal you need to see, not hide.
5. **Evidence panel.** On chip click: the exact retrieved span, its heading breadcrumb, page
   number for PDFs, character offsets, retrieval and rerank scores, and a link-style affordance
   to open the source document at that offset when the browser can render it. The offsets
   shown are **original-file** offsets for navigation; the span text displayed is the
   **clean-space** text the model actually saw (§9), and where the two disagree the panel says
   so rather than pretending the mapping is exact.
6. **Unsupported claims surfaced.** Any sentence with no citation is flagged inline, from the
   `unverified_claims` field. Visible by default, not behind a toggle.
7. **Abstention is its own state.** When `abstained` is true, the answer area shows a distinct
   abstention block naming what was missing from the corpus, never styled like either an answer
   or an error.
8. **No evidence found is distinct from an error.** Empty retrieval, a retrieval error, and a
   generation error render as three different states. Collapsing them makes a working system
   look broken.

### 17.4 States that must exist

| State | Trigger | Rendered as |
|---|---|---|
| Index empty | `GET /api/sessions/{id}/index/status` live count 0 | Setup view, with a reason |
| Indexing | `POST /api/sessions/{id}/index` in flight | Setup view, disabled submit, live status |
| Index component mismatch | `POST .../index` returned 409 `index_component_mismatch` | Name the differing component and offer rebuild; never silently append |
| Query streaming | SSE `token` events | Progressive answer text |
| No evidence | zero hits after gating | Explicit empty-retrieval state, no answer attempt |
| Abstained | `abstained: true` | Abstention block naming the gap |
| Citation broken | marker unresolvable | Broken chip, logged, surfaced |
| Retrieval error | dense/BM25 failure | Retrieval error state, with stage named |
| Generation error | Ollama failure or timeout | Generation error state, answer not truncated silently |
| Offline | SSE reconnect exhausted | Reconnect banner, input re-enabled |
| Token invalid | 401 from any route | Re-prompt for token, discard stale session |
| No sessions yet | `GET /api/sessions` empty | Create-first-session view, with the storage explanation |
| Session loading | switching session | Dimmed content, request cancelled, no stale data shown |
| Session deleted | `DELETE` returned, or 404 on refresh | Return to session picker, clear cached questions |
| Session title empty | `POST /api/sessions` without a title | Inline validation on the title field |

### 17.5 Why this flow

The design principle is **evidence first, answer second**. Retrieval stages are shown before
the answer exists, citations are validated rather than trusted, and uncited claims are flagged.
A local 8B model will produce plausible sentences that its context does not support, and the
only defence is making the evidence legible enough to check in one glance.

This is also what makes the system debuggable. When eval shows faithfulness at 0.71, the click
path from a bad answer to the span that failed to support it should take two clicks. That
debuggability is a real feature of the product, not a development convenience.

### 17.6 Mobile and responsive layout

Mobile is a first-class target, not a fallback. The operator's realistic pattern is a laptop
running the Kaggle session and a phone in their hand at the evidence panel, so the phone is the
reading device and the answer/proof adjacency must survive.

**Breakpoints.** One breakpoint, at 820px. Above it, two panes side by side. Below it, a single
column. Deliberately one breakpoint rather than a scale, because every extra breakpoint is
another state to test and there is no third form factor worth serving.

```css
.viewport { display: grid; grid-template-areas: "q e"; }
@media (max-width: 820px) {
  .viewport { grid-template-areas: "q" "e"; }
}
```

**Evidence becomes a bottom sheet on narrow screens.** Tapping a citation chip slides up a sheet
showing the same span, breadcrumb, page, and scores. It has a drag handle, a visible close
control, and traps nothing that matters. The alternative, pushing the answer off-screen, breaks
the one property the layout exists for.

Non-negotiables, each of which is a real bug if missed:

| Requirement | Why |
|---|---|
| `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">` | Without it, mobile browsers render a 980px desktop page and zoom out |
| `env(safe-area-inset-*)` padding | Otherwise the session bar sits under a notch or a home indicator |
| Touch targets at least 44x44px | Citation chips are the primary interaction; smaller ones are unusable with a thumb |
| **No hover-only affordances** | Hover does not exist on touch. Anything reachable only by hover is unreachable |
| No horizontal scroll at 320px | 320px is the narrowest viewport still worth supporting |
| Text stays at 16px minimum | Below that iOS Safari zooms on focus and breaks the layout |
| Long titles and paths truncate, never wrap the bar | A long session title must not push the layout off screen |

**Streaming on mobile is the hard part.** The `fetch` + `ReadableStream` path from §19.3 works on
mobile Safari, but phones add behaviours a desktop never sees:

- **Backgrounding kills the stream.** Switching apps suspends the connection and the reader
  stalls or errors. The client must treat `AbortError` from a background transition as
  expected: show a reconnect banner, keep the partial answer, and re-enable input. Never lose
  the partial answer, because on a slow 8B model it may represent a minute of work.
- **Reconnect uses `Last-Event-ID`.** The server keeps recent events in the ring buffer and
  replays from the last id seen, so a brief backgrounding resumes instead of restarting
  generation and paying for it twice.
- **Do not hold a wake lock.** It is tempting to keep the screen on, and it is the wrong trade
  for a tool; drain beats a slightly longer session here.
- **Upload works without drag and drop**, using the file input, which §17.2 already covers.

**QR code for the tunnel URL — rendered by the notebook, not the client.** A
`trycloudflare.com` URL is long and changes every session, so typing it on a phone is the
single largest friction point in the mobile path. Cell 27 renders it as a QR in the notebook
output, with the runtime token printed beside it: open the camera, scan, land authenticated.

The client does **not** generate the QR. The tunnel URL only exists inside the notebook, and the
operator holding the phone is looking at Kaggle rather than at the app on their laptop — so the
notebook output is where the QR needs to be. This also keeps a QR dependency out of the static
client and leaves unanswered the question of how the URL would reach the client at all. When
the operator is on a desktop, the client simply displays the URL as text.

**Testing.** Responsive layout is verified at 320, 390, 768, and 1280px widths, plus one real
device, because a CSS grid that looks correct in a desktop browser resized says almost nothing
about a phone. The state table in §17.4 is checked at each width.


---

## 18. Documentation set


Written in Phase 0, not retrofitted, so that design decisions are recorded at the moment they
are made rather than reconstructed later.

| Document | Contents |
|---|---|
| `glossary.md` | Every term in §2, expanded, with links to the section that implements it |
| `architecture.md` | Components, data flow, GPU split rationale |
| `models.md` | Model specs, device assignment, VRAM budget, precision measurements, Ollama API quirks |
| `sessions-and-storage.md` | Session model, on-disk layout, durability rules, append invariant, live-mask deletion |
| `ingestion.md` | Loaders, cleaning, heading extraction, manifest, the clean↔original offset map |
| `chunking.md` | Four strategies, trade-offs, when to use which, the reranker's 512-token ceiling |
| `retrieval.md` | Dense, BM25, RRF, reranking, transforms, gating |
| `evaluation.md` | Metric definitions, judge protocol, ablation methodology |
| `api.md` | Endpoint contracts, SSE event schema, error shapes |
| `ui.md` | Views, Setup and Ask flows, every renderable state |
| `notebook-runbook.md` | Cell-by-cell notebook structure and the workaround catalogue (§19) |
| `security.md` | Threat model, loopback discipline, tunnel policy |
| `observability.md` | Metric names, latency semantics, redaction rules |
| `troubleshooting.md` | OOM, wrong device, tunnel failure, stale index |
| `production-migration.md` | What must change before real hosting, starting with a named tunnel |
| `kaggle-operations.md` | Session runbook, quota budget, persistence rules |

Each document states what the system does **and why**, plus known limitations. No aspirational
descriptions of unimplemented behaviour; an explicit "Current limitations" section is the model
to follow.

`glossary.md` matters more than it looks. §2 is a glossary inside a plan, which is a reasonable
place to *learn* the vocabulary but a poor place to *look things up* while coding. The standalone
glossary is generated from the same source so the two cannot drift.

---

## 19. Kaggle notebook structure



The notebook is the deployment target, so its structure is a design artefact rather than an
afterthought. It follows the runbook discipline already proven on Kaggle: a single CONFIG cell,
assert-based verification at every boundary, and explicit teardown.

Two notebooks:

| Notebook | Purpose |
|---|---|
| `kaggle/00_rag_system.ipynb` | clone, bootstrap, index, tunnel, serve |
| `kaggle/01_ablation.ipynb` | run the eval matrix against a warm index |

### 19.1 Cell plan

| # | Type | Purpose |
|---|---|---|
| 0 | md | Title, one-time setup, quota warning, what to expect |
| 1 | code | **CONFIG** — the only cell edited per run |
| 2 | md | §1 Verify GPU |
| 3 | code | `nvidia-smi` + torch device count; assert 2 visible GPUs |
| 4 | md | §2 Clone repository |
| 5 | code | Clone or pull, `GIT_ASKPASS` if private, `pip install -e .` |
| 6 | md | §3 Preflight |
| 7 | code | Run `scripts/preflight.py`, assert device separation |
| 8 | md | §4 Install cloudflared |
| 9 | code | `zstd` first, then Ollama installer, then cloudflared `.deb` |
| 10 | md | §5 Start Ollama |
| 11 | code | `Popen`, `pkill` stale, readiness poll, assert `/api/tags` responds 200 |
| 12 | md | §6 Pull generator. Note `num_ctx`/`think` need `/api/chat`, not the OpenAI-compat route |
| 13 | code | `ollama pull`; **then** assert the exact name appears in `/api/tags` |
| 14 | md | §7 Warm the retrieval stack |
| 15 | code | Load `bge-m3` + reranker, assert both on `cuda:1` **and** Ollama on GPU 0 via `nvidia-smi --query-compute-apps`; report VRAM |
| 16 | md | §8 Build the index |
| 17 | code | Ingest and embed, or reuse an existing index |
| 18 | md | §9 Smoke test |
| 19 | code | `scripts/smoke_test.py`, assert one grounded answer with a citation |
| 20 | md | §10 Start the API |
| 21 | code | `Popen` uvicorn, log to `/tmp`, detect early exit |
| 22 | md | §11 Cloudflare Quick Tunnel |
| 23 | code | `Popen` cloudflared, scrape URL by regex, pin handle, write to `/tmp` |
| 24 | md | §12 Verify the public path |
| 25 | code | Retry loop through the tunnel for DNS lag, assert `/healthz` 200 |
| 26 | md | §13 Operator instructions |
| 27 | code | Print the token, the tunnel URL, and a **QR code of the URL** (§6.9); print the SSE caveat; define `cleanup()` |
| 28 | md | Troubleshooting and quota recovery |

Cells 3, 7, 11, 13, 15, 19, and 25 all **assert**. A Kaggle session that fails silently is far
more expensive than one that fails loudly in cell 3.

Note the ordering at cells 11 → 13: `/api/tags` is asserted as *reachable* before the pull, and
the exact model name is asserted *after* it. Asserting a non-empty model list at cell 11 would
fail on a clean run, because nothing has been pulled yet.

Cell 15's assertion is the device placement that matters, not just the count: both models land
on `cuda:1` while Ollama holds GPU 0. `preflight.py` additionally cross-checks
`nvidia-smi --query-compute-apps` so the assertion covers *which physical GPU each process is on*,
not just how many devices a process can see. Counting visible devices is the weaker check and
would pass even if both processes had landed on GPU 0.

### 19.2 Workaround catalogue

Every entry here is a specific failure already hit on Kaggle. They are listed because each one
looks like an unexplained flakiness until you know the cause.

| Symptom | Cause | Workaround |
|---|---|---|
| Ollama installer fails | Newer installers need `zstd` | `apt-get install -y zstd` **before** `curl ... \| sh` |
| Server dies when the cell ends | Shell `&` in a notebook cell is killed with the cell | `subprocess.Popen`, never `!cmd &` |
| Port already in use on re-run | Stale process from a previous cell | `pkill ollama` then `sleep(2)` before `Popen` |
| Connection refused right after start | Ollama not ready yet | Poll `GET /` up to 5 times with `sleep(3)`, break on 200 |
| `model not found` | Guessed tag rather than the real one | `ollama pull` first, **then** `assert MODEL_NAME in /api/tags` |
| `num_ctx` seems to do nothing | Used Ollama's OpenAI-compat route | `/v1/chat/completions` accepts neither `num_ctx` nor `think`; use `/api/chat`, and set `OLLAMA_CONTEXT_LENGTH` too (§5.2) |
| Reasoning output despite `think: false` | Some Qwen3 tags ignore the flag | `preflight.py` asserts no reasoning block on a trivial prompt; fall back to a `-no-think` variant |
| Dead kernels, no error, GPU shows P100 | **P100 is sm_60; modern PyTorch needs sm_70+** | Select T4 x2 in the notebook UI, or set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` when pushing via the CLI. `enable_gpu: true` alone lands on P100. |
| Exact search feels slow at 100k chunks | fp16 matmul fell back to a slow numpy loop | Upcast fp16→fp32 before the multiply (§4); numpy has no fast `float16` BLAS path |
| Session switch feels like a hang | BM25 rebuilt synchronously on open | Move session-open outside the query path; measure BM25 build time in Phase 0 (§6.7) |
| Tunnel handle vanishes mid-session | `Popen` object garbage collected | `builtins.tunnel_process = p` |
| Tunnel URL not printed yet | cloudflared takes 5-15s | Regex the log line, 60 iterations, `bufsize=1` |
| URL detection failed entirely | Flaky scrape | Write to `/tmp`, allow manual paste, assert `https://` prefix |
| First request through tunnel times out | Quick Tunnel DNS propagates in 20-60s | 6 attempts, `sleep(15)` between |
| `403` on the tunnel root | Meaningless response from the edge | Only ever assert on a real route |
| App exited silently | Crash before first request | `poll()` check, then dump the `/tmp` log |
| Private clone prompts for credentials | No TTY in a notebook | `GIT_ASKPASS` script from a Kaggle Secret, `unlink` after |
| Optional model bootstrap fails | Partial download, OOM, bad asset | `try/except`, terminate Ollama, fall back to `mock` visibly |
| cloudflared version drifts mid-session | Auto-update between cells | `--no-autoupdate`, optional pinned URL + SHA256 |
| Run All kills the demo you wanted to use | Cleanup fired too eagerly | `CLEANUP_ON_RUN_ALL = False`, call `cleanup()` manually |
| Index gone next session | Session-scoped RAM or temp disk | Everything durable in `/kaggle/working` |
| `pip install` fights Kaggle's preinstalled torch | Version conflict | Do not pin torch; let the image's version stand |
| **Answer streams all at once** | **Quick Tunnel buffers GET responses** | **POST + `fetch` + `ReadableStream`, never `EventSource`** |

### 19.3 Streaming through the tunnel

**Quick Tunnel buffers GET responses and flushes them only when the connection closes.** This is
observed and still open ([cloudflared#1449](https://github.com/cloudflare/cloudflared/issues/1449)),
not a misconfiguration. `GET` SSE arrives frozen, then all at once.

`EventSource`, the browser's native SSE API, can only issue `GET`. So `EventSource` does not
stream through a Quick Tunnel in practice. POST-based SSE on the same route streams correctly, in
real time, with identical framing.

**Treat this as a measurement, not a law.** The evidence is one open issue, and two things
qualify it:

- Cloudflare's maintainer replied that quick tunnels carry **deliberate guardrails because they
  are a demo product**, and that named tunnels are what the product expects. That is a product
  restriction, not a general SSE defect — so it may not describe named-tunnel or proxied
  traffic at all.
- A flush-on-write improvement landed on Cloudflare's edge mid-2025
  ([cloudflared#1496](https://github.com/cloudflare/cloudflared/issues/1496)), which fixed SSE
  buffering on proxied traffic. The behaviour may have changed since the issue was filed.

So the decision is made on current observation, the prose does not claim impossibility, and the
client is written so that switching back to `EventSource` is a one-line change. Re-verify in
Phase 0 each time, and prefer a **named tunnel** the moment one is available — it is the
supported path and removes this entire constraint.

The resolution, and it is a real constraint rather than a preference:

| Option | Verdict |
|---|---|
| `EventSource` over Quick Tunnel | **Observed broken.** Buffered until close (#1449). |
| `POST /api/sessions/{id}/query` + SSE framing, read via `fetch` and `ReadableStream` | **Works.** Requires hand-parsing SSE frames in ~40 lines. |
| WebSocket | **Works.** More machinery. |
| Named tunnel + `EventSource` | **The supported path.** Preferred once available; see `production-migration.md`. |

This project takes **POST + SSE + `ReadableStream`** on a Quick Tunnel. The framing stays
standard, so switching the method restores `EventSource` unchanged. Hand-parsing is a
deliberate, reversible cost paid for a demo-grade transport.

Two supporting requirements:

- **Heartbeats.** Emit an SSE comment (`: heartbeat\n\n`) every 15s. The rerank stage can take
  a couple of seconds and generation is slower still, so an idle gap can trip intermediary
  timeouts on a flaky consumer path.
- **Buffer locally, stream remotely.** The client always works against the same `ReadableStream`
  reader whether the origin is loopback or a tunnel, so the code path is identical and testable
  without a tunnel in the loop.

### 19.4 Session cost discipline

The index is built once and reused, so a steady-state session is short. Cell 1's `CONFIG` cell
carries the switch between `mock` and `local`, and every model-loading cell is gated on it, so a
code-editing pass never touches the network.

Order matters: preflight and install happen before any download, models before the index, index
before the smoke test, and the smoke test before the tunnel. Every later stage depends on
everything above it, so failing early is always cheaper.
## 20. Delivery plan



Each phase is one PR, one doc update, and one measured delta. Phases 0 and 1 need no GPU.

| Phase | Objective | Learning | Acceptance gate |
|---|---|---|---|
| **0** | Scaffold, `mock` profile, preflight, metrics, eval skeleton | Repo shape, types, profiles, tests | `pytest` green; eval harness runs with zero GPU; **fp16 vs fp32 embedding agreement measured (§5.5)**; **BM25 build time measured on a 100k-chunk sample (§6.7)**; **POST-streaming proven through a real Quick Tunnel (§19.3)**; **embedding cache hit rate measured on two chunker settings (§20)** |
| **1** | Loaders, cleaning, **clean↔original offset map**, manifest, session store, **write-through persistence**, embedding cache | Ingestion and its failure modes | Re-ingest skips unchanged docs; offsets round-trip in both directions; two sessions populated and proven mutually invisible; a killed process loses nothing already written |
| **2** | Fixed + recursive chunking, dense index, exact search, **`eval/author.py`**, first 60 questions | Why chunk size dominates naive RAG | Baseline recall@5 recorded against a real gold set |
| **3** | BM25 + RRF | Lexical/dense complementarity, why RRF avoids normalisation bugs | Hybrid vs dense delta measured |
| **4** | Cross-encoder rerank, **512-token windowing** | Bi-encoder vs cross-encoder | Rerank delta measured, expected large |
| **5** | Rewriting, HyDE, decomposition, gate | Query-doc mismatch, abstention | Transform deltas measured; gate delta measured **with abstention rate reported alongside correct abstention** |
| **6** | Small-to-big, compression, citations, packing | Precision/recall tension, grounding | Compression delta; citation P/R measured |
| **7** | Session routes, FastAPI, streaming, UI, tunnel | Serving, streaming, trust UX | Setup and Ask work per session; all §17.4 states render at 320/390/768/1280px; streaming survives phone backgrounding |
| **8** | Input-pane restore, `export_index.py`, notebook QR | The Kaggle constraint that shapes everything | New Kaggle session restores from `/kaggle/input/` with no re-embedding |
| **9** | Judge calibration, final ablation matrix, README | How to report honestly | Full table; human-vs-judge gap reported |

Phases 4 and 5 are where the real learning is. Expect to spend a disproportionate share of
quota there, and expect reranking (Phase 4) to produce the largest single gain.

Three re-orderings from the original plan, each fixing a sequencing error rather than changing
scope:

- **Durability moved from Phase 8 into Phase 1.** §6.2 argues persistence is the constraint that
  shapes everything, and §23 lists auto-save failure as a top risk. Retrofitting write-through
  onto an index designed to live in RAM is exactly the refactor that produces the lost-index
  failure §6.2 exists to prevent. Phase 8 keeps only Input-pane restore and export.
- **The eval set moved into Phase 2, where the first gate depends on it.** Nothing owned it,
  and Phases 2–6 all gate on gold spans that do not exist until it does.
- **The embedding cache lands in Phase 1**, because the chunking ablation grid in Phase 2 is
  where it pays off and it must exist before the first rebuild.

---

## 21. GPU budget



**30 hours per week, 12-hour session cap (the 9-hour figure in Kaggle's docs is the TPU
limit), T4 x2.** That is roughly 2.5 sessions per week.

| Activity | Session estimate | Notes |
|---|---|---|
| Bootstrap, model pull, preflight | 0.5-1.0h | Mostly download, not compute |
| Ingest and embed 1k docs | 0.5-1.0h | Once, then persisted |
| Ablation: retrieval only, warm index | 0.5-1.5h | `/api/sessions/{id}/search`, no generation |
| Ablation: full with generation | 1.5-3.0h | Expensive; use a question subset |
| **Chunking grid (4 strategies × 4 sizes = 16 rebuilds)** | **8-16h** | **The unbudgeted item — see below** |
| UI iteration | 1-2h | Index already warm |

**The chunking grid is the budget risk.** 16 index builds at 0.5-1.0h per 1k documents is 8-16
GPU-hours — more than half the weekly quota for one phase, and previously absent from this table
entirely. Two mitigations, both now in the plan:

1. **Content-addressed embedding cache** (`index/embcache.py`, Phase 1), keyed by chunk text
   hash and shared across chunker settings. `fixed/512` and `recursive/512` overlap heavily, so
   the second build reuses a large fraction of the first. Phase 0 measures the actual hit rate
   rather than assuming it.
2. **Run the chunking grid on a corpus subset** (100-200 docs) and the retrieval matrix on the
   full corpus. Recall@5 *deltas* are comparable across chunkers on a subset; absolute numbers
   are not, which is fine because the table reports deltas.

Roughly **two sessions per week, about 5 hours total** is comfortable and leaves headroom *for
ordinary work*. One full chunking grid eats most of a week, so it is planned as a deliberate
spend in Phase 2, not something run opportunistically.

Rules:

- Shut down the session when idle. An idle tunnel burns quota.
- Use the `None` accelerator for code editing. Only attach GPU to run.
- Persist the index so bootstrap time is paid once, not per session.
- Prefer retrieval-only tuning over full-generation tuning.
- Split retrieval and generation matrices (§15) so phase gates do not implicitly require
  generation.

---

## 22. Test strategy



- Unit tests per chunker: fixed invariants (size bounds, overlap, no mid-boundary splits for
  `structural`), and a determinism check.
- BM25 unit tests against a hand-computed case, not just regression output.
- RRF property test: monotone in rank, invariant to score scale.
- Citation tests: hallucinated marker, over-citation, out-of-range span, marker not in sources.
- Manifest test: unchanged content is skipped; changed parser version forces re-ingest.
- **Dense-search reference test:** exact search on a small hand-computed matrix must match a
  brute-force reference. This is the test that catches the fp16-matmul footgun in §4 — and the
  missing test that would have caught it.
- **Deletion test:** deleting a document preserves `rows == chunk_rows` while `live_count` drops
  by exactly that document's chunks, and retrieval no longer returns it (§6.4).
- **Index-component-mismatch test:** changing the chunker and re-indexing returns 409 rather than
  appending (§6.5).
- **Offset-map test:** clean↔original offsets round-trip on a fixture containing text the
  cleaner rewrites, and unmappable regions report `None` rather than a wrong number (§9).
- **Rerank-batch association test:** two concurrent queries' candidate lists must not be
  cross-contaminated by cross-request batching (§6.7). Misattributing a score to the wrong
  query is trivial to write and catastrophic to retrieval quality.
- Metrics test: p50/p95 on a known sequence.
- Redaction test: assert no `Document` or `Chunk` content reaches logs.
- API tests: every route except static, `/healthz`, and `/api/auth` requires auth; `stream=true`
  emits ordered events; error bodies use the single envelope and clients branch on `code`.
- **Device-placement test:** `preflight.py` confirms Ollama on GPU 0 and the retrieval worker on
  GPU 1 via `nvidia-smi --query-compute-apps`, not merely visible-device count (§19.1).
- Smoke test: `scripts/smoke_test.py` asserts one real end-to-end answer, against whichever
  profile is active.

`mypy --strict` clean, `ruff check` clean, `ruff format --check` clean. Same bar as the avatar
prototype, because that bar is already met and lowering it would be a regression.

---

## 23. Risks



| Risk | Impact | Mitigation |
|---|---|---|
| T4 (Turing) fp16 instability vs fp32 | Silent embedding quality loss | Measure fp16 vs fp32 cosine similarity on a sample in Phase 0; fall back to fp32 for the embedder if it degrades. Reranker on fp16 is usually fine. |
| **`gen_ctx` / `gen_think` silently ignored** | **KV cache sized by Ollama's own default; §5.3 budget invalid without error** | Use native `/api/chat`, set `OLLAMA_CONTEXT_LENGTH` too, and assert no reasoning block under `think=false` in preflight (§5.2) |
| **fp16 matmul falls back to a slow numpy loop** | **Exact search reads as slow; looks like a design flaw, is a dtype bug** | Upcast fp16→fp32 before the multiply (§4); reference test in §22 |
| OOM when Ollama and the embedder collide | Session dies mid-run | Hard `CUDA_VISIBLE_DEVICES` split, `embed_batch` config, preflight VRAM report, per-PID device check |
| 12-hour session cap hit mid-index | Wasted quota | Manifest makes indexing resumable; persist early and often |
| 20GB disk ceiling | Index unpersistable | Track index size in `GET /api/sessions/{id}/index/status`; embeddings are compact, but monitor |
| Small 8B generator is weak at grounded generation | Low faithfulness | Request citations, validate them, measure, report `unverified_claims`. Better retrieval beats a bigger model. |
| Judge drift between phases | Invalid comparisons | Fixed judge revision, human calibration sample, config hashes in every run |
| 8B judge is a weak judge | Pessimistic absolute numbers | Only self-comparison; calibration sample; documented limitation |
| **Judge and generator are the same model** | **Self-preference bias inflates faithfulness** | Blind the judge to variant, randomise presentation order; human calibration sample carries the rest. A 14B judge does not fit beside an 8B generator on one T4 (§15) |
| **Chunking ablation grid costs 8-16 GPU-hours** | **More than half the weekly quota for one phase** | Content-addressed embedding cache across chunker settings; run the grid on a corpus subset (§15, §21) |
| Scope creep | Nothing finishes | Each phase has a measurable gate; do not start a phase until the current gate passes |
| Two GPUs tempts sharding a big model | GPU 1 wasted | Deliberate workload split instead |
| **Quick Tunnel buffers `GET` SSE (#1449)** | Answer appears frozen, then dumps at once | **POST + `fetch` + `ReadableStream` (§19.3).** Prove it in Phase 0, not Phase 7. Framing stays standard, so switching the method restores `EventSource`. Treat as an observation, not a law — a named tunnel removes it entirely. |
| Intermediary idle timeout kills a long stream | Connection drops mid-answer | SSE comment heartbeat every 15s (§19.3) |
| Chunk offsets drift after edits | Citations point at the wrong text | Two coordinate spaces with an explicit map (§9); `parser_version` / `cleaner_version` bumps force re-ingest |
| **Clean and original offsets silently conflated** | **Citation verification runs against text the model never saw; recall@k undefined** | Store both offsets per chunk, verify in clean space, display in original space (§9, §14); gold spans are clean-space (§15) |
| **Reranker truncates chunks over 512 tokens** | **The highest-ROI component scores half the chunk, silently** | Window to `rerank_max_len` with a query-aware head (§10); record `rerank_window` per ablation row |
| **Chunker/embedder changed while appending** | **Heterogeneous index; §6.4 invariant still passes, recall degrades with no error** | `POST /index` compares against the manifest and returns 409 on mismatch (§6.5) |
| Deleting a document corrupts row/line alignment | Silent corruption of every retrieval metric | Deletion is logical, not physical: append-only rows plus a live mask (§6.4) |
| **Loopback check mistaken for a `/metrics` control** | **Provides no protection — tunnelled requests arrive from 127.0.0.1** | Require auth on every route except static, `/healthz`, `/api/auth` (§16) |
| Unknown model tag or digest | Silent swap to the wrong weights | Assert exact model identity; record revision in every eval run |
| CLI push lands on P100 (sm_60) | Dead kernels, no error | `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` (§19.2) |
| Corpus exceeds 250k chunks | Exact search stops being free | Track chunk count; the ceiling is known in advance rather than hit accidentally |
| ColBERT enabled opportunistically | 1MB per chunk; blows the 20GB store instantly | Declared infeasible at this scale, not "deferred" (§5.4) |
| **Auto-save does not complete, or the session is hard-killed** | **A built index is gone, with quota already spent** | Write-through on every mutation (§6.2); `export_index.py` to a private dataset; never rely on session-end save |
| Many sessions fill the 20GB store | Ingest fails partway through | Per-session and total byte counts surfaced in the UI; store is rebuildable from `files/` plus manifest |
| One session's data reachable from another | Silent, permanent contamination | Session id is a URL path segment on every corpus and index route; cross-session isolation test |
| Concurrent queries contend for GPU 1 | Every request slows down, including the first | Semaphore around embed and rerank; batch rerank across requests; dense search on CPU below `dense_gpu_threshold` |
| Cross-request rerank batching misattributes scores | Wrong candidates survive, wrong ones are dropped | Batch-association test in §22 |
| BM25 rebuild stalls the first query after a session switch | Reads as a hang | Measure BM25 build time in Phase 0; open sessions outside the query path; semaphore covers index-open (§6.7) |
| Ollama `keep_alive` expires between queries | First request after an idle gap pays a multi-second reload | `keep_alive` set in settings |
| Vector row and chunk line drift apart | **Silent corruption of every retrieval metric** | Invariant asserted after every append (§6.4); no append path bypasses the writer |
| Gate latency hidden inside `retrieve_ms` | Multi-second unexplained pause looks like a hang | `gate_ms` is its own stage and its own §17.3 readout row |
| Phone backgrounds mid-stream | Partial answer appears lost | Keep the partial answer, reconnect with `Last-Event-ID`, replay from the ring buffer |
| Long session title breaks the mobile bar | Layout unusable | Truncate titles in the shell; never let the bar wrap |

---

## 24. Open questions

**Closed.** Streaming was open in an earlier draft. It is now decided: `POST` +
`fetch` + `ReadableStream`, with the rationale in §19.3, proven in Phase 0 rather than Phase 7,
and reversible by switching the method. Buffer-then-emit was considered and rejected — it gives
up the stage readout that §17.5 calls the product's honesty mechanism, and the citation-arrives-
after-the-text problem is handled by buffering the *server-side* token stream per stream id, not
by withholding it from the client.

1. **Corpus and sessions.** Which documents, roughly how many, and how many sessions do you
   expect to keep side by side? This is the one blocking question for Phase 1. It decides which
   loaders matter, and scale decides whether the 20GB ceiling and the exact-search choice still
   hold. Above roughly 250k chunks per session, revisit exact search and re-baseline everything.
2. **Auth.** This plan assumes a single operator with a runtime token, which fits a keyless
   notebook reached over a temporary tunnel, and sessions are therefore *workspaces*, not
   *security boundaries* — one token sees every session. Confirm, or specify a real identity
   model if sessions ever need to be private to different people.
3. **Multilingual.** `bge-m3` is multilingual, which matters if any corpus text is in Yoruba or
   Hausa. Worth an explicit eval subset rather than discovering a gap later.
4. **Corpus composition.** §24 Q1 asks how many documents, but the *shape* matters as much as
   the count: how many are PDFs with tables or two-column layouts, and how much is tabular?
   Table-heavy corpora need a different retrieval design (§5.6) and would invalidate the
   chunking grid before it starts.