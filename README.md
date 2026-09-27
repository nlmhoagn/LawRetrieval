# DSC 2026 — Vietnamese Legal AI System

Comprehensive AI solution for the **DSC 2026 (UIT Data Science Challenge)**, covering both **Task 1 (Legal Information Retrieval)** and **Task 2 (Legal Question Answering)**.

---

# Task 1: Legal Information Retrieval (LegalIR)

Vietnamese Legal Document Retrieval System. Given a query, return the **top 5 documents** most likely to contain the answer.

**Public Leaderboard: Recall@5 = 0.8637**

---

## Quick Start (Task 1)

```bash
pip install -r requirements.txt

python run_all.py --data "LegalIR - Public Test"
```

A single command produces a validated `submission.zip` ready for submission.

The pipeline consists of 5 stages, **automatically skipping any stage whose output already exists** — safe to re-run and resume interrupted runs without starting from scratch:

```
1. chunk corpus          1m31s   →  chunks.jsonl     592 MB
2. index BM25             202s   →  index/           299 MB
3. extract embeddings    36 min  →  emb/             714 MB   (GPU required)
4. generate submission     74s   →  submission.zip    21 KB
5. validate submission
```

Without a GPU, step 3 is very slow. Skip it with `--skip-encode` to run pure BM25 (fast, but loses ~0.05 Recall).

### Data

This repository does **not** include competition data. Unpack the dataset into the root directory:

```
LegalIR - Public Test/
  selected-contexts/            8,532 context_*.json files
  public-official.json          1,000 queries to answer
  train.json                    7,000 queries + answers (optional)
```

### Options

```bash
python run_all.py --data "..." --alpha 0.7 --k1 2.5 --b 0.9   # currently used parameters
python run_all.py --data "..." --skip-encode                  # BM25 only
python run_all.py --data "..." --force                        # re-run from scratch
```

---

## Pipeline (Task 1)

```
8,532 legal documents (353M characters, median 4,813 words/doc)
        │
        ▼  3-tier chunker based on legal document structure
487,194 chunks (median 165 words, max 407)
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼                         │
   BM25 (sparse)         halong_embedding (dense)        │
   inverted index        487k × 768 fp16                 │
   on disk, 299 MB       on disk, 714 MB                 │
        │                      │                         │
        │  top-1,000 chunks    │  cosine on pool         │
        └──────────┬───────────┘                         │
                   ▼                                     │
       linear score fusion: 0.7 × dense + 0.3 × BM25     │
                   ▼                                     │
       aggregate chunks → document: mean of top 3 chunks │
                   ▼                                     │
              top-5 documents ◄──────────────────────────┘
```

| Component | Recall@5 |
|---|---|
| Pure BM25 | 0.8112 |
| Pure Dense | 0.8307 |
| **Hybrid (current)** | **0.8633** |

*Evaluated on a separate 1,000-query validation split from `train.json`. The 0.8633 score matches almost exactly with the public leaderboard score of 0.8637 — confirming this validation set is a reliable proxy.*

### 1. 3-Tier Chunking

The raw documents are far too long for standard encoders — **99.6% exceed 512 tokens**, with extreme outliers reaching 1.24 million words. However, chunking is not just about fitting within token limits:

```
Keyword match rate in GROUND-TRUTH document : 0.944
Keyword match rate in RANDOM document       : 0.683
```

A randomly chosen document already contains 68% of the query keywords simply because of its length. Chunking is critical to **restore discriminative signal**.

| Tier | Condition | Chunk Count |
|---|---|---|
| 1 | Split by Article (`Điều N`) (present in 85.9% of documents) | 122,520 |
| 2 | Article > 350 words → split by Clause (`Khoản`); if still long → fixed sliding window | 333,416 |
| 3 | Documents without Articles (official dispatches, TCVN standards) → 256-word window, 25% overlap | 31,238 |
| 0 | Empty `passage` → salvage title from slug in `link` | 20 |

Tier 0 is not a minor detail: there are **20 documents with empty `passage`, 6 of which are gold answers**. Without this tier, 6 queries would be unconditionally lost.

Each chunk is prepended with the hierarchical header `Document Title > Article > Clause`, and the Article title is replicated across all sub-chunks — otherwise, subsequent chunks completely lose their context.

Result: **0 / 8,532 documents lost, 0 / 3,105 gold answers missing chunks.**

### 2. On-Disk BM25

The development environment had only ~0.5 GB of free RAM, insufficient to hold a 487k × 70k sparse matrix in memory. The index is stored on disk via `np.memmap` and constructed in two passes (pass 1 counts document frequencies to compute offsets; pass 2 writes postings directly into allocated slots). At query time, only the postings of query terms are read into memory → RAM consumption is virtually zero.

```
69,920 vocabulary · 48.3M postings · 299 MB index · 85 ms / query
```

### 3. Dense Embeddings — Reranker, Not Retriever

Model: [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding) (xlm-roberta-base, 278M parameters, 768 dimensions). Three critical properties verified directly from model config rather than assumed: **mean pooling**, **built-in L2 normalization**, and **no `query:` / `passage:` prefixes required** — an easy pitfall given its architectural resemblance to multilingual-e5 (which *mandates* prefixes).

Dense embeddings are not searched globally across all chunks, as ceiling analysis proves it unnecessary:

| Pool (chunks) | Gold within pool |
|---|---|
| 200 | 0.9767 |
| 1,000 | 0.9900 |
| 2,000 | 0.9967 |

BM25 already retrieves the relevant chunks; it just **ranks them suboptimally**. Thus, the dense stage only needs to read ~1,000 vectors from memmap (~3 MB/query) to rerank the candidates — eliminating the need for FAISS or loading the full 714 MB index into RAM.

The 350-word chunk limit aligns well with the model's 512-token limit: median token length is 209, with **only 1.0% truncated**.

### 4. Score Fusion

BM25 scores (typically 16–40, unbounded) and cosine similarities (0.07–0.58) cannot be directly summed. We apply min-max normalization within the candidate pool before linear combination:

```
score = 0.7 × dense_norm + 0.3 × bm25_norm
```

Reciprocal Rank Fusion (RRF, rank-based) lags behind linear fusion (**0.8375 vs 0.8528**) because RRF discards *margin of relevance* information — a landslide top match is fundamentally different from a marginal edge.

### 5. Chunk-to-Document Aggregation

Labels are at the document level, whereas scoring occurs at the chunk level:

| Aggregation Method | Recall@5 |
|---|---|
| **top-3 mean** | **0.8250** |
| max | 0.7950 |
| avg over all chunks | 0.4175 |
| top-3 / log(chunk_count) | 0.1706 |

`avg` fails dramatically because document `68843`, for instance, comprises 6,514 chunks — a query matches only 2–3 chunks, while thousands of non-matching chunks with near-zero scores drag the average down to zero.

---

## Project Structure (Task 1)

```
run_all.py                          runs end-to-end pipeline
requirements.txt
Chunking-LegalIR/
  chunker.py                        3-tier legal document chunker
Retrieval-LegalIR/
  bm25.py                           build / query on-disk inverted index
  encode.py                         extract halong_embedding vectors → memmap
  hybrid.py                         hybrid BM25 + dense score fusion
  predict.py                        generates submission.json + .zip
  rerank.py                         cross-encoder reranker (optional, see below)
  serve.py                          web search UI for inspection
  eval_bm25.py, sweep_bm25.py       evaluation and grid sweep scripts
Validator-Task-LegalIR/
  validate_submission.py            validates submission zip, checks 14 error codes
RESULTS.md                          full experimental metrics and sample sizes
```

### Running Individual Components

```bash
cd Retrieval-LegalIR

# Manual query
python bm25.py query --index index "mức lương cơ sở là bao nhiêu" --topk 10
python hybrid.py --query "lệ phí làm căn cước công dân"

# Evaluation & sweeps
python hybrid.py --eval "../LegalIR - Public Test/train.json" -n 300
python sweep_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 200

# Web interface
python serve.py --index index --contexts "../LegalIR - Public Test/selected-contexts"
```

`serve.py` launches http://127.0.0.1:8000 — search, click on results to read the full document rendered by Chapter/Article/Clause, auto-scrolls to the matching Article, and highlights keywords. No external dependencies needed (uses stdlib's `http.server`).

### Validator

Catches invalid submissions before consuming daily submission quotas. Aggregates all errors and reports them in a single batch instead of halting on the first error. Detects duplicate `question_id`s — an issue silently overwritten by default `json.load`.

The most critical issue caught is **E010**: `context_*.json` stores `id` as integers, but submission answers require string IDs. Failing to cast these results in **0 points without any error notification** from the competition scoring system.

`run_all.py` automatically invokes the validator in Step 5.

---

## Data Insights & Findings (Task 1)

| Insight | Metric | Implication |
|---|---|---|
| Massive documents | median 4,813 words, max 1,242,409 | Chunking is strictly mandatory |
| Diluted lexical signal | 0.944 vs 0.683 (true vs random) | Full-document BM25 fails |
| Empty documents | 20 documents, **6 are gold** | Tier 0 fallback is critical |
| Missing `name` field | 1,125 / 8,532 documents | Must use `.get()`, never `d["name"]` |
| Skewed document popularity | 36.4% of corpus ever cited as gold; `245154` appears 109 times | 5,427 documents never cited in train may be answers in public/private test — do not prune |
| Abbreviations | 44/1,000 queries have abbreviations, Recall 0.9091 vs 0.8998 overall | Non-issue in practice — empirically verified |

Legal documents routinely define acronyms and abbreviations directly in Article 1 and subsequently use both forms, so chunks naturally contain both. Query-side abbreviation or synonym expansion is **not worth the complexity**: its theoretical ceiling is only +0.0040.

---

## Cross-Encoder — Experimented, Currently Benched

`rerank.py` implements an optional Stage 3 cross-encoder. Benchmark results:

| Configuration | Delta |
|---|---|
| `beta=1.0` (completely replaces hybrid ranking) | ~0.000 |
| `beta=0.4–0.6` (blended with hybrid scores) | **+0.030** |

Blending yields solid gains, whereas standalone replacement does not — the hybrid pipeline is already strong, and an off-the-shelf cross-encoder lacks domain adaptation.

Currently sidelined because an initial candidate cut of `ndocs=20` imposes a hard ceiling of **0.9177** at document level, so the +0.030 gain is offset by a -0.020 loss from pruned gold candidates. To effectively deploy it, `ndocs` must be widened to 50 (ceiling 0.9463), increasing inference time per submission from 74 seconds to ~21 minutes.

Requires `sentencepiece` and `protobuf` (see `requirements.txt`).

---

## Empirical Insights & Evaluation Lessons (Task 1)

On sample sizes of 200–500 queries, **score differences under ~0.01 are purely noise**. We narrowly avoided three suboptimal parameter choices caused by trusting single-run grid search peaks:

- `α=0.8` — peaked at 0.9061 on seed 777, but dropped to 0.8445 on seed 2024
- `k1/b` — required 2,000 queries + bootstrap resampling to reveal confidence intervals containing 0, meaning the competing configs were statistically indistinguishable
- Cross-encoder — `beta=1.0` yielded 0 improvement, while `beta=0.5` delivered +0.03

Finalized evaluation protocol: grid sweep on one sample split → **validate on a different seed/split** → rely strictly on **relative delta**, never absolute numbers. Ultimately, public leaderboard feedback supersedes all offline training estimates.

---

## Next Steps / Future Work (Task 1)

1. **Fine-tune `halong_embedding`** on 6,000 (query, ground-truth chunk) pairs extracted from `train.json` — all current components are zero-shot and have never trained on competition legal texts. This remains the highest-leverage opportunity.
2. **Query-side abbreviation dictionary** — low-cost, verifiable within 10 minutes.
3. **Word segmentation + bigrams for BM25** — Vietnamese multi-syllable compound words (e.g., 'mức lương cơ sở') are currently split into loose unigrams.
4. **Re-enable cross-encoder** with `ndocs=50`.

---

# Task 2: Legal Question Answering (LegalQA)

Vietnamese Long-Form Legal Question Answering System. Given a complex legal question, retrieve relevant legal statutes and generate an accurate, coherent answer grounded in official legislation with explicit legal citations.

**Evaluation Metrics: METEOR (Primary) & ROUGE-L (Secondary)**  
**Runtime Constraint: Total Parameters < 4B (Offline Kaggle T4)**

---

## Pipeline (Task 2)

```
Legal Query (median 19 words)
        │
        ▼  BM25 / Hybrid retrieval on 256-word chunks
Top-k Chunks (R@1: 55.4%, R@10: 89.7%, R@100: 97.8%)
        │
        ▼  "Small-to-Big" Evidence Expansion
Full Legal Article (`Điều`) containing the matched chunk
        │
        ▼  Citation-Gated Legal Graph Traversal
Referenced Statutes & Decrees (expanded ONLY if explicitly cited)
        │
        ▼  Context Assembly & Legal Grounding
System Prompt + Hierarchical Evidence (Doc > Article > Clause) + Query
        │
        ▼  Generator (< 4B Parameters, FP16 / 4-bit)
Ground-Truth Grounded Legal Answer with Legal Citations
```

| Pipeline Iteration | Indexing Unit | Recall@1 | Recall@10 | Recall@100 | Primary Failure Mode |
|---|---|---|---|---|---|
| **v9.2 (Legacy)** | Full Document | 22.3% | 50.9% | 76.3% | Severe BM25 length dilution (median 4,828 words/doc vs 19 words/query) |
| **v10 (Current)** | **256-word Chunk** | **55.4%** | **89.7%** | **97.8%** | Evidence context overflow handled by Article-level expansion |

### 1. "Small-to-Big" Retrieval & Evidence Expansion

Full legal documents average 4,828 words and can reach over 1.24 million words, drastically diluting keyword match signals against concise user queries.

- **Retrieval Unit (Small)**: Indexing operates on fine-grained **256-word chunks**, dramatically sharpening retrieval focus and boosting **Recall@1 from 22.3% to 55.4%** and **Recall@10 from 50.9% to 89.7%**.
- **Generation Unit (Big)**: Passing isolated 256-word fragments into the generator truncates statutory context and omits essential legal clauses. Once a chunk is retrieved, the system dynamically expands it to its **full parent Article (`Điều`)**, providing the LLM with complete, legally coherent evidence.

### 2. HTML Text Normalization & Article Recovery

Raw legal documents crawled from HTML contain substantial formatting corruption, especially soft line breaks (`\r\n\n`) within legislative headings:

- In raw data, **62.3% of Article titles were split mid-sentence**, causing regex parsers and segmenters to fail.
- Targeted text normalization and regex repairing reduced title fragmentation from **62.3% down to 0.2%**, ensuring clean article boundaries and reliable citation extraction.

### 3. Citation-Gated Legal Graph Traversal

Previous versions (v9.2) experimented with complex graph traversals with relation count gates, which failed because **93.1% of legal documents contain boilerplate "căn cứ" (pursuant to) phrases**, flooding the graph with noisy edges.

- v10 replaces noisy heuristics with **citation-gated expansion**: the graph traverses an edge *only* when the retrieved evidence explicitly cites a specific decree, law, or circular by official code/title.
- Redundant components like multi-candidate selectors, agentic tool-calling loops, and train-memory were eliminated in favor of deterministic evidence assembly, drastically cutting latency and preventing hallucinated citations.

### 4. Generation Under Parameter Budget (< 4B)

To meet the strict competition constraint of **offline execution on Kaggle T4 with < 4B total runtime parameters**:

- **Model Budget**: Employs an instruction-tuned LLM (< 4B parameters, e.g., Qwen2.5 / Gemma2 series) running in FP16 or 4-bit quantization.
- **Hierarchical Prompt Construction**: Evidence is structured strictly as `Văn bản > Điều > Khoản` to encourage verbatim quoting of legal clauses, directly maximizing n-gram precision and recall for **METEOR** and **ROUGE-L**.
- **Constrained Decoding**: Low-temperature sampling with repetition penalties suppresses conversational filler and enforces concise, formal legal Vietnamese.

---

## Data Insights & Findings (Task 2)

| Insight | Metric / Observation | Architectural Decision |
|---|---|---|
| Document Length Variance | Median 4,828 words/doc vs 19 words/query | Mandatory 256-word chunking for retrieval |
| Raw Crawl Soft Wraps | `\r\n\n` split 62.3% of Article headers | Pre-chunk regex normalization (< 0.2% errors) |
| Boilerplate Legal Phrases | 93.1% of documents contain "căn cứ" | Prune relation-count graphs; use strict citation gating |
| Agentic Loops on T4 | Excessive inference latency & memory overhead | Switch to deterministic small-to-big RAG pipeline |

---

## Empirical Insights & Architectural Lessons (Task 2)

1. **Retrieval Quality Caps Generation Ceiling**: In long-form LegalQA, generator hallucination is primarily driven by retrieval misses. Raising Recall@100 from 76.3% to 97.8% eliminated the vast majority of unsupported answers.
2. **Simplicity Over Agentic Complexity**: Multi-turn agentic tool calling and memory layers underperformed a well-tuned single-turn Small-to-Big RAG pipeline, while consuming 4× the inference time on resource-constrained Kaggle T4 GPUs.
3. **METEOR Metric Sensitivity**: METEOR rewards exact stem and synonym alignment. Preserving original legal phrasing rather than conversational paraphrasing yields consistently higher evaluation scores.

---

## Next Steps / Future Work (Task 2)

1. **Supervised Fine-Tuning (SFT) / QLoRA**: Fine-tune the < 4B generator directly on legal question-evidence-answer triplets extracted from training splits.
2. **DPO / KTO Alignment**: Align response style to match official competition ground truth answers (structured legal rationale followed by explicit conclusion).
3. **Dynamic Context Packing**: Implement token-aware packing to fit multi-article cross-references within the model's active context window without KV cache eviction.
