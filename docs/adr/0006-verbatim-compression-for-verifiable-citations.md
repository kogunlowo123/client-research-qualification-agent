# ADR-0006: Verbatim (extractive) context compression to keep citations verifiable

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

Context compression reduces prompt tokens and noise. The common approach is
abstractive: an LLM summarises each retrieved passage. For this product the
central promise is that every "Verified Fact" in the brief is backed by a quote
a reader can find in the public source. An abstractive summary breaks that
promise: the cited text no longer exists in the source, numbers can be
rewritten, and a summarising model can be steered by instructions embedded in
the passage.

## Decision

- `retrieval/compression.py::ContextCompressor` is extractive only. It scores
  sentences by query-term overlap (with a small bonus for sentences that carry
  digits and overlap the query), keeps the top `retrieval.compression_max_sentences`
  (default 6) sentences **verbatim and in original document order**, and drops
  the rest. When nothing relevant is found it keeps the leading sentences.
  Attached parent context is capped at twice that sentence budget.
- Chunkers operate on character spans of the source text and record
  `metadata["char_start"]` / `metadata["char_end"]`, so any chunk text is a
  substring of the stored document.
- Evidence quotes (`qualification/evidence.py::EvidenceRegistry`) are
  contiguous substrings of chunk text, capped at `MAX_QUOTE_CHARS = 500`.
- Citation validation compares claims with that verbatim evidence using lexical
  entailment with hard number and entity rules (`citations/entailment.py`), which
  is only meaningful because the evidence text is unmodified source text.

## Consequences

- Positive: every quote in the Evidence table is findable in the source; number
  checks compare against the numbers the source actually printed.
- Positive: compression is deterministic, free, and cannot be prompt-injected.
- Negative: extractive compression is less token-efficient than summarisation
  and can keep a sentence whose meaning depends on a dropped neighbour. Parent
  expansion (small-to-big) supplies the surrounding context to the LLM, while the
  child text remains the citation anchor.
- Negative: lexical sentence scoring misses sentences that are relevant only
  semantically. The upstream stages (dense retrieval, reranking, MMR) carry the
  semantic load; compression only trims within chunks already selected.
