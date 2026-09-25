# ADR-0005: A deterministic fallback for every LLM step

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

A brief touches the LLM many times: query rewriting and HyDE, multi-query
generation, CRAG relevance grading, reranking, Self-RAG critique, contextual
situating, five criterion assessments, opportunity analysis, narrative writing,
discovery questions and citation judging. With independent failure probability
per call, an all-or-nothing design turns a partial FMAPI outage into a total
product outage. LLM output is also untrusted: it can be malformed JSON, cite
evidence ids it was never shown, or disagree wildly with the evidence.

## Decision

Every LLM-dependent step has a deterministic implementation that runs when the
LLM is absent (`llm=None`), raises `TransientError` / `CircuitOpenError`, or
returns output that fails schema validation (`OutputValidationError` from
`services/structured.py::complete_structured`, which performs bounded
parse-repair turns first).

| Step | Module | Deterministic path | Metric on fallback |
|---|---|---|---|
| Query rewrite / HyDE | `retrieval/query_rewriting.py` | company-name variants + domain synonym table | `retrieval.rewrite.fallback`, `retrieval.hyde.fallback` |
| Multi-query | `retrieval/multi_query.py` | angle templates | `retrieval.multi_query.fallback` |
| CRAG grading | `retrieval/crag.py::RelevanceGrader` | lexical relevance | `retrieval.crag.grader_fallback` |
| Reranking | `retrieval/reranking.py` | `LexicalReranker` | `retrieval.rerank.llm_fallback` |
| Self-RAG critique | `retrieval/self_rag.py` | token-overlap critique | `retrieval.self_rag.fallback` |
| Contextual situating | `retrieval/enrichment.py` | deterministic contextual header | `enrichment.situating_fallback` |
| Semantic chunking | `retrieval/chunking.py` | recursive splitter | `chunking.semantic_fallback` |
| Criterion scoring | `qualification/agent.py` | `HeuristicQualifier` | `qualification_llm_fallback_total` |
| Opportunity analysis | `briefing/opportunity.py` | rules over scores and evidence | `opportunity_llm_fallback_total` |
| Narrative sections | `briefing/generator.py` | `DeterministicBriefBuilder` | `brief_writer_fallback_total` |
| Discovery questions | `briefing/generator.py` | criterion `discovery_question` templates | `discovery_questions_fallback_total` |
| Citation judge | `citations/validator.py` | lexical entailment only | `citation_judge_fallback_total` |

When the LLM does answer, the deterministic path still runs as a cross-check
where it matters: the qualification agent compares the LLM score with the
heuristic score and reduces confidence by 15% per level of disagreement beyond
one (`DISAGREEMENT_PENALTY_PER_LEVEL`), and discards an LLM score that cites no
evidence id it was shown (confidence capped at `UNGROUNDED_CONFIDENCE_CAP = 0.3`).

## Consequences

- Positive: a brief is always produced; degradations are explicit in
  `ClientBrief.warnings`, metrics and MLflow traces rather than silent.
- Positive: CI can exercise the whole pipeline with no model endpoint, and the
  heuristic gives a model-independent baseline for regression analysis.
- Negative: two implementations per step. The rubric, retrieval queries and
  signal lexicons for each criterion are kept together in
  `qualification/criteria.py::CriterionDefinition` so the heuristic cannot drift
  from the prompt rubric silently.
- Negative: a degraded brief can look complete to a reader. The renderer prints
  the `Warnings` section and the model versions; the operational guide sets an
  SLO on the fallback rate so sustained degradation pages someone.
