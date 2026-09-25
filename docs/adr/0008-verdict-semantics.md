# ADR-0008: Verdict semantics - three verdicts, low fit reported as NOT_ENOUGH_EVIDENCE with reason LOW_FIT

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

The product contract defines exactly three verdicts (`models/domain.py::FitVerdict`):
`GOOD_FIT`, `POTENTIAL_FIT`, `NOT_ENOUGH_EVIDENCE`. There is no "poor fit"
verdict. The scoring engine must decide what to report when the evidence is
sufficient to assess but the weighted score is low.

The two plausible options were:

1. Report `POTENTIAL_FIT` for any well-evidenced account below the good-fit
   threshold.
2. Report `NOT_ENOUGH_EVIDENCE` and make clear that it means "not enough evidence
   *of fit*", with a machine-readable reason distinguishing it from missing
   evidence.

For an account team the costly error is a false positive: an overstated account
enters the pipeline and consumes seller time.

## Decision

`scoring/engine.py::ScoringEngine._decide` evaluates, in order:

| Order | Condition | Verdict | `VerdictReason` |
|---|---|---|---|
| 1 | two or more criteria cite no evidence (`MISSING_EVIDENCE_LIMIT = 2`) | `NOT_ENOUGH_EVIDENCE` | `MISSING_EVIDENCE` |
| 2 | overall confidence `< scoring.min_confidence` (0.45) | `NOT_ENOUGH_EVIDENCE` | `LOW_CONFIDENCE` |
| 3 | weighted score `>= scoring.good_fit_threshold` (3.5) | `GOOD_FIT` | `GOOD_FIT` |
| 4 | weighted score `>= scoring.potential_fit_threshold` (2.25) | `POTENTIAL_FIT` | `POTENTIAL_FIT` |
| 5 | otherwise | `NOT_ENOUGH_EVIDENCE` | `LOW_FIT` |

For case 5, `verdict_rationale` begins "Evidence indicates low fit: the evidence
is sufficient to assess, but the score is below the ... potential-fit threshold",
so humans can distinguish it from cases 1 and 2. `ScoreBreakdown.reason` carries
the machine-readable code for downstream consumers.

Supporting rules in the same engine:

- The weighted score is `sum(score_c * weight_c)` using `ScoringSettings.weights`
  (authoritative; a stale weight on a `CriterionScore` is replaced).
- Overall confidence is the weight-averaged criterion confidence, where a
  criterion with fewer than `scoring.min_evidence_per_criterion` evidence items
  contributes a quarter of its confidence, and the average is multiplied by
  `1 - 0.1 * n_under_evidenced`.
- `ScoringEngine.sensitivity` recomputes the verdict with each criterion moved by
  -1 and +1, reporting which single judgement could flip the outcome.

## Consequences

- Positive: the engine never overstates an account; a low score and missing
  evidence both keep the account out of the "fit" tiers.
- Positive: `AccountRanker` orders by verdict tier first, then a
  confidence-adjusted lower bound, so a speculative high score cannot jump a
  well-evidenced one.
- Negative: `NOT_ENOUGH_EVIDENCE` is overloaded. Any consumer that must separate
  "low fit" from "unknown" has to read `ScoreBreakdown.reason` or the rationale
  prefix; the `briefs` table stores only `verdict`, so the
  `not_enough_evidence_share` alert cannot distinguish the two causes without
  parsing `brief_json`.
- Revisit if the product contract adds a fourth verdict.
