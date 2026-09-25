"""Retrieval poisoning defences (OWASP LLM03 supply chain, LLM08 vector & embedding weaknesses).

Every chunk that could reach the vector index or a prompt is checked for:

* embedded prompt injection (quarantine when blocked, flag when suspicious);
* low source trust;
* a source domain outside the allow-list (company domain + configured suffixes);
* near-identical content published on several *different* domains, the
  signature of SEO spam / coordinated content farms;
* keyword stuffing (one term dominating the token distribution);
* claims about the researched company made by untrusted off-domain sources.

The guard returns a decision per chunk. ``QUARANTINE`` chunks must not be
indexed or cited; ``FLAG`` chunks may be used but lower their evidence weight
and surface in the brief's warnings.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from client_research_agent.models import Chunk
from client_research_agent.security.prompt_injection import PromptInjectionDetector, combine_weights


class ChunkAction(StrEnum):
    ACCEPT = "accept"
    FLAG = "flag"
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class ChunkDecision:
    chunk_id: str
    action: ChunkAction
    risk: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PoisoningReport:
    decisions: tuple[ChunkDecision, ...]

    def _ids(self, action: ChunkAction) -> frozenset[str]:
        return frozenset(d.chunk_id for d in self.decisions if d.action is action)

    @property
    def quarantined_ids(self) -> frozenset[str]:
        return self._ids(ChunkAction.QUARANTINE)

    @property
    def flagged_ids(self) -> frozenset[str]:
        return self._ids(ChunkAction.FLAG)

    @property
    def accepted_ids(self) -> frozenset[str]:
        return frozenset(d.chunk_id for d in self.decisions if d.action is not ChunkAction.QUARANTINE)

    def decision_for(self, chunk_id: str) -> ChunkDecision | None:
        return next((d for d in self.decisions if d.chunk_id == chunk_id), None)

    def filter(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        blocked = self.quarantined_ids
        return [c for c in chunks if c.chunk_id not in blocked]


@dataclass(frozen=True, slots=True)
class PoisoningPolicy:
    quarantine_trust_below: float = 0.2
    flag_trust_below: float = 0.4
    claim_trust_threshold: float = 0.6
    enforce_allow_list: bool = False
    duplicate_similarity: float = 0.85
    spam_domain_count: int = 3
    shingle_size: int = 5
    stuffing_flag_ratio: float = 0.08
    stuffing_quarantine_ratio: float = 0.15
    stuffing_min_tokens: int = 40
    allowed_domain_suffixes: tuple[str, ...] = field(default=("sec.gov",))


_WEIGHTS: Mapping[str, float] = {
    "prompt_injection": 1.0,
    "suspected_injection": 0.35,
    "untrusted_source": 0.9,
    "low_trust_source": 0.3,
    "off_allow_list": 0.25,
    "cross_domain_duplicate": 0.3,
    "seo_spam_cluster": 0.8,
    "keyword_stuffing": 0.3,
    "keyword_stuffing_severe": 0.8,
    "off_domain_company_claim": 0.3,
}
_QUARANTINE_REASONS = frozenset(
    {"prompt_injection", "untrusted_source", "seo_spam_cluster", "keyword_stuffing_severe", "blocked_domain"}
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'&-]*", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "is",
        "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "we", "our", "us",
        "they", "their", "he", "she", "his", "her", "will", "would", "can", "could", "has", "have", "had",
        "not", "but", "also", "more", "than", "into", "over", "about", "which", "who", "whom", "whose", "all",
        "any", "each", "new", "per",
    }
)  # fmt: skip
_CLAIM_RE = re.compile(
    r"\b(?:announc(?:ed|es|ing)|acquir(?:ed|es|ing)|to\s+acquire|merg(?:ed|er|ing)|partner(?:ed|ship|s)\s+with|"
    r"signed|awarded|selected|launch(?:ed|es)|revenue|earnings|profit|loss(?:es)?|layoffs?|bankrupt(?:cy)?|"
    r"lawsuit|sued|investigation|breach|fined|ceo|cfo|appointed|resign(?:ed|s)|will\s+(?:invest|spend|migrate|"
    r"adopt|replace)|is\s+(?:the\s+)?(?:leader|leading|largest|best|only))\b",
    re.IGNORECASE,
)


def domain_matches(domain: str, suffixes: Iterable[str]) -> bool:
    host = domain.lower().strip(".").removeprefix("www.")
    for suffix in suffixes:
        cleaned = suffix.lower().strip(".").removeprefix("www.")
        if cleaned and (host == cleaned or host.endswith("." + cleaned)):
            return True
    return False


def shingles(text: str, size: int) -> frozenset[tuple[str, ...]]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class RetrievalPoisoningGuard:
    def __init__(
        self,
        detector: PromptInjectionDetector | None = None,
        policy: PoisoningPolicy | None = None,
        *,
        blocked_domains: Iterable[str] = (),
    ) -> None:
        self._detector = detector or PromptInjectionDetector()
        self._policy = policy or PoisoningPolicy()
        self._blocked = tuple(blocked_domains)

    @property
    def policy(self) -> PoisoningPolicy:
        return self._policy

    def evaluate(
        self,
        chunks: Sequence[Chunk],
        *,
        company_domain: str | None = None,
        trust_scores: Mapping[str, float] | None = None,
    ) -> PoisoningReport:
        reasons: dict[str, list[str]] = {c.chunk_id: [] for c in chunks}
        for chunk in chunks:
            reasons[chunk.chunk_id].extend(self._chunk_reasons(chunk, company_domain, trust_scores or {}))
        for chunk_id, reason in self._duplicate_reasons(chunks, company_domain):
            if reason not in reasons[chunk_id]:
                reasons[chunk_id].append(reason)

        decisions = []
        for chunk in chunks:
            found = tuple(dict.fromkeys(reasons[chunk.chunk_id]))
            risk = (
                1.0 if "blocked_domain" in found else combine_weights([_WEIGHTS.get(r, 0.0) for r in found])
            )
            if any(r in _QUARANTINE_REASONS for r in found):
                action = ChunkAction.QUARANTINE
            elif found:
                action = ChunkAction.FLAG
            else:
                action = ChunkAction.ACCEPT
            decisions.append(ChunkDecision(chunk.chunk_id, action, risk, found))
        return PoisoningReport(tuple(decisions))

    # -- per-chunk checks ------------------------------------------------------------------------------
    def _trust(self, chunk: Chunk, trust_scores: Mapping[str, float]) -> float:
        if chunk.source_domain in trust_scores:
            return float(trust_scores[chunk.source_domain])
        value = chunk.metadata.get("trust_score")
        if isinstance(value, int | float):
            return float(value)
        return chunk.confidence

    def _chunk_reasons(
        self, chunk: Chunk, company_domain: str | None, trust_scores: Mapping[str, float]
    ) -> list[str]:
        policy = self._policy
        found: list[str] = []
        if self._blocked and domain_matches(chunk.source_domain, self._blocked):
            found.append("blocked_domain")

        assessment = self._detector.assess(chunk.text)
        if assessment.blocked:
            found.append("prompt_injection")
        elif assessment.score >= self._detector.threshold / 2:
            found.append("suspected_injection")

        trust = self._trust(chunk, trust_scores)
        if trust < policy.quarantine_trust_below:
            found.append("untrusted_source")
        elif trust < policy.flag_trust_below:
            found.append("low_trust_source")

        first_party = bool(company_domain) and domain_matches(chunk.source_domain, [company_domain or ""])
        allowed = first_party or domain_matches(chunk.source_domain, policy.allowed_domain_suffixes)
        if not allowed:
            found.append("blocked_domain" if policy.enforce_allow_list else "off_allow_list")

        found.extend(self._stuffing(chunk))

        if (
            not first_party
            and trust < policy.claim_trust_threshold
            and chunk.company.lower() in chunk.text.lower()
            and _CLAIM_RE.search(chunk.text)
        ):
            found.append("off_domain_company_claim")
        return found

    def _stuffing(self, chunk: Chunk) -> list[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(chunk.text)]
        if len(tokens) < self._policy.stuffing_min_tokens:
            return []
        content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
        if not content:
            return []
        _, top_count = Counter(content).most_common(1)[0]
        ratio = top_count / len(tokens)
        company_tokens = [t.lower() for t in _TOKEN_RE.findall(chunk.company)]
        if company_tokens:
            company_hits = sum(
                1
                for i in range(len(tokens) - len(company_tokens) + 1)
                if tokens[i : i + len(company_tokens)] == company_tokens
            )
            ratio = max(ratio, company_hits * len(company_tokens) / len(tokens))
        if ratio >= self._policy.stuffing_quarantine_ratio:
            return ["keyword_stuffing_severe"]
        if ratio >= self._policy.stuffing_flag_ratio:
            return ["keyword_stuffing"]
        return []

    # -- cross-chunk checks ----------------------------------------------------------------------------
    def _duplicate_reasons(
        self, chunks: Sequence[Chunk], company_domain: str | None
    ) -> list[tuple[str, str]]:
        policy = self._policy
        prints = [shingles(c.text, policy.shingle_size) for c in chunks]
        neighbours: dict[int, set[int]] = {i: set() for i in range(len(chunks))}
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if chunks[i].source_domain.lower() == chunks[j].source_domain.lower():
                    continue
                if jaccard(prints[i], prints[j]) >= policy.duplicate_similarity:
                    neighbours[i].add(j)
                    neighbours[j].add(i)

        def is_first_party(position: int) -> bool:
            return bool(company_domain) and domain_matches(
                chunks[position].source_domain, [company_domain or ""]
            )

        results: list[tuple[str, str]] = []
        for index, linked in neighbours.items():
            if not linked:
                continue
            chunk = chunks[index]
            domains = {chunk.source_domain.lower()} | {chunks[j].source_domain.lower() for j in linked}
            # A copy of first-party content on wire services is syndication, not a content farm.
            syndicated = is_first_party(index) or any(is_first_party(j) for j in linked)
            if len(domains) >= policy.spam_domain_count and not syndicated:
                results.append((chunk.chunk_id, "seo_spam_cluster"))
            else:
                results.append((chunk.chunk_id, "cross_domain_duplicate"))
        return results
