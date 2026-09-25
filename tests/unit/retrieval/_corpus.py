"""Labelled multi-company corpus used by retrieval unit tests and the RAG quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from client_research_agent.models import Chunk, ChunkStrategy, DocumentType
from client_research_agent.retrieval.enrichment import MetadataEnricher

_COMPANIES = {
    "acme": ("Acme Corp", "acme.example.com"),
    "globex": ("Globex Industries", "globex.example.com"),
    "initech": ("Initech Holdings", "initech.example.com"),
}

_TOPIC_META = {
    "cloud": (DocumentType.PRESS_RELEASE, "Technology update"),
    "ai": (DocumentType.PRESS_RELEASE, "Innovation news"),
    "earn": (DocumentType.EARNINGS_RELEASE, "Quarterly results"),
    "lead": (DocumentType.LEADERSHIP_ANNOUNCEMENT, "Leadership announcement"),
    "data": (DocumentType.INVESTOR_RELATIONS, "Investor day remarks"),
    "sec": (DocumentType.SEC_FILING, "Form 8-K risk disclosure"),
}

RAW_CORPUS: dict[str, str] = {
    "acme-cloud-1": (
        "Acme Corp completed the migration of 70 percent of its enterprise workloads to "
        "Microsoft Azure in fiscal 2026 and closed two on-premises data centers in Ohio."
    ),
    "acme-cloud-2": (
        "The cloud migration program, led by CIO Maria Chen, targets a full exit from "
        "legacy data centers by 2027 and is expected to reduce infrastructure costs by "
        "$40 million annually."
    ),
    "acme-ai-1": (
        "Acme launched an AI initiative to deploy machine learning models for predictive "
        "maintenance across 14 factories, using sensor data from its production lines."
    ),
    "acme-ai-2": (
        "Acme's generative AI pilot gives service engineers a copilot assistant trained "
        "on repair manuals; the company plans to scale the assistant to 3,000 "
        "technicians."
    ),
    "acme-earn-1": (
        "Acme Corp reported fourth quarter revenue of $4.2 billion, up 8 percent year "
        "over year, with operating margin expanding to 14.1 percent."
    ),
    "acme-earn-2": (
        "For fiscal 2027, Acme guided revenue growth of 5 to 7 percent and raised capital"
        " expenditure to $600 million to fund automation and digital investments."
    ),
    "acme-lead-1": (
        "Acme appointed Maria Chen as Chief Information Officer; she previously led "
        "digital transformation at a global logistics firm."
    ),
    "acme-lead-2": (
        "Chief Executive Officer Robert Klein said the board approved a three-year "
        "technology modernization plan focused on data and automation."
    ),
    "acme-data-1": (
        "Acme is consolidating plant data into a lakehouse data platform to unify "
        "analytics, quality reporting and supply chain visibility."
    ),
    "acme-sec-1": (
        "Acme expanded its cybersecurity program after a ransomware incident at a "
        "supplier, adopting zero trust network access for all remote users."
    ),
    "globex-cloud-1": (
        "Globex Industries signed a five-year agreement with Amazon Web Services (AWS) to"
        " move its e-commerce platform and inventory systems to the cloud."
    ),
    "globex-cloud-2": (
        "Globex said the cloud migration of its point-of-sale systems across 1,200 stores"
        " will finish in the second half of 2026."
    ),
    "globex-ai-1": (
        "Globex uses artificial intelligence for demand forecasting, cutting out-of-stock"
        " rates by 12 percent in pilot regions."
    ),
    "globex-ai-2": (
        "The retailer introduced a generative AI shopping assistant on its mobile app and"
        " will expand personalization models across marketing."
    ),
    "globex-earn-1": (
        "Globex Industries posted quarterly net sales of $9.8 billion, with comparable "
        "store sales up 3.4 percent and digital sales up 21 percent."
    ),
    "globex-earn-2": (
        "Globex lowered full-year earnings guidance, citing higher freight costs and "
        "pressure on discretionary spending."
    ),
    "globex-lead-1": (
        "Globex named Priya Natarajan as Chief Technology Officer to lead its "
        "engineering, data and cloud organizations."
    ),
    "globex-lead-2": (
        "Chief Executive Officer Daniel Ortiz will retire next year; the board has begun "
        "a search for his successor."
    ),
    "globex-data-1": (
        "Globex is building a customer data platform that combines loyalty, online and "
        "store transactions for real-time analytics."
    ),
    "globex-sec-1": (
        "Globex reported a payment card security incident affecting a limited number of "
        "stores and hired forensic investigators."
    ),
    "initech-cloud-1": (
        "Initech Holdings is migrating its policy administration mainframe to Google "
        "Cloud as part of a multi-year core modernization program."
    ),
    "initech-cloud-2": (
        "Initech expects the core systems cloud migration to retire 40 legacy "
        "applications and cut run costs by 18 percent."
    ),
    "initech-ai-1": (
        "Initech deployed machine learning claims triage that routes simple auto claims "
        "to straight-through processing within minutes."
    ),
    "initech-ai-2": (
        "Initech's AI governance council approved responsible AI guidelines before "
        "scaling large language models in underwriting."
    ),
    "initech-earn-1": (
        "Initech Holdings reported gross written premiums of $3.1 billion for the quarter"
        " and a combined ratio of 94.2 percent."
    ),
    "initech-earn-2": (
        "Initech's investment income rose 11 percent on higher yields, while catastrophe "
        "losses reduced quarterly earnings."
    ),
    "initech-lead-1": (
        "Initech appointed Samuel Okafor as Chief Data Officer, a new role reporting to the chief executive."
    ),
    "initech-lead-2": (
        "Chief Executive Officer Laura Benton outlined a strategy to grow commercial "
        "lines and invest in digital distribution."
    ),
    "initech-data-1": (
        "Initech is standardizing actuarial and claims data on a cloud data warehouse to "
        "speed up pricing analytics."
    ),
    "initech-sec-1": (
        "Initech strengthened cybersecurity controls, including multi-factor "
        "authentication for brokers and a new security operations center."
    ),
}


@dataclass(frozen=True)
class LabelledQuery:
    query: str
    company: str
    relevant: frozenset[str]


QUERIES: tuple[LabelledQuery, ...] = (
    LabelledQuery(
        "Acme cloud migration and data center exit", "Acme Corp", frozenset({"acme-cloud-1", "acme-cloud-2"})
    ),
    LabelledQuery(
        "What AI and machine learning initiatives is Acme pursuing?",
        "Acme Corp",
        frozenset({"acme-ai-1", "acme-ai-2"}),
    ),
    LabelledQuery(
        "Acme quarterly revenue and earnings guidance", "Acme Corp", frozenset({"acme-earn-1", "acme-earn-2"})
    ),
    LabelledQuery("Who is Acme's new Chief Information Officer?", "Acme Corp", frozenset({"acme-lead-1"})),
    LabelledQuery("Acme lakehouse data platform for analytics", "Acme Corp", frozenset({"acme-data-1"})),
    LabelledQuery(
        "Globex move to AWS cloud", "Globex Industries", frozenset({"globex-cloud-1", "globex-cloud-2"})
    ),
    LabelledQuery(
        "Globex generative AI and artificial intelligence use cases",
        "Globex Industries",
        frozenset({"globex-ai-1", "globex-ai-2"}),
    ),
    LabelledQuery(
        "Globex chief executive officer succession and technology leadership",
        "Globex Industries",
        frozenset({"globex-lead-1", "globex-lead-2"}),
    ),
    LabelledQuery("Globex quarterly net sales results", "Globex Industries", frozenset({"globex-earn-1"})),
    LabelledQuery(
        "Initech mainframe modernization and cloud migration",
        "Initech Holdings",
        frozenset({"initech-cloud-1", "initech-cloud-2"}),
    ),
    LabelledQuery(
        "Initech machine learning in claims and AI in underwriting",
        "Initech Holdings",
        frozenset({"initech-ai-1", "initech-ai-2"}),
    ),
    LabelledQuery("Initech cybersecurity controls", "Initech Holdings", frozenset({"initech-sec-1"})),
    # Paraphrased questions with little lexical overlap with the evidence.
    LabelledQuery(
        "Is Acme shrinking its on-premises infrastructure footprint?",
        "Acme Corp",
        frozenset({"acme-cloud-1", "acme-cloud-2"}),
    ),
    LabelledQuery(
        "Which Globex executive is stepping down?", "Globex Industries", frozenset({"globex-lead-2"})
    ),
    LabelledQuery(
        "How is Initech using large language models responsibly?",
        "Initech Holdings",
        frozenset({"initech-ai-2"}),
    ),
    LabelledQuery("Globex payment breach at stores", "Globex Industries", frozenset({"globex-sec-1"})),
)


def corpus_chunks(*, enrich: bool = True) -> list[Chunk]:
    enricher = MetadataEnricher()
    chunks: list[Chunk] = []
    for index, (chunk_id, text) in enumerate(RAW_CORPUS.items()):
        company_key, topic = chunk_id.split("-")[:2]
        company, domain = _COMPANIES[company_key]
        document_type, title = _TOPIC_META[topic]
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=text,
            company=company,
            url=f"https://{domain}/news/{chunk_id}",
            title=f"{company} {title}",
            document_type=document_type,
            source_domain=domain,
            chunk_index=index,
            strategy=ChunkStrategy.CHILD,
            publication_date=date(2026, 1 + index % 9, 1 + index % 27),
            confidence=0.8,
            token_count=len(text.split()),
        )
        chunks.append(enricher.enrich_chunk(chunk) if enrich else chunk)
    return chunks
