"""The five qualification criteria: rubric, retrieval queries and signal lexicons.

Each ``CriterionDefinition`` is the single source of truth for one criterion.
The rubric text is injected verbatim into the ``criterion_qualifier`` prompt,
the queries drive evidence retrieval, and the lexicons power the deterministic
``HeuristicQualifier`` used as fallback and cross-check. Keeping all three
together means a rubric change cannot silently drift from the heuristic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from client_research_agent.models import Criterion


@dataclass(frozen=True, slots=True)
class CriterionDefinition:
    criterion: Criterion
    title: str
    description: str
    rubric: Mapping[int, str]
    queries: tuple[str, ...]
    positive_signals: tuple[str, ...]
    negative_signals: tuple[str, ...]
    discovery_question: str

    def __post_init__(self) -> None:
        if sorted(self.rubric) != [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"{self.criterion}: rubric must define levels 0-5")
        if not self.queries or any("{company}" not in query for query in self.queries):
            raise ValueError(f"{self.criterion}: every retrieval query must contain '{{company}}'")

    def render_queries(self, company: str) -> list[str]:
        return [query.replace("{company}", company) for query in self.queries]

    def rubric_text(self) -> str:
        return "\n".join(f"{level} - {self.rubric[level]}" for level in range(6))

    def render_discovery_question(self, company: str) -> str:
        return self.discovery_question.replace("{company}", company)


_DEFINITIONS: tuple[CriterionDefinition, ...] = (
    CriterionDefinition(
        criterion=Criterion.COMPANY_SCALE,
        title="Company Size & Scale",
        description=(
            "Whether the organisation has the revenue, headcount, geographic footprint and operational "
            "complexity to sustain an enterprise-grade data and AI engagement."
        ),
        rubric={
            0: "No public evidence about size, revenue, headcount or footprint.",
            1: "Small business: revenue under $50M or fewer than 200 employees; single market.",
            2: "Lower mid-market: revenue $50M-$250M or 200-1,000 employees; limited geographic reach.",
            3: (
                "Mid-market to upper mid-market: revenue $250M-$1B or 1,000-10,000 employees; "
                "multiple markets."
            ),
            4: "Large enterprise: revenue $1B-$10B or 10,000-50,000 employees; multinational operations.",
            5: (
                "Global enterprise: revenue above $10B or more than 50,000 employees; complex "
                "multi-segment operations."
            ),
        },
        queries=(
            "{company} annual revenue net sales fiscal year results",
            "{company} number of employees workforce headcount",
            "{company} global operations countries customers segments",
        ),
        positive_signals=(
            "fortune 500",
            "fortune 100",
            "global",
            "worldwide",
            "multinational",
            "countries",
            "subsidiaries",
            "segments",
            "record revenue",
            "annual revenue",
            "net sales",
            "employees",
            "headquarters",
            "publicly traded",
            "nyse",
            "nasdaq",
            "acquisition",
            "market capitalization",
        ),
        negative_signals=(
            "startup",
            "seed round",
            "series a",
            "small business",
            "going concern",
            "delisted",
            "bankruptcy",
            "chapter 11",
            "divestiture",
        ),
        discovery_question=(
            "Which business units or regions at {company} would own a data and AI programme, and how is "
            "technology budget allocated across them?"
        ),
    ),
    CriterionDefinition(
        criterion=Criterion.TECH_MODERNIZATION,
        title="Technology Modernization",
        description=(
            "Evidence of active investment in modernising the technology estate: cloud migration, legacy "
            "replacement, ERP or core-system upgrades, platform engineering and digital transformation."
        ),
        rubric={
            0: "No public evidence about the technology estate or modernisation plans.",
            1: "Evidence points to a legacy estate with no stated modernisation agenda.",
            2: "General statements about digital ambition without named programmes or investment.",
            3: "At least one named modernisation initiative (e.g. cloud migration, ERP upgrade) under way.",
            4: "Multiple funded modernisation programmes with executive sponsorship or stated timelines.",
            5: (
                "Enterprise-wide transformation programme with disclosed investment, milestones and "
                "cloud-first mandate."
            ),
        },
        queries=(
            "{company} digital transformation cloud migration modernization program",
            "{company} legacy systems ERP upgrade technology investment",
            "{company} chief technology officer CIO technology strategy platform",
        ),
        positive_signals=(
            "digital transformation",
            "cloud migration",
            "cloud-first",
            "migrate to the cloud",
            "modernization",
            "modernisation",
            "modernize",
            "legacy systems",
            "erp",
            "s/4hana",
            "microservices",
            "kubernetes",
            "platform engineering",
            "devops",
            "api",
            "technology investment",
            "hybrid cloud",
            "aws",
            "azure",
            "google cloud",
            "saas",
            "automation",
            "chief digital officer",
            "chief technology officer",
            "chief information officer",
        ),
        negative_signals=(
            "technology freeze",
            "it budget cut",
            "paused the migration",
            "delayed implementation",
            "outage",
            "system failure",
            "write-off of software",
            "cancelled the program",
        ),
        discovery_question=(
            "Which legacy platforms at {company} are the biggest constraint on delivery speed "
            "today, and what "
            "is the timeline for retiring them?"
        ),
    ),
    CriterionDefinition(
        criterion=Criterion.AI_DATA_FOCUS,
        title="AI and Data Focus",
        description=(
            "How central data, analytics and AI are to the company's stated strategy: data platforms, "
            "machine learning in production, generative AI initiatives, data governance and AI leadership."
        ),
        rubric={
            0: "No public evidence about data, analytics or AI.",
            1: "Data or AI mentioned only in passing, with no initiative or owner.",
            2: "Stated interest in analytics or AI, but no named initiative, platform or leader.",
            3: "Named data or AI initiative, platform or pilot, or a dedicated data/AI leader.",
            4: "AI or data explicitly in corporate strategy with production use cases and investment.",
            5: (
                "AI and data are a core strategic pillar: disclosed investment, AI in production at "
                "scale, governance and leadership."
            ),
        },
        queries=(
            "{company} artificial intelligence machine learning generative AI initiative",
            "{company} data platform analytics data strategy governance",
            "{company} chief data officer chief AI officer AI investment",
        ),
        positive_signals=(
            "artificial intelligence",
            "machine learning",
            "generative ai",
            "genai",
            "large language model",
            "llm",
            "ai-powered",
            "ai strategy",
            "data platform",
            "data strategy",
            "lakehouse",
            "data lake",
            "data warehouse",
            "analytics",
            "data governance",
            "data science",
            "mlops",
            "chief data officer",
            "chief ai officer",
            "predictive",
            "copilot",
            "responsible ai",
            "databricks",
            "snowflake",
        ),
        negative_signals=(
            "ai moratorium",
            "banned the use of ai",
            "data breach",
            "regulatory fine",
            "paused ai",
            "data quality issues",
        ),
        discovery_question=(
            "Which AI or analytics use cases has {company} already moved into production, and what has "
            "blocked the others from scaling?"
        ),
    ),
    CriterionDefinition(
        criterion=Criterion.INDUSTRY_TRENDS,
        title="Industry Trends",
        description=(
            "Whether the company's industry is under pressure (regulation, competition, customer "
            "expectations, "
            "consolidation) that makes data and AI investment urgent, and whether the company is responding."
        ),
        rubric={
            0: "No public evidence about industry context or pressures.",
            1: "Stable industry with no stated pressures or response.",
            2: "Industry pressures mentioned in general terms without a company response.",
            3: (
                "Specific industry pressure (regulation, competition, disruption) acknowledged by "
                "the company."
            ),
            4: (
                "Company is publicly responding to industry shifts with new strategy, products or "
                "partnerships."
            ),
            5: (
                "Company is positioned as a leader in an industry transformation, with strategy "
                "tied to data and AI."
            ),
        },
        queries=(
            "{company} industry trends competition market disruption",
            "{company} regulatory changes compliance industry outlook",
            "{company} strategy response customer expectations partnerships",
        ),
        positive_signals=(
            "industry",
            "market share",
            "competitive",
            "competition",
            "disruption",
            "regulation",
            "regulatory",
            "compliance",
            "customer expectations",
            "consolidation",
            "partnership",
            "strategic partnership",
            "sustainability",
            "supply chain",
            "personalization",
            "growth strategy",
            "new market",
            "innovation",
        ),
        negative_signals=(
            "declining market",
            "loss of market share",
            "headwinds",
            "downturn",
            "recession",
            "uncertain outlook",
        ),
        discovery_question=(
            "Which industry shifts, such as regulation, new competitors or changing customer "
            "expectations, are "
            "most likely to change {company}'s priorities over the next 18 months?"
        ),
    ),
    CriterionDefinition(
        criterion=Criterion.NEAR_TERM_OPPORTUNITY,
        title="Near-Term Opportunity",
        description=(
            "Timing signals indicating a buying window in the next 6-12 months: new technology leadership, "
            "announced programmes, budget releases, RFPs, acquisitions to integrate, or stated deadlines."
        ),
        rubric={
            0: "No public evidence of timing or triggering events.",
            1: "No triggering event; evidence suggests spending restraint.",
            2: "Weak or dated triggers (older than 18 months) with no current programme.",
            3: (
                "One recent trigger: new technology leader, announced initiative or acquisition in "
                "the last 12 months."
            ),
            4: "Several recent triggers or a funded programme starting within 12 months.",
            5: (
                "Explicit near-term buying signal: announced budget, RFP, deadline or programme "
                "launch within 6 months."
            ),
        },
        queries=(
            "{company} appoints new chief information officer chief technology officer chief data officer",
            "{company} announces new program investment launch next year",
            "{company} acquisition integration transformation plan timeline",
        ),
        positive_signals=(
            "appointed",
            "appoints",
            "named",
            "joins as",
            "new chief",
            "announces",
            "announced",
            "launch",
            "launches",
            "investment of",
            "will invest",
            "plans to",
            "by the end of",
            "next fiscal year",
            "request for proposal",
            "rfp",
            "acquisition",
            "acquire",
            "integration",
            "roadmap",
            "multi-year",
            "budget",
        ),
        negative_signals=(
            "hiring freeze",
            "cost reduction",
            "cost-cutting",
            "layoffs",
            "restructuring",
            "spending cuts",
            "budget freeze",
            "postponed",
            "profit warning",
        ),
        discovery_question=(
            "What decisions about data and AI does {company} need to make in the next two quarters, and who "
            "owns the budget and timeline for them?"
        ),
    ),
)

CRITERIA: Mapping[Criterion, CriterionDefinition] = MappingProxyType({d.criterion: d for d in _DEFINITIONS})


def get_definition(criterion: Criterion) -> CriterionDefinition:
    return CRITERIA[criterion]


def all_definitions() -> tuple[CriterionDefinition, ...]:
    """Definitions in the canonical criterion order."""
    return tuple(CRITERIA[criterion] for criterion in Criterion)
