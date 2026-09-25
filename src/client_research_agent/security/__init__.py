"""Security controls mapped to the OWASP Top 10 for LLM Applications (2025)."""

from client_research_agent.security.output_guard import (
    OutputGuard,
    OutputReport,
    OutputViolation,
    ViolationKind,
)
from client_research_agent.security.pii import (
    PiiDetector,
    PiiFinding,
    PiiPolicy,
    PiiRedactor,
    PiiType,
    RedactionResult,
)
from client_research_agent.security.poisoning import (
    ChunkAction,
    ChunkDecision,
    PoisoningPolicy,
    PoisoningReport,
    RetrievalPoisoningGuard,
)
from client_research_agent.security.prompt_injection import (
    InjectionAssessment,
    InjectionSignal,
    PromptInjectionDetector,
    SpotlightedText,
    spotlight,
)
from client_research_agent.security.rate_limiter import (
    BudgetExceededError,
    PrincipalRateLimiter,
    RateLimitExceededError,
    RunBudget,
    TokenBucket,
)
from client_research_agent.security.rbac import (
    AccessDeniedError,
    Permission,
    Principal,
    Role,
    authorize,
    map_groups_to_roles,
    requires,
)
from client_research_agent.security.sanitizer import ContentSanitizer, SanitizedText
from client_research_agent.security.secrets import (
    ChainedSecretProvider,
    DatabricksSecretProvider,
    EnvSecretProvider,
    SecretProvider,
)

__all__ = [
    "AccessDeniedError",
    "BudgetExceededError",
    "ChainedSecretProvider",
    "ChunkAction",
    "ChunkDecision",
    "ContentSanitizer",
    "DatabricksSecretProvider",
    "EnvSecretProvider",
    "InjectionAssessment",
    "InjectionSignal",
    "OutputGuard",
    "OutputReport",
    "OutputViolation",
    "Permission",
    "PiiDetector",
    "PiiFinding",
    "PiiPolicy",
    "PiiRedactor",
    "PiiType",
    "PoisoningPolicy",
    "PoisoningReport",
    "Principal",
    "PrincipalRateLimiter",
    "PromptInjectionDetector",
    "RateLimitExceededError",
    "RedactionResult",
    "RetrievalPoisoningGuard",
    "Role",
    "RunBudget",
    "SanitizedText",
    "SecretProvider",
    "SpotlightedText",
    "TokenBucket",
    "ViolationKind",
    "authorize",
    "map_groups_to_roles",
    "requires",
    "spotlight",
]
