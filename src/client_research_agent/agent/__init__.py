"""Composition root, LLM metering and the MLflow ResponsesAgent serving wrapper.

``serving_agent`` and ``agent_model`` import MLflow lazily and are not imported
here, so the package works without the ``databricks`` extra.
"""

from client_research_agent.agent.factory import AgentRuntime, IngestionRunner, build_runtime
from client_research_agent.agent.metering import (
    LLMBudgetExhaustedError,
    MeteredLLMClient,
    RunAccounting,
    run_accounting,
)

__all__ = [
    "AgentRuntime",
    "IngestionRunner",
    "LLMBudgetExhaustedError",
    "MeteredLLMClient",
    "RunAccounting",
    "build_runtime",
    "run_accounting",
]
