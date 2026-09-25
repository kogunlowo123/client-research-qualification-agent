"""Models-from-code entry point logged by ``cra-deploy-agent --stage log``.

MLflow executes this file when the model is loaded (in Model Serving or via
``mlflow.pyfunc.load_model``). The agent builds its runtime lazily from
environment settings (``CRA_ENVIRONMENT`` and ``CRA_*`` variables set on the
serving endpoint) on the first request, so loading the model never contacts
Databricks.
"""

from __future__ import annotations

import importlib

from client_research_agent.agent.serving_agent import ClientResearchResponsesAgent

AGENT = ClientResearchResponsesAgent()

importlib.import_module("mlflow.models").set_model(AGENT)
