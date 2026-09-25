"""Databricks adapters for the hexagonal ports.

Each module binds one port to a Databricks service and imports its vendor SDK
directly, so this package requires the ``databricks`` extra
(``databricks-sdk``, ``databricks-vectorsearch``/``databricks-ai-search``,
``mlflow`` and ``openai``). Nothing outside this package imports those SDKs.

* ``auth`` - unified-auth ``WorkspaceClient`` construction and bearer tokens
* ``model_serving`` - chat completions over the Foundation Model API
* ``embeddings`` - embeddings over the Foundation Model API
* ``vector_search`` - Delta Sync Vector Search index adapter and provisioning
* ``unity_catalog`` - Delta tables via the SQL Statement Execution API
* ``mlflow_registry`` - Unity Catalog model registry helpers
* ``jobs`` - Databricks Workflows trigger/monitor helper
"""
