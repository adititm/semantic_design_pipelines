"""Runnable, configurable scripts for the semantic-design pipelines.

Each pipeline is a single YAML-configured entry point that handles sequence
generation, ORF calling, quality filtering, structure prediction and scoring
end to end, with every tool running in an automatically provisioned
environment. Thresholds are documented with the data behind them; see
``README.md``.

Importing anything here resolves ``proto_language`` and ``proto_tools``
first, so import order inside a module never matters. See
:mod:`semantic_design_pipelines.core.bootstrap`.
"""

from semantic_design_pipelines.core import bootstrap as _bootstrap

_bootstrap.activate()
