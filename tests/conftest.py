"""Fixtures partagées : aucun test ne doit appeler le LLM ni l'API d'embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def no_llm(monkeypatch):
    """Fait échouer tout appel réseau au LLM non mocké."""

    def _forbidden(*args, **kwargs):
        raise RuntimeError("Appel LLM interdit dans les tests : mocker call_llm_json.")

    monkeypatch.setattr("agentic_pipeline.llm_client.call_llm_json", _forbidden)
    return _forbidden
