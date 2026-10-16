"""Accès au cadrage validé (outputs/1b_cadrage/cadrage.json), seule source du savoir sur le problème.

Le domaine, les libellés des segments et l'ordre des sévérités se lisent ici. Les causes et
conséquences ne sont pas fixées à l'avance : elles sont découvertes dans les données.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from paths import FRAMING


@dataclass(frozen=True)
class Cadrage:
    raw: dict

    @property
    def domain(self) -> str:
        return self.raw["domaine"]

    @property
    def segments(self) -> list[str]:
        return list(self.raw["segments"])

    def label(self, segment: str) -> str:
        return self.raw["segments"].get(segment, segment)

    @property
    def severities(self) -> list[str]:
        return self.raw["severites"]

    @property
    def mildest(self) -> str:
        return self.raw["severites"][0]

    @property
    def most_severe(self) -> str:
        return self.raw["severites"][-1]


@lru_cache(maxsize=1)
def current() -> Cadrage:
    # Chemin relatif au dossier courant (comme toute la pipeline), sinon à la racine du repo (webapp).
    path = next((p for p in (FRAMING, Path(__file__).resolve().parent.parent / FRAMING) if p.exists()), None)
    if path is None:
        raise SystemExit(f"{FRAMING} introuvable : lancer d'abord l'étape cadrage (python main.py --steps cadrage).")
    return Cadrage(json.loads(path.read_text(encoding="utf-8")))
