"""Accès au cadrage validé (outputs/1b_cadrage/cadrage.json), seule source du savoir sur le problème.

Tout ce qui, avant, était écrit en dur pour un problème donné (facteurs, composants, catégories de
conséquence, profils de site, mesures préventives types, libellés des segments...) se lit ici.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from paths import FRAMING

from .config import AGE_CRITERION, MAINTENANCE_CRITERION, OTHER, UNDETERMINED


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
    def factors(self) -> list[str]:
        """Facteurs à l'origine des incidents (liste fermée de l'extraction), « indéterminé » compris."""
        return [f["nom"] for f in self.raw["facteurs"]] + [UNDETERMINED]

    @property
    def components(self) -> list[str]:
        return self.raw["composants"] + [OTHER]

    @property
    def consequences(self) -> list[str]:
        return [c["nom"] for c in self.raw["categories_consequence"]] + [OTHER]

    @property
    def nominal_consequences(self) -> list[str]:
        """Catégories qui sont le fonctionnement normal d'une protection (exclues des plans)."""
        return [c["nom"] for c in self.raw["categories_consequence"] if c["fonctionnement_normal"]]

    @property
    def severities(self) -> list[str]:
        return self.raw["severites"]

    @property
    def mildest(self) -> str:
        return self.raw["severites"][0]

    @property
    def most_severe(self) -> str:
        return self.raw["severites"][-1]

    def measure(self, factor: str) -> tuple[str, str, str] | None:
        """Mesure préventive type d'un facteur (type, mesure, mécanisme), proposée au cadrage."""
        f = next((f for f in self.raw["facteurs"] if f["nom"] == factor), None)
        return (f["type"], f["mesure"].rstrip("."), f["mecanisme"].rstrip(".")) if f and f["mesure"] else None

    def concepts(self) -> dict[str, str]:
        """Phénomènes surveillés par le validateur de rédaction : {facteur: regex de ses mots-clés}."""
        return {f["nom"]: "|".join(re.escape(k) for k in f["mots_cles"]) for f in self.raw["facteurs"] if f["mots_cles"]}

    # ---- profils de site (facultatifs : seulement si le contexte en décrit) ------------------------

    def profiles(self, segment: str) -> dict[str, str]:
        """{identifiant: description} des profils du segment ({} : pas de profils)."""
        return {pid: p["description"] for pid, p in self.raw["profils"].get(segment, {}).items()}

    def expectations(self, segment: str) -> dict[str, list[tuple[str, str, str]]]:
        """{profil: [(critère, « élevé » ou « faible », raison)]}."""
        return {pid: [(e["critere"], e["sens"], e["raison"]) for e in p["attentes"]]
                for pid, p in self.raw["profils"].get(segment, {}).items()}

    @property
    def profiled_segments(self) -> list[str]:
        return [s for s in self.segments if self.profiles(s)]

    @property
    def criteria(self) -> list[str]:
        """Critères mesurés sur chaque site : les facteurs et les deux indicateurs de l'extraction."""
        return [f["nom"] for f in self.raw["facteurs"]] + [AGE_CRITERION, MAINTENANCE_CRITERION]


@lru_cache(maxsize=1)
def current() -> Cadrage:
    # Chemin relatif au dossier courant (comme toute la pipeline), sinon à la racine du repo (webapp).
    path = next((p for p in (FRAMING, Path(__file__).resolve().parent.parent / FRAMING) if p.exists()), None)
    if path is None:
        raise SystemExit(f"{FRAMING} introuvable : lancer d'abord l'étape cadrage (python main.py --steps cadrage).")
    return Cadrage(json.loads(path.read_text(encoding="utf-8")))
