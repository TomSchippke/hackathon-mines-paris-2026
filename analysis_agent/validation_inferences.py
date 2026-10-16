"""Validation manuelle des causes INFÉRÉES par le LLM.

1. `inference_sample` tire un échantillon aléatoire stratifié (graine fixe) d'incidents dont la cause
   a été inférée (VALIDATION_SAMPLE par segment) et l'écrit dans
   outputs/validation/inferences_a_valider.csv, avec une colonne vide `verdict_humain`.
   Le fichier est créé une seule fois et n'est JAMAIS écrasé : il contient le travail manuel.
2. On remplit `verdict_humain` à la main : « oui » si la cause inférée est correcte, « non » sinon.
3. `agreement` calcule le taux d'accord ; le rapport l'intègre automatiquement (ou indique que la
   validation est en cours). Une ligne dont la cause a changé depuis (nouvelle extraction) est
   comptée comme obsolète, pas dans le taux.

Usage :
    python -m analysis_agent.validation_inferences      # crée le fichier s'il n'existe pas, affiche le taux
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import ANALYSIS_DIR, INFERENCE_VALIDATION

from .config import VALIDATION_SAMPLE, VALIDATION_SEED

YES, NO = {"oui", "o", "yes", "y", "1", "vrai", "correct", "ok"}, {"non", "n", "no", "0", "faux", "incorrect"}


def inference_sample(incidents: pd.DataFrame, path: Path = INFERENCE_VALIDATION) -> Path:
    """Crée le fichier de validation s'il n'existe pas (sinon le laisse intact)."""
    if path.exists():
        return path
    inferred = incidents[incidents["cause_type"] == "inférée"]
    if inferred.empty:  # rien à valider : pas de fichier (il sera créé quand il y aura des inférences)
        return path
    sample = pd.concat([g.sample(min(VALIDATION_SAMPLE, len(g)), random_state=VALIDATION_SEED)
                        for _, g in inferred.groupby("asset_type")])
    out = sample[["incident_id", "asset_type", "site", "text", "cause_racine", "indice", "facteur_principal"]].rename(
        columns={"text": "rapport", "cause_racine": "cause_inferee"})
    out["verdict_humain"] = ""
    out["commentaire"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def agreement(incidents: pd.DataFrame, path: Path = INFERENCE_VALIDATION) -> dict:
    """Taux d'accord humain / LLM sur les causes inférées : {"remplies", "accord_pct", "obsoletes", "total"}."""
    if not path.exists():
        return {"total": 0, "remplies": 0, "accord_pct": None, "obsoletes": 0}
    v = pd.read_csv(path, dtype=str).fillna("")
    current = incidents.set_index("incident_id")["cause_racine"].astype(str)
    v["obsolete"] = [current.get(int(i), "") != c for i, c in zip(v["incident_id"], v["cause_inferee"])]
    verdict = v["verdict_humain"].str.strip().str.lower()
    filled = v[~v["obsolete"] & verdict.isin(YES | NO)]
    rate = round(100 * verdict[filled.index].isin(YES).mean(), 1) if len(filled) else None
    return {"total": len(v), "remplies": len(filled), "accord_pct": rate, "obsoletes": int(v["obsolete"].sum())}


def agreement_text(incidents: pd.DataFrame) -> str:
    a = agreement(incidents)
    if a["total"] == 0:
        return "validation manuelle des causes inférées : fichier non créé"
    if a["remplies"] == 0:
        return (f"validation manuelle des causes inférées en cours ({a['total']} incidents à vérifier dans "
                f"`{INFERENCE_VALIDATION}`)")
    return (f"validation manuelle des causes inférées : {a['accord_pct']} % d'accord sur {a['remplies']} incidents "
            f"vérifiés à la main" + (f" ({a['obsoletes']} lignes obsolètes ignorées)" if a["obsoletes"] else ""))


def main() -> None:
    incidents = pd.read_csv(ANALYSIS_DIR / "incidents_analyse.csv")
    print(f"Fichier : {inference_sample(incidents)}")
    print(agreement_text(incidents))


if __name__ == "__main__":
    main()
