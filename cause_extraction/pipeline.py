"""Cause racine de chaque incident, extraite par le LLM (utilisé par l'agent d'analyse).

Les embeddings regroupent les incidents par type d'événement (incendie, surtension...), pas par
cause. Ici, le LLM lit chaque rapport et renvoie une analyse structurée (JSON imposé) : composant,
cause racine, facteur principal parmi une liste fermée, rôle de l'usure et de la maintenance,
action corrective. Les listes fermées permettent ensuite de compter et de comparer les sites.

L'agent d'analyse (analysis_agent) appelle analyse_incidents via son outil extract_causes ; le
cache des réponses est partagé : ce qui a été analysé une fois ne l'est plus jamais.

Le site n'est volontairement PAS donné au LLM : l'analyse d'un incident ne doit pas être
influencée par ce qu'on sait (ou croit savoir) du site, sinon l'association site -> profil
deviendrait circulaire.

Lancement manuel (hors pipeline principale), sorties dans outputs/exploration/causes/ :
    incidents_causes.csv       chaque incident avec son analyse ;
    facteurs_par_site.csv      % des incidents de chaque site par facteur principal ;
    composants_par_site.csv    % des incidents de chaque site par composant ;
    indicateurs_par_site.csv   part liée à l'usure, à la maintenance, coût et durée moyens.

Usage :
    python -m cause_extraction
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm.client import chat
from paths import CAUSES_DIR, INCIDENTS_CAUSES, INCIDENTS_FULL

from .config import CAUSE_INCONNUE, QUOTE_MIN_MATCH, RESPONSE_FORMAT, SYSTEM_PROMPT, UNKNOWN_CAUSE_PATTERN, WORKERS

FIELDS = list(RESPONSE_FORMAT["json_schema"]["schema"]["properties"])


def incident_prompt(row) -> str:
    """Le rapport tel que le LLM le voit (sans le site, voir la docstring du module)."""
    asset = {"solar": "centrale photovoltaïque", "wind": "parc éolien"}.get(row.asset_type, row.asset_type)
    return (f"Type d'installation : {asset}\n"
            f"Équipement : {row.equipment_id}\n"
            f"Date : {row.date}\n"
            f"Sévérité : {row.severity}\n"
            f"Durée d'arrêt : {row.downtime_hours} h\n"
            f"Coût estimé : {row.estimated_cost_eur} €\n"
            f"Conditions météo : {row.weather_condition}\n"
            f"Description : {row.text}")


def simplify(text: str) -> str:
    """Minuscules, sans accents ni ponctuation : la comparaison tolère les petites variations de recopie."""
    text = unicodedata.normalize("NFKD", (text or "").lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def quote_found(quote: str, text: str) -> bool:
    """L'extrait cité par le LLM figure-t-il dans le rapport (à QUOTE_MIN_MATCH près) ?"""
    q, t = simplify(quote), simplify(text)
    if len(q.split()) < 3:
        return False
    match = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return match.size >= QUOTE_MIN_MATCH * len(q)


def normalize_cause(result: dict, text: str) -> dict:
    """Contrôles déterministes de la cause renvoyée par le LLM :
    - une seule valeur pour une cause inconnue, quelle que soit sa formulation ;
    - une cause explicite ou inférée sans indice vérifiable dans le rapport est annulée (inconnue)."""
    result["indice_verifie"] = False
    if result["cause_type"] != "inconnue" and not UNKNOWN_CAUSE_PATTERN.search(result["cause_racine"] or ""):
        result["indice_verifie"] = quote_found(result.get("indice", ""), text)
    if not result["indice_verifie"]:
        if result["cause_type"] != "inconnue":
            result["cause_annulee"] = result["cause_racine"]  # gardée pour transparence
        result.update(cause_racine=CAUSE_INCONNUE, cause_type="inconnue", facteur_principal="indéterminé")
    return result


def analyse(row) -> dict:
    """Analyse d'un incident ; en cas d'échec, les champs sont vides et l'erreur est notée."""
    try:
        answer = chat([{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": incident_prompt(row)}],
                      role="extraction", response_format=RESPONSE_FORMAT)
        result = json.loads(answer["content"])
        result["facteurs_secondaires"] = " | ".join(result["facteurs_secondaires"])
        return {**normalize_cause(result, row.text), "erreur_analyse": ""}
    except Exception as e:  # filtre de contenu Azure, réponse invalide... : signalé, n'arrête rien
        return {**{f: None for f in FIELDS}, "erreur_analyse": f"{type(e).__name__}: {e}"[:300]}


def analyse_incidents(df: pd.DataFrame, workers: int = WORKERS) -> pd.DataFrame:
    """Analyse de chaque ligne de `df` (appels en parallèle, réponses en cache). Une ligne de
    résultat par incident, dans le même ordre et avec le même index que `df`."""
    rows = list(df.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(analyse, rows), total=len(rows), desc="Analyse des causes"))
    return pd.DataFrame(results, index=df.index)


def percent_by_site(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = (pd.crosstab(df[column], df["site"], normalize="columns") * 100).round(1)
    return table.loc[table.max(axis=1).sort_values(ascending=False).index]


def run(input_csv: Path, workers: int) -> Path:
    if not input_csv.exists():
        raise SystemExit(f"{input_csv} introuvable : lancer d'abord l'étape pdf (python main.py --steps pdf).")
    df = pd.read_csv(input_csv)
    df = df[df["text"].notna()].reset_index(drop=True)
    print(f"{len(df)} incidents à analyser (réponses déjà en cache : aucun nouvel appel)")

    keep = ["incident_id", "site", "asset_type", "equipment_id", "date", "severity", "downtime_hours",
            "estimated_cost_eur", "weather_condition", "text"]
    out = pd.concat([df[keep], analyse_incidents(df, workers)], axis=1)
    CAUSES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(INCIDENTS_CAUSES, index=False)

    ok = out[out["erreur_analyse"] == ""]
    percent_by_site(ok, "facteur_principal").to_csv(CAUSES_DIR / "facteurs_par_site.csv")
    percent_by_site(ok, "composant").to_csv(CAUSES_DIR / "composants_par_site.csv")
    indicators = ok.groupby("site").agg(
        incidents=("incident_id", "count"),
        pct_lie_vieillissement=("lie_vieillissement", lambda s: round(100 * s.astype(bool).mean(), 1)),
        pct_lie_maintenance=("lie_maintenance", lambda s: round(100 * s.astype(bool).mean(), 1)),
        pct_maintenance_suspectee=("maintenance_suspectee", lambda s: round(100 * (s == "probable").mean(), 1)),
        cout_moyen_eur=("estimated_cost_eur", lambda s: round(s.mean())),
        arret_moyen_h=("downtime_hours", lambda s: round(s.mean(), 1)),
        pct_confiance_faible=("confiance", lambda s: round(100 * (s == "faible").mean(), 1)),
    )
    indicators.to_csv(CAUSES_DIR / "indicateurs_par_site.csv")

    print(f"\nSauvegardé dans {CAUSES_DIR}/ : incidents_causes.csv, facteurs_par_site.csv, "
          "composants_par_site.csv, indicateurs_par_site.csv")
    if len(ok) < len(out):
        print(f"{len(out) - len(ok)} incidents non analysés (colonne erreur_analyse)")
    print("\nFacteurs dominants par site :")
    factors = percent_by_site(ok, "facteur_principal").drop(index="indéterminé", errors="ignore")
    for site in factors.columns:
        top = factors[site].sort_values(ascending=False).head(3)
        print(f"  {site:22s} " + ", ".join(f"{f} {p:.0f} %" for f, p in top.items()))
    print("\n" + indicators.to_string())
    return CAUSES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INCIDENTS_FULL)
    parser.add_argument("--workers", type=int, default=WORKERS, help="appels LLM en parallèle")
    args = parser.parse_args()
    run(args.input, args.workers)

