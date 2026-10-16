"""Causes de chaque incident, extraites par le LLM (utilisé par l'agent d'analyse).

Les embeddings regroupent les incidents par type d'événement (incendie, surtension...), pas par
cause. Ici, le LLM lit chaque rapport et renvoie une analyse structurée (JSON imposé) : une cause
racine (le phénomène qui a déclenché l'incident), des causes contributives (les conditions qui ont
permis ou aggravé l'incident), la conséquence, le composant tel qu'écrit, l'hypothèse de maintenance
et l'action corrective. Aucune liste fermée : les thèmes sont découverts ensuite, dans les données.

Chaque cause (racine et contributives) doit citer un extrait du rapport, vérifié par le code : une
cause dont l'extrait n'est pas dans le rapport est annulée (racine) ou écartée (contributive).

L'agent d'analyse (analysis_agent) appelle analyse_incidents via son outil extract_causes ; le
cache des réponses est partagé : ce qui a été analysé une fois ne l'est plus jamais.

Le site (ni les identifiants, qui le contiennent) n'est volontairement PAS donné au LLM : l'analyse d'un incident ne doit pas être
influencée par ce qu'on sait (ou croit savoir) du site, sinon la comparaison des sites
deviendrait circulaire.

Lancement manuel (hors pipeline principale), sorties dans outputs/exploration/causes/ :
    incidents_causes.csv          chaque incident avec son analyse ;
    contributives_par_site.csv    par site : nombre de causes contributives vérifiées, % écartées,
                                  part de maintenance suspectée probable, coût et durée moyens.

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

from framing.cadrage import current
from framing.config import SEGMENT_COLUMN

from .config import (CAUSE_INCONNUE, FIELDS, MAX_CONTRIBUTIVES, PROMPT_EXCLUDED_COLUMNS, QUOTE_MIN_MATCH, UNKNOWN_CAUSE_PATTERN, WORKERS,
                     response_format, system_prompt)


def incident_prompt(row: dict) -> str:
    """Le rapport tel que le LLM le voit : toutes les colonnes sauf le site et les identifiants (voir la
    docstring du module), puis la description."""
    fields = [f"Segment : {current().label(row[SEGMENT_COLUMN])}"]
    fields += [f"{col} : {value}" for col, value in row.items()
               if col not in PROMPT_EXCLUDED_COLUMNS and col != SEGMENT_COLUMN and pd.notna(value)]
    return "\n".join(fields + [f"Description : {row['text']}"])


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
    """Contrôles déterministes de la cause racine renvoyée par le LLM :
    - une seule valeur pour une cause inconnue, quelle que soit sa formulation ;
    - une cause explicite ou inférée sans indice vérifiable dans le rapport est annulée (inconnue)."""
    result["indice_verifie"] = False
    if result["cause_type"] != "inconnue" and not UNKNOWN_CAUSE_PATTERN.search(result["cause_racine"] or ""):
        result["indice_verifie"] = quote_found(result.get("indice", ""), text)
    if not result["indice_verifie"]:
        if result["cause_type"] != "inconnue":
            result["cause_annulee"] = result["cause_racine"]  # gardée pour transparence
        result.update(cause_racine=CAUSE_INCONNUE, cause_type="inconnue")
    return result


def normalize_contributing(items: list[dict], text: str) -> tuple[list[dict], int]:
    """Garde les causes contributives dont l'extrait figure dans le rapport ; renvoie aussi le nombre
    d'items écartés (extrait absent : cause probablement inventée)."""
    kept = [item for item in items if quote_found(item.get("indice", ""), text)]
    return kept[:MAX_CONTRIBUTIVES], len(items) - len(kept)


def analyse(row: dict, prompt: str, schema: dict) -> dict:
    """Analyse d'un incident ; en cas d'échec, les champs sont vides et l'erreur est notée."""
    try:
        answer = chat([{"role": "system", "content": prompt},
                       {"role": "user", "content": incident_prompt(row)}],
                      role="extraction", response_format=schema)
        result = json.loads(answer["content"])
        result = normalize_cause(result, row["text"])
        kept, n_dropped = normalize_contributing(result["causes_contributives"], row["text"])
        result["causes_contributives"] = json.dumps(kept, ensure_ascii=False)
        result["contributives_rejetees"] = n_dropped
        return {**result, "erreur_analyse": ""}
    except Exception as e:  # filtre de contenu Azure, réponse invalide... : signalé, n'arrête rien
        return {**{f: None for f in FIELDS}, "contributives_rejetees": None,
                "erreur_analyse": f"{type(e).__name__}: {e}"[:300]}


def analyse_incidents(df: pd.DataFrame, workers: int = WORKERS) -> pd.DataFrame:
    """Analyse de chaque ligne de `df` (appels en parallèle, réponses en cache). Une ligne de
    résultat par incident, dans le même ordre et avec le même index que `df`."""
    rows = df.to_dict(orient="records")
    prompt, schema = system_prompt(), response_format()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(lambda r: analyse(r, prompt, schema), rows), total=len(rows),
                            desc="Analyse des causes"))
    return pd.DataFrame(results, index=df.index)


def contributives_by_site(ok: pd.DataFrame) -> pd.DataFrame:
    """Par site : causes contributives vérifiées, part écartée, maintenance suspectée, coût et durée."""
    ok = ok.assign(verifiees=ok["causes_contributives"].map(lambda s: len(json.loads(s))),
                   rejetees=ok["contributives_rejetees"].astype(int))
    by_site = ok.groupby("site")
    total = by_site["verifiees"].sum() + by_site["rejetees"].sum()
    return pd.DataFrame({
        "incidents": by_site.size(),
        "contributives_moyenne": by_site["verifiees"].mean().round(2),
        "pct_contributives_rejetees": (100 * by_site["rejetees"].sum() / total).round(1),
        "pct_maintenance_probable": by_site["maintenance_suspectee"].apply(
            lambda s: round(100 * (s == "probable").mean(), 1)),
        "cout_moyen_eur": by_site["estimated_cost_eur"].mean().round(),
        "arret_moyen_h": by_site["downtime_hours"].mean().round(1),
    })


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
    contributives = contributives_by_site(ok)
    contributives.to_csv(CAUSES_DIR / "contributives_par_site.csv")

    print(f"\nSauvegardé dans {CAUSES_DIR}/ : incidents_causes.csv, contributives_par_site.csv")
    if len(ok) < len(out):
        print(f"{len(out) - len(ok)} incidents non analysés (colonne erreur_analyse)")
    print("\n" + contributives.to_string())
    return CAUSES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INCIDENTS_FULL)
    parser.add_argument("--workers", type=int, default=WORKERS, help="appels LLM en parallèle")
    args = parser.parse_args()
    run(args.input, args.workers)
