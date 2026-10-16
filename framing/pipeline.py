"""Étape 1b : cadrage de l'analyse, à partir du contexte écrit par l'utilisateur et des données.

Les agents ne savent RIEN du problème a priori. Tout ce qu'ils en savent vient d'ici :
  1. context/contexte.md : un court texte écrit par l'utilisateur (qui est le client, quels sites,
     ce qu'on sait d'eux) ;
  2. un échantillon des données : valeurs des colonnes, descriptions d'incidents tirées au hasard.

Le LLM en déduit trois éléments (JSON imposé) : le domaine, le libellé de chaque segment et l'ordre
des sévérités, de la moins grave à la plus grave. Le CODE vérifie ce cadre contre les données (tous
les segments ont un libellé, toutes les sévérités présentes) ; en cas d'erreur, le LLM reçoit la
liste des erreurs et réessaie.

Le cadrage est relu par un humain avant l'analyse : outputs/1b_cadrage/cadrage.md. Pour le corriger,
copier outputs/1b_cadrage/cadrage.json en context/cadrage.json et l'éditer : ce fichier remplace
alors la proposition du LLM (il est vérifié de la même façon).

Usage :
    python -m framing
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from llm.client import chat
from paths import CONTEXT_PROMPT, FRAMING, FRAMING_DIR, FRAMING_OVERRIDE, INCIDENTS_FULL

from .config import (MAX_CATEGORY_VALUES, RESPONSE_SCHEMA, RETRIES, ROLE, SAMPLE_MAX_CHARS, SAMPLE_PER_SEGMENT,
                     SAMPLE_SEED, SEGMENT_COLUMN, TASK)

# Colonnes jamais montrées comme catégories (identifiants, texte libre, chiffres).
NOT_CATEGORIES = {"incident_id", "site", "text", "date", "document_path", "document_format", "equipment_id",
                  "technician_id", "extraction_method", "extraction_warnings"}


def read_context() -> str:
    if not CONTEXT_PROMPT.exists():
        raise SystemExit(f"{CONTEXT_PROMPT} introuvable : décrire le problème en quelques lignes (voir README).")
    return CONTEXT_PROMPT.read_text(encoding="utf-8").strip()


def data_summary(df: pd.DataFrame) -> str:
    """Ce que le LLM voit des données : colonnes, valeurs, sites par segment, descriptions tirées au hasard."""
    lines = [f"{len(df)} incidents, du {df['date'].min()} au {df['date'].max()}.", "Colonnes et valeurs :"]
    for col in df.columns:
        if col in NOT_CATEGORIES or df[col].dtype.kind in "fiu":
            continue
        values = df[col].dropna().value_counts()
        if 1 < len(values) <= MAX_CATEGORY_VALUES:
            lines.append(f"- {col} : " + ", ".join(f"{v} ({n})" for v, n in values.items()))
    for segment, g in df.groupby(SEGMENT_COLUMN):
        lines.append(f"\nSegment « {segment} » ({SEGMENT_COLUMN}) : {len(g)} incidents, sites : "
                     + ", ".join(f"{s} ({n})" for s, n in g["site"].value_counts().sort_index().items()))
        lines.append("Descriptions d'incidents tirées au hasard :")
        sample = g[g["text"].notna()].sample(min(SAMPLE_PER_SEGMENT, len(g)), random_state=SAMPLE_SEED)
        lines += [f"- {str(t)[:SAMPLE_MAX_CHARS]}" for t in sample["text"]]
    return "\n".join(lines)


def from_llm(raw: dict) -> dict:
    """Réponse du LLM (listes) -> format du cadrage (dictionnaires), celui de cadrage.json."""
    return {"domaine": raw["domaine"], "segments": {s["valeur"]: s["libelle"] for s in raw["segments"]},
            "severites": raw["severites"]}


def clean(name) -> str:
    return " ".join(str(name or "").split())


def validate(c: dict, df: pd.DataFrame) -> tuple[dict, list[str], list[str]]:
    """Vérifie un cadrage contre les données. Renvoie (cadrage nettoyé, erreurs, avertissements) :
    une erreur fait redemander le cadrage, un avertissement est corrigé par le code et signalé."""
    errors, warnings = [], []
    segments = sorted(df[SEGMENT_COLUMN].dropna().unique())

    labels = {s: clean(c.get("segments", {}).get(s)) for s in segments}
    for s in [s for s, label in labels.items() if not label]:
        warnings.append(f"segment « {s} » sans libellé : la valeur brute est utilisée")
        labels[s] = s

    severities = [clean(x) for x in c.get("severites", [])]
    expected = sorted(df["severity"].dropna().astype(str).unique())
    if sorted(severities) != expected:
        errors.append(f"severites doit contenir exactement ces valeurs, chacune une fois, de la moins grave à la plus "
                      f"grave : {expected} (reçu : {severities})")

    framing = {"domaine": clean(c.get("domaine")), "segments": labels, "severites": severities}
    return framing, errors, warnings


def propose(context: str, df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Cadrage proposé par le LLM, redemandé tant que le code y trouve des erreurs (RETRIES fois)."""
    messages = [{"role": "system", "content": ROLE},
                {"role": "user", "content": f"CONTEXTE (écrit par l'utilisateur) :\n{context}\n\n"
                                            f"DONNÉES :\n{data_summary(df)}\n\n{TASK.format(segment_column=SEGMENT_COLUMN)}"}]
    response_format = {"type": "json_schema", "json_schema": {"name": "cadrage", "strict": True, "schema": RESPONSE_SCHEMA}}
    for attempt in range(1, RETRIES + 2):
        answer = chat(messages, role="analysis", response_format=response_format)
        framing, errors, warnings = validate(from_llm(json.loads(answer["content"])), df)
        if not errors:
            return framing, warnings
        print(f"  cadrage, essai {attempt} refusé : {' ; '.join(errors)}")
        messages += [{"role": "assistant", "content": answer["content"]},
                     {"role": "user", "content": "Le code a trouvé ces erreurs : " + json.dumps(errors, ensure_ascii=False)
                      + ". Corrige-les et réponds à nouveau en JSON."}]
    raise SystemExit(f"Cadrage refusé après {RETRIES + 1} essais ({' ; '.join(errors)}). Préciser {CONTEXT_PROMPT}, "
                     f"ou écrire le cadrage à la main dans {FRAMING_OVERRIDE} (même format que {FRAMING}).")


def markdown(framing: dict) -> str:
    """Version lisible, à relire avant l'analyse."""
    lines = ["# Cadrage de l'analyse\n", f"*Source : {framing['source']}. Pour corriger : copier `{FRAMING}` en "
             f"`{FRAMING_OVERRIDE}` et l'éditer.*\n", f"**Domaine :** {framing['domaine']}\n", "## Segments\n"]
    lines += [f"- `{v}` : {label}" for v, label in framing["segments"].items()]
    lines += ["\n## Sévérités (de la moins grave à la plus grave)\n", " < ".join(framing["severites"]) + "\n"]
    if framing["avertissements"]:
        lines += ["\n## Avertissements\n", *[f"- {w}" for w in framing["avertissements"]]]
    return "\n".join(lines) + "\n"


def run() -> dict:
    if not INCIDENTS_FULL.exists():
        raise SystemExit(f"{INCIDENTS_FULL} introuvable : lancer d'abord l'étape pdf.")
    df = pd.read_csv(INCIDENTS_FULL)
    context = read_context()
    if FRAMING_OVERRIDE.exists():
        framing, errors, warnings = validate(json.loads(FRAMING_OVERRIDE.read_text(encoding="utf-8")), df)
        if errors:
            raise SystemExit(f"{FRAMING_OVERRIDE} invalide : {' ; '.join(errors)}")
        source = f"écrit à la main ({FRAMING_OVERRIDE})"
    else:
        framing, warnings = propose(context, df)
        source = f"proposé par le LLM à partir de {CONTEXT_PROMPT} et d'un échantillon des données"
    framing = {"source": source, "contexte": context, **framing, "avertissements": warnings}
    FRAMING_DIR.mkdir(parents=True, exist_ok=True)
    FRAMING.write_text(json.dumps(framing, ensure_ascii=False, indent=2), encoding="utf-8")
    (FRAMING_DIR / "cadrage.md").write_text(markdown(framing), encoding="utf-8")
    from .cadrage import current
    current.cache_clear()
    print(f"Cadrage ({source}) : domaine « {framing['domaine']} », {len(framing['segments'])} segments, "
          f"sévérités : {' < '.join(framing['severites'])}")
    for w in warnings:
        print(f"  ⚠️ {w}")
    print(f"À relire : {FRAMING_DIR / 'cadrage.md'}")
    return framing


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run()
