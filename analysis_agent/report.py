"""Mise en forme du rapport d'analyse et export des résultats.

Le texte (synthèse, causes, chaînes cause -> conséquence, profils, actions) vient de l'agent (outil
write_report) ; tous les tableaux chiffrés sont calculés ici, à partir des données, pour qu'aucun
chiffre des tableaux ne puisse être inventé.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from llm.client import env

from .config import ASSET_LABELS, ASSET_TYPES, DIMENSION_LABELS, DIMENSIONS, SITE_PROFILES, TOP_PAIRS
from .tools import AnalysisWorkspace, profile_indicators


def md_table(df: pd.DataFrame) -> str:
    """Tableau Markdown (sans dépendance supplémentaire)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def percent_table(df: pd.DataFrame, column: str, top: int = 6) -> pd.DataFrame:
    """% des incidents de chaque site par valeur de `column` (valeurs les plus fréquentes)."""
    table = (pd.crosstab(df[column].fillna("non classé"), df["site"], normalize="columns") * 100).round(0)
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index[:top]].astype(int).astype(str) + " %"
    return table.reset_index().rename(columns={column: ""})


def pairs_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paires cause -> conséquence les plus fréquentes, avec leur coût."""
    t = df.assign(theme_cause=df["theme_cause"].fillna("non classé"),
                  theme_consequence=df["theme_consequence"].fillna("non classé"))
    pairs = (t.groupby(["theme_cause", "theme_consequence"])
             .agg(incidents=("incident_id", "count"), cout_moyen=("estimated_cost_eur", "mean"),
                  cout_total=("estimated_cost_eur", "sum"))
             .sort_values("incidents", ascending=False).head(TOP_PAIRS).reset_index())
    pairs["part"] = (100 * pairs["incidents"] / len(t)).round(0).astype(int).astype(str) + " %"
    pairs["cout_moyen"] = pairs["cout_moyen"].round(0).astype(int).astype(str) + " €"
    pairs["cout_total"] = pairs["cout_total"].round(0).astype(int).astype(str) + " €"
    return pairs.rename(columns={"theme_cause": "cause", "theme_consequence": "conséquence"})[
        ["cause", "conséquence", "incidents", "part", "cout_moyen", "cout_total"]]


def render(ws: AnalysisWorkspace) -> str:
    report = ws.report
    incidents = ws.incidents_table()
    ok = incidents[incidents["erreur_analyse"] == ""]
    lines = [
        "# Rapport d'analyse des incidents PV et éolien",
        f"\n*Généré le {date.today():%d/%m/%Y} par l'agent d'analyse (modèle "
        f"`{env('AZURE_OPENAI_CHAT_DEPLOYMENT')}`). Le texte est rédigé par l'agent ; tous les tableaux sont "
        "calculés à partir des données. Déroulé complet des décisions : `journal.md`.*",
        "\n## 1. Synthèse\n", report["synthese"],
        "\n## 2. Données et méthode\n",
        f"{len(incidents)} incidents ({incidents['date'].min()} à {incidents['date'].max()}), "
        f"{incidents['site'].nunique()} sites. Pour chaque incident, le LLM extrait sa **cause** (pourquoi) et sa "
        "**conséquence** (ce qu'il a provoqué). Chacune est ensuite regroupée par clustering (embeddings, UMAP, "
        "HDBSCAN), séparément pour le photovoltaïque et l'éolien, puis les deux sont croisées.\n",
    ]
    runs = pd.DataFrame([r["summary"] for r in ws.runs.values()])
    runs["choisi"] = runs["run_id"].isin([s["run_id"] for s in ws.selected.values()]).map({True: "✅", False: ""})
    lines.append(md_table(runs[["asset_type", "dimension", "min_cluster_size", "nombre_clusters", "non_classes_pct",
                                "dbcv", "stabilite", "conforme_aux_seuils", "choisi"]]))
    for k, choice in ws.selected.items():
        lines.append(f"\n**Choix {k}** (`{choice['run_id']}`) : {choice['justification']}")

    lines.append("\n## 3. Thèmes de causes et de conséquences\n")
    for d in DIMENSIONS:
        themes = pd.DataFrame([{"type": ASSET_LABELS[t["asset_type"]], "thème": t["name"], "incidents": t["incidents"],
                                "cohésion": t["cohesion"], "description": t["description"]}
                               for t in ws.themes.values() if t["dimension"] == d])
        lines += [f"\n**Thèmes de {DIMENSION_LABELS[d]}**\n",
                  md_table(themes.sort_values(["type", "incidents"], ascending=[True, False]))]

    lines += ["\n## 4. Causes racines\n", report["causes_racines"], "\n**Thèmes de causes par site**\n"]
    for asset_type in ASSET_TYPES:
        lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()}*\n",
                  md_table(percent_table(ok[ok["asset_type"] == asset_type], "theme_cause"))]

    lines += ["\n## 5. Chaînes cause → conséquence\n", report["chaines_cause_consequence"]]
    for asset_type in ASSET_TYPES:
        lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()} : paires les plus fréquentes*\n",
                  md_table(pairs_table(incidents[incidents["asset_type"] == asset_type]))]

    lines += ["\n## 6. Analyse par site\n",
              "**Critères des profils mesurés sur chaque site** (rang parmi les 3 sites du même type, 1 = le plus "
              "élevé). C'est sur ce tableau que s'appuie l'association site → profil.\n"]
    for asset_type in ASSET_TYPES:
        ind = profile_indicators(ok[ok["asset_type"] == asset_type])
        table = pd.DataFrame({site: {c: f"{v[site]['valeur']} ({v[site]['rang'] or '='})" for c, v in ind.items()}
                              for site in sorted(next(iter(ind.values())))})
        lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()}*\n", md_table(table.reset_index().rename(columns={"index": "critère"}))]
    for s in report["sites"]:
        g = ok[ok["site"] == s["site"]]
        asset_type = g["asset_type"].iloc[0]
        top_causes = ", ".join(f"{k} {v:.0f} %" for k, v in
                               (g["theme_cause"].value_counts(normalize=True).head(3) * 100).items())
        lines += [
            f"\n### {s['site']} — profil {s['profil']}\n",
            f"*Profil {s['profil']} ({ASSET_LABELS[asset_type]}) : {SITE_PROFILES[asset_type][s['profil']]}.*\n",
            f"**Pourquoi ce profil :** {s['justification_profil']}\n",
            f"**Causes dominantes :** {s['causes_dominantes']}\n",
            f"Chiffres : {len(g)} incidents ; principales causes : {top_causes} ; "
            f"{100 * g['lie_vieillissement'].astype(bool).mean():.0f} % liés à l'usure, "
            f"{100 * g['lie_maintenance'].astype(bool).mean():.0f} % liés à la maintenance, "
            f"{100 * (g['severity'] == 'sévère').mean():.0f} % sévères, coût moyen "
            f"{g['estimated_cost_eur'].mean():.0f} €, arrêt moyen {g['downtime_hours'].mean():.1f} h.\n",
            "**Plan de remédiation**\n",
            md_table(pd.DataFrame(s["plan_remediation"])[["priorite", "action", "justification"]]
                     .sort_values("priorite", key=lambda p: p.map({"haute": 0, "moyenne": 1, "basse": 2}))),
        ]
    lines += ["\n## 7. Limites\n", report["limites"]]
    return "\n".join(lines) + "\n"


def export(ws: AnalysisWorkspace, out_dir: Path) -> None:
    """Résultats chiffrés de la session (même si l'agent n'a pas terminé)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {k: {kk: vv for kk, vv in r["summary"].items() if kk != "clusters"} for k, r in ws.runs.items()}
    if summaries:
        pd.DataFrame(summaries.values()).to_csv(out_dir / "clustering_essais.csv", index=False)
    if ws.themes:
        pd.DataFrame([{"theme_id": tid, **t} for tid, t in ws.themes.items()]).to_csv(out_dir / "themes.csv", index=False)
    ws.incidents_table().drop(columns=["saison"]).to_csv(out_dir / "incidents_analyse.csv", index=False)
    state = {"clustering_choisi": ws.selected, "rapport": ws.report, "essais": summaries}
    (out_dir / "decisions.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
