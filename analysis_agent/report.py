"""Mise en forme du rapport d'analyse et export des résultats.

Tous les chiffres et tableaux viennent de facts.py (calculés en Python). Les seuls textes rédigés
par le LLM (caractérisation et causes dominantes des sites, mesures du plan, synthèse) ont été validés par
writing.py, ou remplacés par du texte Python. Les pourcentages d'un site sont toujours exprimés en
part de l'ensemble de ses incidents.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from framing.cadrage import current
from llm.client import deployment

from . import facts as F
from .config import (DIMENSION_LABELS, DIMENSIONS, MAINTENANCE_INDICATOR, MAX_RECLUSTER_ROUNDS, NOMINAL_LEVERS,
                     NOMINAL_MIN_MILDEST_PCT, NONE_MAX_PCT, PLAN_COVERAGE_PCT, PLAN_MAX_MEASURES, PLAN_MIN_INCIDENTS)
from .tools import AnalysisWorkspace
from .validation_inferences import agreement_text

# Rampe séquentielle bleue (une teinte, clair -> foncé) pour la heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PERCENT_NOTE = "Pourcentages : part de l'ensemble des incidents du site (causes non précisées comprises)."


def md_table(df: pd.DataFrame) -> str:
    """Tableau Markdown (sans dépendance supplémentaire)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def percent_table(df: pd.DataFrame, column: str, top: int = 8) -> pd.DataFrame:
    """% des incidents de chaque site par valeur de `column` (valeurs les plus fréquentes)."""
    table = (pd.crosstab(df[column].fillna(F.UNCLASSIFIED), df["site"], normalize="columns") * 100).round(1)
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index[:top]].astype(str) + " %"
    return table.reset_index().rename(columns={column: ""})


def indicators_table(indicators: dict) -> pd.DataFrame:
    sites = sorted(next(iter(indicators.values()))["valeurs"])
    rows = [{"critère": c, **{s: f"{d['valeurs'][s]} {d['unite']}" for s in sites}} for c, d in indicators.items()]
    return pd.DataFrame(rows)


def render(ws: AnalysisWorkspace) -> str:
    report = ws.report
    c = current()
    segments = ws.segments
    labels = {a: c.label(a) for a in segments}
    incidents = ws.incidents_table()
    ok = incidents[incidents["erreur_analyse"] == ""]
    names = {tid: th["name"] for tid, th in ws.themes.items() if th["dimension"] == "cause"}
    juge = report.get("juge", {})
    lines = [
        "# Rapport d'analyse des incidents",
        f"\n**Contexte :** {c.domain}",
        f"\n*Généré le {date.today():%d/%m/%Y}. Agent d'analyse : `{deployment('analysis')}` ; rédaction : "
        f"`{deployment('writing')}`. Tous les chiffres sont calculés à partir des données ; les textes rédigés par le "
        "LLM ont été vérifiés par un validateur (chiffres et phénomènes cités), puis relus par un juge dont les "
        f"corrections ne sont acceptées que si elles passent ce validateur ({juge.get('corriges', 0)} "
        f"corrections acceptées, {juge.get('refuses', 0)} refusées). Déroulé complet : `journal.md`.*",
        "\n## 1. Synthèse\n", report["synthese"],
    ]

    # 2. Données et méthode
    dropped = int(ok["contributives_rejetees"].sum()) if "contributives_rejetees" in ok else 0
    lines += ["\n## 2. Données et méthode\n",
              f"{len(incidents)} incidents ({incidents['date'].min()} à {incidents['date'].max()}), "
              f"{incidents['site'].nunique()} sites. Pour chaque incident, le LLM extrait sa **cause racine** (pourquoi), "
              "des **causes contributives** (conditions qui l'ont rendu possible ou aggravé, chacune avec un extrait "
              "du rapport vérifié par le code) et sa **conséquence**. Aucune liste de causes n'est fixée à l'avance : "
              "les phrases de cause sont regroupées par clustering (embeddings, UMAP, HDBSCAN), les groupes sont nommés "
              "par l'agent en thèmes, puis chaque phrase est affectée à un thème par le LLM, ou laissée « aucun ». Les "
              "« aucun » sont reclustérisés au plus "
              f"{MAX_RECLUSTER_ROUNDS} fois ; ce qui reste forme les « causes isolées » (section 9). Les conséquences "
              f"suivent le même chemin, séparément pour chaque segment ({', '.join(labels.values())}). Les composants "
              "bruts sont normalisés par segment. Les tableaux comptent la cause racine ; les causes contributives "
              f"servent à comparer les sites. {dropped} causes contributives ont été écartées faute d'extrait retrouvé. "
              "Les incidents dont le rapport ne donne pas la cause sont exclus et comptés à part.\n",
              "**Origine des causes racines** (part des incidents)\n"]
    origin = (pd.crosstab(ok["cause_type"], ok["asset_type"], normalize="columns") * 100).round(1)
    lines.append(md_table(origin.rename(columns=labels).reset_index().rename(columns={"cause_type": "cause"})))
    cancelled = int(ok["cause_annulee"].notna().sum()) if "cause_annulee" in ok else 0
    lines += [f"\nUne cause explicite ou inférée doit citer un extrait du rapport (`indice`), vérifié par le code : "
              f"{cancelled} causes proposées par le LLM ont été annulées faute d'extrait retrouvé dans le rapport. Les "
              f"inférences de confiance faible sont comptées à part (« {F.UNCERTAIN} »). État de la "
              f"{agreement_text(ok)}."]
    main_runs = [r["summary"] for r in ws.runs.values() if not r["summary"].get("reprise")]
    runs = pd.DataFrame(main_runs)
    runs["choisi"] = runs["run_id"].isin([s["run_id"] for s in ws.selected.values()]).map({True: "✅", False: ""})
    lines += ["\n**Essais de clustering** (choix de l'agent, qualité qualifiée par des seuils fixes)\n",
              md_table(runs[["asset_type", "dimension", "min_cluster_size", "qualite", "conforme_aux_seuils", "choisi"]])]
    for k, choice in ws.selected.items():
        auto = " (choix automatique par le code)" if choice["justification"].startswith("CHOIX AUTOMATIQUE") else ""
        lines.append(f"\n- **{k}** : `{choice['run_id']}` — {ws.runs[choice['run_id']]['summary']['qualite']}{auto}.")
    reprises = [r["summary"] for r in ws.runs.values() if r["summary"].get("reprise")]
    if reprises:
        lines.append("\n**Reprises sur les phrases restées « aucun »** : " + " ; ".join(
            f"`{s['run_id']}` : {s['phrases_reprises']} phrases, {s['nombre_clusters']} clusters" for s in reprises) + ".")
    lines.append("\n**Part de phrases restées « aucun » après affectation** (seuil : "
                 f"{NONE_MAX_PCT:g} %) : " + " ; ".join(
                     f"{labels[a]}, {DIMENSION_LABELS[d]} {ws.none_share(a, d)} %" for a in segments for d in DIMENSIONS) + ".")

    # 3. Thèmes
    lines.append("\n## 3. Thèmes de causes et de conséquences\n")
    for d in DIMENSIONS:
        rows = [{"type": labels[t["asset_type"]], "thème": t["name"], "phrases": t["phrases"],
                 "cohésion": t["cohesion"],
                 "nommé par": "code" if t["justification"].startswith("THÈME AUTOMATIQUE") else "agent",
                 "description": t["description"]} for t in ws.themes.values() if t["dimension"] == d]
        lines += [f"\n**Thèmes de {DIMENSION_LABELS[d]}**\n",
                  md_table(pd.DataFrame(rows).sort_values(["type", "phrases"], ascending=[True, False]))]

    # 4. Causes racines
    lines += ["\n## 4. Causes racines par site\n", PERCENT_NOTE]
    for asset_type in segments:
        sub = ok[ok["asset_type"] == asset_type]
        lines += [f"\n*{labels[asset_type].capitalize()}* — causes non précisées : {F.unknown_share(sub)} % des "
                  "incidents.\n", md_table(percent_table(sub, "theme_cause"))]

    # 5. Heatmap causes x conséquences et chaînes prioritaires (choisies par l'agent)
    lines += ["\n## 5. Heatmap causes × conséquences et chaînes prioritaires\n",
              "Chaque case croise un thème de cause **connue** et un thème de conséquence : couleur = coût total, "
              "chiffre = nombre d'incidents (arrêts de sécurité nominaux exclus). L'agent lit les cases les plus coûteuses "
              "(outil `heatmap_hotspots`) et choisit les chaînes à traiter en priorité ; la gravité de la conséquence peut "
              "faire passer une chaîne avant une autre plus coûteuse."]
    for asset_type in segments:
        choice = ws.priority_chains.get(asset_type)
        lines.append(f"\n### {labels[asset_type].capitalize()}\n\n![Heatmap {asset_type}](heatmap_{asset_type}.png)\n")
        if choice:
            chains = pd.DataFrame(choice["chaines"])
            auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
            chains["coût total"] = chains["cout_total"].map(F.euros)
            chains["part du coût"] = chains["part_cout_pct"].astype(str) + " %"
            chains["maintenance suspectée"] = chains["maintenance_suspectee_pct"].astype(str) + " %"
            lines += [f"**Chaînes prioritaires** ({'choix automatique par le code' if auto else 'choisies par l’agent'})\n",
                      md_table(chains[["cause", "consequence", "incidents", "coût total", "part du coût",
                                       "maintenance suspectée"]].rename(columns={"consequence": "conséquence"})),
                      f"\n*Justification :* {choice['justification']}"]

    # 6. Fiches des causes et criticité Pareto (sans mesure : les mesures sont dans le plan, section 8)
    sheets = pd.DataFrame(report.get("fiches", []))
    if len(sheets):
        lines += ["\n## 6. Fiches des causes et criticité (Pareto)\n",
                  "Thèmes de causes connues triés par coût total (arrêts nominaux exclus). Criticité : part cumulée du coût "
                  "≤ 70 % critique, ≤ 90 % élevée, ≤ 98 % moyenne, sinon faible. Maintenance suspectée : part des incidents "
                  "où le LLM juge, d'après la conséquence, qu'un défaut de maintenance est probable (hypothèse, non vérifiée)."]
        for asset_type in segments:
            sub = sheets[sheets["asset_type"] == asset_type]
            if not len(sub):
                continue
            table = sub.assign(**{"coût total": sub["cout_total"].map(F.euros),
                                  "cumul du coût": sub["cumul_cout_pct"].astype(str) + " %",
                                  "maintenance suspectée": sub["maintenance_suspectee_pct"].astype(str) + " %",
                                  "sites avec mesure": sub["sites_avec_mesure"].map(lambda x: ", ".join(x) or "aucun")})
            lines += [f"\n*{labels[asset_type].capitalize()}*\n",
                      md_table(table[["criticite", "cause", "incidents", "coût total", "cumul du coût",
                                      "consequence_principale", "composant_principal", "maintenance suspectée",
                                      "sites avec mesure"]]
                               .rename(columns={"criticite": "criticité", "consequence_principale": "conséquence principale",
                                                "composant_principal": "composant principal"}))]
            for f in sub[sub["criticite"].isin(["Critique", "Élevée"])].itertuples():
                lines.append(f"\n**{f.cause}** ({f.criticite.lower()}) — {f.description} {f.consequences}")

    # 7. Analyse par site
    lines += ["\n## 7. Analyse par site\n", "**Critères mesurés sur chaque site**\n",
              "Ces critères comptent les incidents dont la cause racine **ou** une cause contributive correspond à "
              "chaque thème (un incident peut compter pour plusieurs critères). La ligne « "
              f"{MAINTENANCE_INDICATOR} » et la part d'incidents les plus graves sont des indicateurs secondaires. "
              "La section 4 range au contraire chaque incident dans une seule cause racine : les deux pourcentages d'un "
              "même phénomène peuvent donc différer.\n"]
    for asset_type in segments:
        lines += [f"\n*{labels[asset_type].capitalize()}*\n",
                  md_table(indicators_table(F.site_indicators(ok[ok["asset_type"] == asset_type], names)))]
    for s in report["sites"]:
        g = ok[ok["site"] == s["site"]]
        mine = [m for m in report["mesures"] if m["site"] == s["site"]]
        lines += [
            f"\n### {s['site']} ({labels[s['asset_type']]})\n",
            f"**Ce qui distingue ce site :** {s['caracterisation']}\n",
            "**Points distinctifs mesurés :** " + ("; ".join(s["points_distinctifs"]) or "aucun") + ".\n",
            f"**Causes dominantes :** {s['causes_dominantes']} *(origine de ces causes : "
            f"{s['origine_causes_dominantes']['explicite']} % explicites, {s['origine_causes_dominantes']['inférée']} % "
            "inférées)*\n",
            f"Chiffres : {len(g)} incidents, causes non précisées {F.unknown_share(g)} %, coût total "
            f"{F.euros(g['estimated_cost_eur'].sum())}, coût moyen {F.euros(g['estimated_cost_eur'].mean())}, arrêt moyen "
            f"{g['downtime_hours'].mean():.1f} h.\n",
            *([f"*Maintenance suspectée (hypothèse du LLM d'après la conséquence, indicateur secondaire) :* "
               f"{F.share(int((g['maintenance_suspectee'] == 'probable').sum()), len(g))} % probable, "
               f"{F.share(int((g['maintenance_suspectee'] == 'plausible').sum()), len(g))} % plausible.\n"]
              if "maintenance_suspectee" in g else []),
            "**Mesures de ce site** (section 8) : " + ("" if mine else "aucune cause connue n'atteint le seuil du plan.\n"),
            *[f"- {m['id']} — {m['cause']} : {m['mesure']}" for m in mine],
            "",
        ]

    # 8. Plan d'actions : une mesure par couple (site, cause), classées par coût par an
    couv = report["couverture"]
    lines += ["\n## 8. Plan d'actions\n",
              f"**{couv['mesures']} mesures couvrent {couv['pct']} % du coût des causes connues hors arrêts de sécurité "
              f"nominaux** (couples site × cause d'au moins {PLAN_MIN_INCIDENTS} incidents, classés par coût par an, jusqu'à "
              f"{PLAN_COVERAGE_PCT:g} % du coût)."
              + (f" Plafond de {PLAN_MAX_MEASURES} mesures atteint : les autres couples ne sont pas traités dans ce plan."
                 if couv["plafond_atteint"] else "")
              + " Une mesure « À définir » n'a pas pu être rédigée à partir des faits : elle reste à compléter par l'équipe. "
                "Le coût par an est la borne haute de l'économie possible si la mesure supprime la cause.\n"]
    if report["mesures"]:
        plan = pd.DataFrame(report["mesures"])
        plan = plan.assign(statut=plan["a_definir"].map({True: "**À définir**", False: "rédigée"}),
                           **{"coût par an": plan["cout_par_an"].map(F.euros), "cumul du coût": plan["cumul_pct"].astype(str) + " %",
                              "portée": plan["portee"]})
        lines.append(md_table(plan[["id", "site", "cause", "portée", "composant_principal", "mesure", "mecanisme",
                                    "coût par an", "cumul du coût", "statut"]]
                              .rename(columns={"composant_principal": "composant", "mecanisme": "mécanisme"})))
    else:
        lines.append("Aucun couple site × cause ne dépasse le seuil d'incidents du plan.")

    # 9. Causes isolées : phrases que l'affectation n'a rattachées à aucun thème
    lines += ["\n## 9. Causes isolées, non regroupées\n",
              f"Phrases de cause que l'affectation n'a rattachées à aucun thème, même après les reprises (au plus "
              f"{MAX_RECLUSTER_ROUNDS}). Elles restent visibles ici, sans thème ni mesure ; une part au-dessus de "
              f"{NONE_MAX_PCT:g} % signale un thème manquant.\n"]
    for asset_type in segments:
        iso = F.isolated_causes(ws.phrases, asset_type)
        lines.append(f"*{labels[asset_type].capitalize()}* — {iso['pct']} % des phrases de cause "
                     f"({iso['phrases']} phrases) :\n")
        lines += [f"- « {e} »" for e in iso["exemples"]] or ["- aucune"]

    # 10. Arrêts de sécurité nominaux : perte structurelle, pas des pannes
    lines += ["\n## 10. Arrêts de sécurité nominaux (perte structurelle)\n",
              f"Incidents de gravité « {c.mildest} » dont la conséquence est un fonctionnement normal des protections. Le "
              f"thème est choisi par l'agent, et le code vérifie qu'au moins {NOMINAL_MIN_MILDEST_PCT:g} % de ses incidents "
              "sont à cette gravité. Ce ne sont pas des pannes : ils sont exclus du plan et de la priorisation, et "
              "reflètent l'exposition du site.\n"]
    for asset_type in segments:
        nominal = ws.nominal.get(asset_type, {"theme_ids": [], "justification": ""})
        names_nom = [ws.themes[t]["name"] for t in nominal["theme_ids"] if t in ws.themes]
        lines.append(f"*{labels[asset_type].capitalize()}* : {', '.join(names_nom) or 'aucun thème nominal'}. "
                     f"*Justification :* {nominal['justification']}\n")
    structural = F.structural_losses(ok)
    if len(structural):
        structural["coût total"] = structural["cout_total"].map(F.euros)
        structural["coût par an"] = structural["cout_par_an"].map(F.euros)
        structural["part des incidents"] = structural["part_incidents_pct"].astype(str) + " %"
        lines += [md_table(structural[["site", "cause_principale", "arrets", "part des incidents", "coût total",
                                       "coût par an"]]), f"\n{NOMINAL_LEVERS}"]

    # 11. Limites (calculées)
    unknown = {labels[a]: F.unknown_share(ok[ok["asset_type"] == a]) for a in segments}
    inferred = round(100 * float((ok["cause_type"] == "inférée").mean()), 1)
    lines += ["\n## 11. Limites\n",
              f"- Causes non précisées dans les rapports : {', '.join(f'{k} {v} %' for k, v in unknown.items())} des "
              "incidents. Ces incidents sont exclus des causes et des chaînes cause → conséquence ; mieux renseigner la "
              "cause dans les rapports d'incident est une recommandation en soi.",
              f"- Causes inférées (déduites d'un extrait du rapport, non écrites explicitement) : {inferred} % des "
              f"incidents ; {agreement_text(ok)}.",
              "- Thèmes et affectation : découverts par clustering puis attribués par le LLM phrase par phrase, sans "
              "vérification manuelle dans cette version. Une phrase peut être mal rangée ; les causes isolées (section 9) "
              "sont celles que l'affectation n'a pas rangées.",
              "- Causes contributives : retenues seulement si leur extrait est retrouvé dans le rapport ; elles servent à "
              "comparer les sites, pas à compter les incidents.",
              "- Le cadrage (domaine, segments, ordre des sévérités) est proposé par le LLM à partir du contexte et d'un "
              "échantillon des données : à relire dans `outputs/1b_cadrage/cadrage.md`.",
              "- Maintenance suspectée : hypothèse du LLM déduite de la conséquence, sans extrait vérifié ; elle "
              "oriente les inspections mais n'entre ni dans les causes ni dans le plan.",
              "- Coûts : estimations des rapports d'incident ; l'économie réelle d'une action est inférieure au coût "
              "observé de la cause.",
              "- Le plan couvre les couples de plus de "
              f"{PLAN_MIN_INCIDENTS} incidents : un petit écart de comptage peut changer le classement des mesures."]
    return "\n".join(lines) + "\n"


def export(ws: AnalysisWorkspace, out_dir: Path) -> None:
    """Résultats chiffrés de la session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {k: {kk: vv for kk, vv in r["summary"].items() if kk != "clusters"} for k, r in ws.runs.items()}
    if summaries:
        pd.DataFrame(summaries.values()).to_csv(out_dir / "clustering_essais.csv", index=False)
    if ws.themes:
        pd.DataFrame([{"theme_id": tid, **t} for tid, t in ws.themes.items()]).to_csv(out_dir / "themes.csv", index=False)
    if len(ws.phrases):
        ws.phrases.to_csv(out_dir / "phrases.csv", index=False)
    table = ws.incidents_table()
    table.drop(columns=["saison"]).to_csv(out_dir / "incidents_analyse.csv", index=False)
    for asset_type in ws.segments:
        if ws.key(asset_type, "cause") in ws.selected and ws.key(asset_type, "consequence") in ws.selected:
            sub = table[table["asset_type"] == asset_type]
            F.chain_stats(sub).to_csv(out_dir / f"heatmap_{asset_type}.csv", index=False)
            heatmap_png(F.heatmap(sub), out_dir / f"heatmap_{asset_type}.png", current().label(asset_type),
                        F.heatmap(sub, "incidents"))
    if ws.report and ws.report.get("fiches"):
        pd.DataFrame(ws.report["fiches"]).drop(columns=["faits"]).to_csv(out_dir / "fiches_causes.csv", index=False)
    if ws.report and ws.report.get("mesures"):
        pd.DataFrame(ws.report["mesures"]).drop(columns=["faits"]).to_csv(out_dir / "mesures.csv", index=False)
    state = {"clustering_choisi": ws.selected, "chaines_prioritaires": ws.priority_chains,
             "colonnes_contexte": ws.context_columns, "nominaux": ws.nominal,
             "isoles": {a: F.isolated_causes(ws.phrases, a) for a in ws.segments}, "composants": ws.components,
             "rapport": ws.report, "essais": summaries}
    (out_dir / "decisions.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def heatmap_png(cost: pd.DataFrame, path: Path, title: str, counts: pd.DataFrame) -> None:
    """Heatmap thèmes de causes x thèmes de conséquences : couleur = coût total, texte = incidents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if cost.empty:
        return
    counts = counts.reindex(index=cost.index, columns=cost.columns, fill_value=0)  # même ordre que les coûts
    short = lambda x: x if len(x) <= 38 else x[:36] + "…"
    fig, ax = plt.subplots(figsize=(1.1 * cost.shape[1] + 4, 0.45 * cost.shape[0] + 2.5))
    image = ax.imshow(cost.to_numpy() / 1000, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), aspect="auto")
    ax.set_xticks(range(cost.shape[1]), [short(c) for c in cost.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(cost.shape[0]), [short(c) for c in cost.index], fontsize=8)
    top = cost.to_numpy().max()
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            n = int(counts.iat[i, j])
            if n:
                ax.text(j, i, n, ha="center", va="center", fontsize=7,
                        color="white" if cost.iat[i, j] > 0.5 * top else "#0b0b0b")
    ax.set_xlabel("Thème de conséquence")
    ax.set_ylabel("Thème de cause")
    ax.set_title(f"{title.capitalize()} : coût total (k€) par cause × conséquence ; chiffre = incidents", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="coût total (k€)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
