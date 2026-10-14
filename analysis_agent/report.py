"""Mise en forme du rapport d'analyse et export des résultats.

Tous les chiffres et tableaux viennent de facts.py (calculés en Python). Les seuls textes rédigés
par le LLM (justification des profils, causes dominantes, actions, synthèse) ont été validés par
writing.py, ou remplacés par du texte Python. Les pourcentages d'un site sont toujours exprimés en
part de l'ensemble de ses incidents.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from llm.client import deployment

from . import facts as F
from .config import (ASSET_LABELS, ASSET_TYPES, DIMENSION_LABELS, DIMENSIONS, MAINTENANCE_INDICATOR, NOMINAL_LEVERS,
                     SITE_PROFILES)
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
    incidents = ws.incidents_table()
    ok = incidents[incidents["erreur_analyse"] == ""]
    lines = [
        "# Rapport d'analyse des incidents PV et éolien",
        f"\n*Généré le {date.today():%d/%m/%Y}. Agent d'analyse : `{deployment('analysis')}` ; rédaction : "
        f"`{deployment('writing')}`. Tous les chiffres sont calculés à partir des données ; les textes rédigés par le "
        "LLM ont été vérifiés par un validateur (chiffres et phénomènes cités), puis relus par un juge dont les "
        f"corrections ne sont acceptées que si elles passent ce validateur ({report.get('juge', {}).get('corriges', 0)} "
        f"corrections acceptées, {report.get('juge', {}).get('refuses', 0)} refusées). Déroulé complet : `journal.md`.*",
        "\n## 1. Synthèse\n", report["synthese"],
    ]

    # 2. Données et méthode
    lines += ["\n## 2. Données et méthode\n",
              f"{len(incidents)} incidents ({incidents['date'].min()} à {incidents['date'].max()}), "
              f"{incidents['site'].nunique()} sites. Pour chaque incident, le LLM extrait sa **cause** (pourquoi) et sa "
              "**conséquence** (ce qu'il a provoqué). Les causes sont regroupées par clustering (embeddings, UMAP, "
              "HDBSCAN, selon la méthode BERTopic), de même que les conséquences, séparément pour le photovoltaïque et "
              "l'éolien. Les incidents dont le rapport ne donne pas la cause sont exclus du clustering des causes et "
              "comptés à part.\n", "**Origine des causes** (part des incidents)\n"]
    origin = (pd.crosstab(ok["cause_type"], ok["asset_type"], normalize="columns") * 100).round(1)
    lines.append(md_table(origin.rename(columns=ASSET_LABELS).reset_index().rename(columns={"cause_type": "cause"})))
    cancelled = int(ok["cause_annulee"].notna().sum()) if "cause_annulee" in ok else 0
    lines += [f"\nUne cause explicite ou inférée doit citer un extrait du rapport (`indice`), vérifié par le code : "
              f"{cancelled} causes proposées par le LLM ont été annulées faute d'extrait retrouvé dans le rapport. Les "
              f"inférences de confiance faible sont comptées à part (« {F.UNCERTAIN} »). État de la "
              f"{agreement_text(ok)}."]
    runs = pd.DataFrame([r["summary"] for r in ws.runs.values()])
    runs["choisi"] = runs["run_id"].isin([s["run_id"] for s in ws.selected.values()]).map({True: "✅", False: ""})
    lines += ["\n**Essais de clustering** (choix de l'agent, qualité qualifiée par des seuils fixes)\n",
              md_table(runs[["asset_type", "dimension", "min_cluster_size", "qualite", "conforme_aux_seuils", "choisi"]])]
    for k, choice in ws.selected.items():
        auto = " (choix automatique par le code)" if choice["justification"].startswith("CHOIX AUTOMATIQUE") else ""
        lines.append(f"\n- **{k}** : `{choice['run_id']}` — {ws.runs[choice['run_id']]['summary']['qualite']}{auto}.")

    # 3. Thèmes
    lines.append("\n## 3. Thèmes de causes et de conséquences\n")
    for d in DIMENSIONS:
        rows = [{"type": ASSET_LABELS[t["asset_type"]], "thème": t["name"], "incidents": t["incidents"],
                 "cohésion": t["cohesion"],
                 "nommé par": "code" if t["justification"].startswith("THÈME AUTOMATIQUE") else "agent",
                 "description": t["description"]} for t in ws.themes.values() if t["dimension"] == d]
        lines += [f"\n**Thèmes de {DIMENSION_LABELS[d]}**\n",
                  md_table(pd.DataFrame(rows).sort_values(["type", "incidents"], ascending=[True, False]))]

    # 4. Causes racines
    lines += ["\n## 4. Causes racines par site\n", PERCENT_NOTE]
    for asset_type in ASSET_TYPES:
        sub = ok[ok["asset_type"] == asset_type]
        lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()}* — causes non précisées : {F.unknown_share(sub)} % des "
                  "incidents.\n", md_table(percent_table(sub, "theme_cause"))]

    # 5. Heatmap causes x conséquences et chaînes prioritaires (choisies par l'agent)
    lines += ["\n## 5. Heatmap causes × conséquences et chaînes prioritaires\n",
              "Chaque case croise un thème de cause **connue** et un thème de conséquence : couleur = coût total, "
              "chiffre = nombre d'incidents (arrêts de sécurité nominaux exclus). L'agent lit les cases les plus coûteuses "
              "(outil `heatmap_hotspots`) et choisit les chaînes à traiter en priorité ; la gravité de la conséquence peut "
              "faire passer une chaîne avant une autre plus coûteuse."]
    for asset_type in ASSET_TYPES:
        choice = ws.priority_chains.get(asset_type)
        lines.append(f"\n### {ASSET_LABELS[asset_type].capitalize()}\n\n![Heatmap {asset_type}](heatmap_{asset_type}.png)\n")
        if choice:
            chains = pd.DataFrame(choice["chaines"])
            auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
            chains["coût total"] = chains["cout_total"].map(F.euros)
            chains["part du coût"] = chains["part_cout_pct"].astype(str) + " %"
            chains["maintenance suspectée"] = chains["maintenance_suspectee_pct"].astype(str) + " %"
            lines += [f"**Chaînes prioritaires** ({"choix automatique par le code" if auto else "choisies par l'agent"})\n",
                      md_table(chains[["cause", "consequence", "incidents", "coût total", "part du coût",
                                       "maintenance suspectée"]].rename(columns={"consequence": "conséquence"})),
                      f"\n*Justification :* {choice['justification']}"]

    # 6. Fiches des causes et criticité Pareto
    sheets = pd.DataFrame(report.get("fiches", []))
    if len(sheets):
        lines += ["\n## 6. Fiches des causes et criticité (Pareto)\n",
                  "Thèmes de causes connues triés par coût total (arrêts nominaux exclus). Criticité : part cumulée du coût "
                  "≤ 70 % critique, ≤ 90 % élevée, ≤ 98 % moyenne, sinon faible. Maintenance suspectée : part des incidents "
                  "où le LLM juge, d'après la conséquence, qu'un défaut de maintenance est probable (hypothèse, non vérifiée)."]
        for asset_type in ASSET_TYPES:
            sub = sheets[sheets["asset_type"] == asset_type]
            if not len(sub):
                continue
            table = sub.assign(**{"coût total": sub["cout_total"].map(F.euros),
                                  "cumul du coût": sub["cumul_cout_pct"].astype(str) + " %",
                                  "maintenance suspectée": sub["maintenance_suspectee_pct"].astype(str) + " %"})
            lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()}*\n",
                      md_table(table[["criticite", "cause", "incidents", "coût total", "cumul du coût",
                                      "consequence_principale", "maintenance suspectée", "mesure"]]
                               .rename(columns={"criticite": "criticité", "consequence_principale": "conséquence principale",
                                                "mesure": "mesure préventive"}))]
            for f in sub[sub["criticite"].isin(["Critique", "Élevée"])].itertuples():
                lines.append(f"\n**{f.cause}** ({f.criticite.lower()}) — {f.description} {f.consequences} "
                             f"*Mesure ({f.type}) :* {f.mesure} *Mécanisme :* {f.mecanisme}")

    # 7. Analyse par site
    lines += ["\n## 7. Analyse par site\n", "**Critères des profils mesurés sur chaque site**\n",
              "Ces critères comptent les incidents dont le facteur principal **ou secondaire** extrait par le LLM "
              "correspond (un incident peut compter pour plusieurs critères). La ligne « "
              f"{MAINTENANCE_INDICATOR} » est un indicateur secondaire : elle n'entre pas dans l'attribution des profils. "
              "La section 4 range au contraire chaque "
              "incident dans un seul thème de cause : les deux pourcentages d'un même phénomène peuvent donc différer "
              "de quelques points.\n"]
    for asset_type in ASSET_TYPES:
        lines += [f"\n*{ASSET_LABELS[asset_type].capitalize()}*\n",
                  md_table(indicators_table(F.profile_indicators(ok[ok["asset_type"] == asset_type])))]
        explicit, n = F.explicit_only_profiles(ok, asset_type)
        chosen = {s["site"]: s["profil"] for s in report["sites"] if s["site"] in ws.sites[asset_type]}
        verdict = ("**identique** : l'association ne dépend pas des causes inférées" if chosen in explicit else
                   f"**différente** ({explicit}) : l'association dépend des causes inférées, à confirmer")
        lines.append(f"\nTest de robustesse ({ASSET_LABELS[asset_type]}) : avec les seules causes explicites ({n} "
                     f"incidents), la meilleure association est {verdict}.")
    for s in report["sites"]:
        g = ok[ok["site"] == s["site"]]
        asset_type = g["asset_type"].iloc[0]
        indicators = F.profile_indicators(ok[ok["asset_type"] == asset_type])
        gaps = F.profile_gaps(indicators, s["site"], asset_type, s["profil"])
        auto = ws.profiles[s["site"]]["raisonnement"].startswith("ATTRIBUTION AUTOMATIQUE")
        plan = pd.DataFrame(s["plan"])
        lines += [
            f"\n### {s['site']} — profil {s['profil']}{' (attribution automatique)' if auto else ''}\n",
            f"*Profil {s['profil']} ({ASSET_LABELS[asset_type]}) : {SITE_PROFILES[asset_type][s['profil']]}.*\n",
            f"**Pourquoi ce profil :** {s['justification_profil']}\n",
            "**Points distinctifs mesurés :** " + ("; ".join(s["points_distinctifs"]) or "aucun") + ".\n",
            "**Ce qui ne colle pas avec le profil :** " + ("; ".join(gaps) if gaps else "rien de notable") + ".\n",
            f"**Causes dominantes :** {s['causes_dominantes']} *(origine de ces causes : "
            f"{s['origine_causes_dominantes']['explicite']} % explicites, {s['origine_causes_dominantes']['inférée']} % "
            "inférées)*\n",
            f"Chiffres : {len(g)} incidents, causes non précisées {F.unknown_share(g)} %, coût total "
            f"{F.euros(g['estimated_cost_eur'].sum())}, coût moyen {F.euros(g['estimated_cost_eur'].mean())}, arrêt moyen "
            f"{g['downtime_hours'].mean():.1f} h.\n",
            *([f"*Maintenance suspectée (hypothèse du LLM d'après la conséquence, indicateur secondaire) :* "
               f"{F.share(int((g['maintenance_suspectee'] == 'probable').sum()), len(g))} % probable, "
               f"{F.share(int((g['maintenance_suspectee'] == 'plausible').sum()), len(g))} % plausible ; défaut de "
               f"maintenance écrit dans le rapport : {F.share(int(g['lie_maintenance'].astype(str).str.lower().isin(['true', '1']).sum()), len(g))} %.\n"]
              if "maintenance_suspectee" in g else []),
            "**Plan de remédiation** : mesures préventives pour les causes connues les plus coûteuses, hors arrêts de "
            "sécurité nominaux ; priorité selon leur part du coût du site.\n",
            md_table(plan.assign(**{"coût total": plan["cout_total"].map(F.euros),
                                    "part du coût du site": plan["part_cout_pct"].astype(str) + " %"})
                     [["priorite", "cause", "consequence_principale", "type", "action", "mecanisme", "incidents",
                       "coût total", "part du coût du site"]]
                     .rename(columns={"action": "mesure préventive", "mecanisme": "mécanisme",
                                      "consequence_principale": "conséquence principale"})),
        ]

    # 8. Arrêts de sécurité nominaux : perte structurelle, pas des pannes
    structural = F.structural_losses(ok)
    if len(structural):
        structural["coût total"] = structural["cout_total"].map(F.euros)
        structural["coût par an"] = structural["cout_par_an"].map(F.euros)
        structural["part des incidents"] = structural["part_incidents_pct"].astype(str) + " %"
        lines += ["\n## 8. Arrêts de sécurité nominaux (perte structurelle)\n",
                  "Arrêts automatiques de gravité mineure : la protection a fonctionné comme prévu (par exemple une "
                  "éolienne arrêtée au-delà de la vitesse de vent de coupure). Ce ne sont pas des pannes : ils sont "
                  "exclus des plans de remédiation et de la priorisation, et reflètent l'exposition du site.\n",
                  md_table(structural[["site", "cause_principale", "arrets", "part des incidents", "coût total",
                                       "coût par an"]]), f"\n{NOMINAL_LEVERS}"]

    # 9. Priorisation à l'échelle du parc
    fleet = F.fleet_priorities(ok)
    actions = {(r["site"], p["cause"]): p["action"] for r in report["sites"] for p in r["plan"]}
    fleet["action"] = [actions.get((r.site, r.cause), "à définir (hors des 3 causes prioritaires du site)")
                       for r in fleet.itertuples()]
    fleet["coût total"] = fleet["cout_total"].map(F.euros)
    fleet["coût par an"] = fleet["cout_par_an"].map(F.euros)
    lines += ["\n## 9. Priorisation à l'échelle du parc\n",
              f"Couples (site, cause connue) au plus fort coût observé sur {F.years(ok):.1f} ans, hors arrêts de "
              "sécurité nominaux. Le coût par an est la borne haute de l'économie possible si la mesure supprime la cause.\n",
              md_table(fleet[["site", "cause", "incidents", "coût total", "coût par an", "action"]]
                       .rename(columns={"action": "mesure préventive"}))]

    # 10. Limites (calculées)
    unknown = {ASSET_LABELS[a]: F.unknown_share(ok[ok["asset_type"] == a]) for a in ASSET_TYPES}
    inferred = round(100 * float((ok["cause_type"] == "inférée").mean()), 1)
    lines += ["\n## 10. Limites\n",
              f"- Causes non précisées dans les rapports : {', '.join(f'{k} {v} %' for k, v in unknown.items())} des "
              "incidents. Ces incidents sont exclus des causes et des chaînes cause → conséquence ; mieux renseigner la "
              "cause dans les rapports d'incident est une recommandation en soi.",
              f"- Causes inférées (déduites d'un extrait du rapport, non écrites explicitement) : {inferred} % des "
              f"incidents ; {agreement_text(ok)}.",
              "- Les profils sont attribués par comparaison des sites entre eux : un écart au profil (section 6) peut "
              "venir des données comme de l'attribution.",
              "- Maintenance suspectée : hypothèse du LLM déduite de la conséquence, sans extrait vérifié ; elle "
              "oriente les inspections mais n'entre ni dans les causes ni dans l'attribution des profils.",
              "- Coûts : estimations des rapports d'incident ; l'économie réelle d'une action est inférieure au coût "
              "observé de la cause."]
    return "\n".join(lines) + "\n"


def export(ws: AnalysisWorkspace, out_dir: Path) -> None:
    """Résultats chiffrés de la session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {k: {kk: vv for kk, vv in r["summary"].items() if kk != "clusters"} for k, r in ws.runs.items()}
    if summaries:
        pd.DataFrame(summaries.values()).to_csv(out_dir / "clustering_essais.csv", index=False)
    if ws.themes:
        pd.DataFrame([{"theme_id": tid, **t} for tid, t in ws.themes.items()]).to_csv(out_dir / "themes.csv", index=False)
    ws.incidents_table().drop(columns=["saison"]).to_csv(out_dir / "incidents_analyse.csv", index=False)
    table = ws.incidents_table()
    for asset_type in ASSET_TYPES:
        if ws.key(asset_type, "cause") in ws.selected and ws.key(asset_type, "consequence") in ws.selected:
            sub = table[table["asset_type"] == asset_type]
            F.chain_stats(sub).to_csv(out_dir / f"heatmap_{asset_type}.csv", index=False)
            heatmap_png(F.heatmap(sub), out_dir / f"heatmap_{asset_type}.png", ASSET_LABELS[asset_type],
                        F.heatmap(sub, "incidents"))
    if ws.report and ws.report.get("fiches"):
        pd.DataFrame(ws.report["fiches"]).drop(columns=["faits"]).to_csv(out_dir / "fiches_causes.csv", index=False)
    state = {"clustering_choisi": ws.selected, "profils": ws.profiles, "chaines_prioritaires": ws.priority_chains,
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
