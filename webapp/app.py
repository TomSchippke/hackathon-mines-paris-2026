"""Webapp Streamlit : le rapport d'analyse en version interactive, avec graphiques.

Synthèse exécutive (plan d'actions : une mesure par couple site × cause, couverture du coût), Pareto
et fiches diagnostic par cause, heatmap causes x conséquences, sites (ce qui les distingue, leurs mesures,
causes isolées), explorateur d'incidents et vérification.

Les graphiques viennent de figures.py, partagé avec l'export HTML autonome (export_html.py).
Lecture seule des sorties de la pipeline (aucun appel LLM) :
    outputs/2_analyse/      incidents_analyse.csv, fiches_causes.csv, decisions.json, rapport_analyse.md
    outputs/3_verification/ verification.json

Lancement :
    python main.py --steps webapp       # ou : streamlit run webapp/app.py (depuis la racine du repo)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import (F, HIGHLIGHT, METRICS, TIER_COLORS, TIERS, VERDICTS, MissingOutputs, chains_frame,  # noqa: E402
                     euros, fig_cause_consequences, fig_cause_sites, fig_chains, fig_cost_by_site, fig_heatmap,
                     fig_pareto, fig_plan, fig_risk_matrix, fig_site_causes, fig_site_criteria, load_outputs,
                     plan_table, site_measures, theme_names)
from analysis_agent.config import MAINTENANCE_INDICATOR  # noqa: E402
from framing.cadrage import current  # noqa: E402

st.set_page_config(page_title="Incidents | Rapport interactif", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None, str]:
    try:
        return load_outputs()
    except MissingOutputs as e:
        st.error(str(e))
        st.stop()


def asset_picker(key: str) -> str:
    return st.radio("Segment", current().segments, format_func=lambda a: current().label(a).capitalize(),
                    horizontal=True, key=key)


def tier_badge(tier: str) -> str:
    return (f'<span style="background-color:{TIER_COLORS.get(tier, "#64748B")};color:white;padding:3px 10px;'
            f'border-radius:4px;font-weight:600;font-size:0.85rem;">CRITICITÉ : {tier.upper()}</span>')


# ------------------------------------------------------------------------------------------------
# Barre latérale
# ------------------------------------------------------------------------------------------------

def sidebar(incidents: pd.DataFrame, sheets: pd.DataFrame, review: dict | None, report_md: str) -> None:
    st.sidebar.title("⚡ Incident Intelligence")
    st.sidebar.markdown("**Rapport interactif** du workflow agentique : causes et conséquences extraites, "
                        "regroupées en thèmes, croisées dans une heatmap, priorisées par coût.")
    st.sidebar.divider()
    st.sidebar.metric("Incidents analysés", f"{len(incidents):,}".replace(",", " "))
    st.sidebar.metric("Impact financier total", euros(incidents["estimated_cost_eur"].sum()))
    if len(sheets):
        st.sidebar.metric("Thèmes de causes connues", len(sheets))
        st.sidebar.metric("Thèmes critiques (Pareto)", int((sheets["criticite"] == "Critique").sum()))
    if review:
        st.sidebar.metric("Verdict du vérificateur", VERDICTS.get(review["verdict"]["verdict"], review["verdict"]["verdict"]))
    st.sidebar.divider()
    if report_md:
        st.sidebar.download_button("📥 Rapport complet (Markdown)", report_md, "rapport_analyse.md", "text/markdown")
    st.sidebar.caption("Tous les chiffres sont calculés à partir des données (outputs/2_analyse/) ; les textes du LLM "
                       "ont été validés puis relus par un juge.")


# ------------------------------------------------------------------------------------------------
# Onglet 1 : synthèse exécutive
# ------------------------------------------------------------------------------------------------

def tab_executive(incidents: pd.DataFrame, sheets: pd.DataFrame, decisions: dict) -> None:
    report = decisions.get("rapport") or {}
    st.subheader("Vue d'ensemble des sites")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Incidents", f"{len(incidents):,}".replace(",", " "))
    c2.metric("Impact financier", euros(incidents["estimated_cost_eur"].sum()))
    c3.metric("Arrêt cumulé", f"{incidents['downtime_hours'].sum():,.0f} h".replace(",", " "))
    c4.metric("Coût moyen / incident", euros(incidents["estimated_cost_eur"].mean()))
    c5.metric("Causes non précisées", f"{F.unknown_share(incidents)} %",
              help="Rapports qui ne donnent pas la cause : exclus des causes et de la heatmap.")
    st.info(f"**Synthèse.** {report.get('synthese', 'Pas de rapport : l’analyse n’a pas terminé.')}")

    left, right = st.columns([6, 5])
    with left:
        st.markdown("#### Coût par site")
        st.plotly_chart(fig_cost_by_site(incidents), width="stretch")
    with right:
        st.markdown("#### Matrice de risque des causes")
        if len(sheets):
            st.plotly_chart(fig_risk_matrix(sheets), width="stretch")
            st.caption("Une bulle par thème de cause ; taille = arrêt total. En haut à droite : fréquent ET cher.")

    st.subheader("Plan d'actions")
    plan = plan_table(decisions)
    if plan.empty:
        st.info("Pas de plan d'actions : aucun couple site × cause ne dépasse le seuil d'incidents.")
        return
    couv = report["couverture"]
    st.metric("Plan d'actions", f"{couv['mesures']} mesures couvrent {couv['pct']} %")
    st.caption("Un couple (site, cause connue) par mesure, classé par coût par an, hors arrêts de sécurité nominaux "
               "(au moins 5 incidents, jusqu'à 80 % du coût, 12 mesures au plus). Le coût par an est la borne haute de "
               "l'économie possible si la mesure supprime la cause. Une mesure « À définir » reste à rédiger par l'équipe."
               + (" Plafond de 12 mesures atteint : les autres couples ne sont pas traités." if couv["plafond_atteint"] else ""))
    st.plotly_chart(fig_plan(plan), width="stretch")
    shown = plan.assign(**{"coût par an": plan["cout_par_an"].map(euros)}).rename(columns={"portee": "portée"})
    st.dataframe(shown[["id", "site", "cause", "portée", "mesure", "coût par an", "statut"]],
                 hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------
# Onglet 2 : heatmap causes x conséquences
# ------------------------------------------------------------------------------------------------

def tab_heatmap(incidents: pd.DataFrame, decisions: dict) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        asset_type = asset_picker("heatmap_asset")
    sub = incidents[incidents["asset_type"] == asset_type]
    with c2:
        site = st.selectbox("Site", ["Tous les sites", *sorted(sub["site"].unique())], key="heatmap_site")
    with c3:
        metric_label = st.selectbox("Couleur", list(METRICS), key="heatmap_metric")
    if site != "Tous les sites":
        sub = sub[sub["site"] == site]
    fig = fig_heatmap(sub, metric_label)
    if fig is None:
        st.info("Aucune cause connue sur ce périmètre.")
        return
    st.plotly_chart(fig, width="stretch")
    st.caption("Causes connues seulement ; arrêts de sécurité nominaux exclus. Chiffre de chaque case : nombre d'incidents.")

    choice = (decisions.get("chaines_prioritaires") or {}).get(asset_type)
    if choice:
        auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
        st.subheader("Chaînes prioritaires " + ("(choix automatique par le code)" if auto else "(choisies par l'agent)"))
        st.plotly_chart(fig_chains(chains_frame(choice)), width="stretch")
        st.caption("Dans l'ordre de priorité choisi par l'agent.")
        st.markdown(f"*Justification de l'agent :* {choice['justification']}")

    st.subheader("Incidents d'une case")
    stats = F.chain_stats(sub)
    if len(stats):
        pair = st.selectbox("Case (cause → conséquence)", stats["id"],
                            format_func=lambda i: " → ".join(stats.loc[stats["id"] == i, ["cause", "consequence"]].iloc[0]))
        cause_id, consequence_id = pair.split(">")
        cell = sub[(sub["theme_cause_id"] == cause_id) & (sub["theme_consequence_id"] == consequence_id)]
        cols = ["incident_id", "site", "date", "severity", "estimated_cost_eur", "downtime_hours", "cause_racine",
                "consequence", "maintenance_suspectee"]
        st.dataframe(cell[[c for c in cols if c in cell]], hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------
# Onglet 3 : Pareto et fiches diagnostic
# ------------------------------------------------------------------------------------------------

def tab_sheets(incidents: pd.DataFrame, sheets: pd.DataFrame) -> None:
    if sheets.empty:
        st.info("Pas de fiches de causes : relancer l'analyse.")
        return
    asset_type = asset_picker("pareto_asset")
    sub = sheets[sheets["asset_type"] == asset_type].sort_values("rang")
    st.markdown("#### Analyse Pareto des causes (règle 80/20)")
    st.plotly_chart(fig_pareto(sub), width="stretch")
    tier_filter = st.multiselect("Filtrer par criticité", TIERS, default=["Critique", "Élevée"])
    shown = sub[sub["criticite"].isin(tier_filter)] if tier_filter else sub
    table = shown.assign(**{"coût total": shown["cout_total"].map(euros), "part du coût": shown["part_cout_pct"].astype(str) + " %",
                            "maintenance suspectée": shown["maintenance_suspectee_pct"].astype(str) + " %"})
    table = table.assign(**{"sites avec mesure": shown["sites_avec_mesure"]})
    st.dataframe(table[["rang", "criticite", "cause", "incidents", "coût total", "part du coût", "consequence_principale",
                        "maintenance suspectée", "sites avec mesure"]]
                 .rename(columns={"criticite": "criticité", "consequence_principale": "conséquence principale"}),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Fiche diagnostic d'une cause")
    cause = st.selectbox("Cause", sub["cause"],
                         format_func=lambda c: f"[{sub.loc[sub['cause'] == c, 'criticite'].iat[0]}] {c}")
    f = sub[sub["cause"] == cause].iloc[0]
    color = TIER_COLORS.get(f["criticite"], "#64748B")
    st.markdown(f"""<div style="padding:16px 18px;border-radius:8px;border-left:6px solid {color};
        background-color:rgba(128,128,128,0.08);margin-bottom:16px;">
        <h3 style="margin-top:0;">{f['cause']}</h3>{tier_badge(f['criticite'])}
        <span style="margin-left:10px;font-size:0.9rem;opacity:0.8;">Composant principal : {f['composant_principal']}
        · {f['part_cout_pct']} % du coût des causes connues</span></div>""", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidents", int(f["incidents"]))
    m2.metric("Coût total", euros(f["cout_total"]))
    m3.metric("Coût moyen", euros(f["cout_moyen"]))
    m4.metric("Maintenance suspectée", f"{f['maintenance_suspectee_pct']} %",
              help="Part des incidents où le LLM juge, d'après la conséquence, qu'un défaut de maintenance est probable.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔬 Analyse causale")
        st.info(f"**Ce qui se passe :** {f['description']}")
        st.warning(f"**Conséquences :** {f['consequences']}")
    with right:
        st.markdown("#### 🏭 Sites concernés")
        st.success(f"**Sites avec une mesure du plan pour cette cause :** {f['sites_avec_mesure']}")
        st.caption("Le texte de chaque mesure est dans le plan d'actions (synthèse) et dans l'onglet Sites.")

    g = incidents[(incidents["asset_type"] == asset_type) & (incidents["theme_cause"] == cause)]
    left, right = st.columns(2)
    with left:
        fig = fig_cause_consequences(incidents, asset_type, cause)
        if fig is not None:
            st.markdown("##### Conséquences de cette cause")
            st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("##### Répartition par site")
        st.plotly_chart(fig_cause_sites(g), width="stretch")

    st.markdown(f"#### Incidents rattachés à cette cause ({len(g)})")
    cols = ["incident_id", "site", "equipment_id", "date", "severity", "estimated_cost_eur", "downtime_hours",
            "theme_consequence", "cause_type", "maintenance_suspectee"]
    st.dataframe(g[[c for c in cols if c in g]], hide_index=True, width="stretch", height=240)
    if len(g):
        inc = st.selectbox("Rapport d'intervention à lire", g["incident_id"],
                           format_func=lambda x: f"Incident #{x} ({g.loc[g['incident_id'] == x, 'site'].iat[0]})")
        incident_reader(g[g["incident_id"] == inc].iloc[0])


def incident_reader(r: pd.Series) -> None:
    with st.expander("Texte complet du rapport et analyse du LLM", expanded=True):
        st.write(r["text"])
        st.markdown(f"**Cause** ({r['cause_type']}) : {r['cause_racine']}  \n"
                    f"**Indice cité** : {r.get('indice') if pd.notna(r.get('indice')) and r.get('indice') else '—'}  \n"
                    f"**Conséquence** : {r['consequence']}  \n"
                    f"**Maintenance suspectée** : {r.get('maintenance_suspectee', '—')}"
                    + (f" — {r['justification_maintenance']}" if pd.notna(r.get("justification_maintenance"))
                       and r.get("justification_maintenance") else "") + "  \n"
                    f"**Action corrective proposée** : {r['action_corrective']}")
        st.caption(f"Date : {r.get('date', '—')} · Technicien : {r.get('technician_id', '—')} · "
                   f"Format source : {r.get('document_format', '—')}")


# ------------------------------------------------------------------------------------------------
# Onglet 4 : sites (caractérisation, mesures du plan, causes isolées)
# ------------------------------------------------------------------------------------------------

def tab_sites(incidents: pd.DataFrame, decisions: dict) -> None:
    sites = (decisions.get("rapport") or {}).get("sites", [])
    if not sites:
        st.info("Pas de plans par site : l'analyse n'a pas terminé.")
        return
    site = st.selectbox("Site", [s["site"] for s in sites], key="site_plan")
    s = next(x for x in sites if x["site"] == site)
    g = incidents[incidents["site"] == site]
    asset_type = g["asset_type"].iloc[0]
    same_type = incidents[incidents["asset_type"] == asset_type]

    st.markdown(f"""<div style="padding:16px 18px;border-radius:8px;border-left:6px solid {HIGHLIGHT};
        background-color:rgba(128,128,128,0.08);margin-bottom:16px;">
        <h3 style="margin-top:0;">{site}</h3>
        <span style="opacity:0.8;">{current().label(asset_type).capitalize()}</span>
        </div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", len(g))
    c2.metric("Coût total", euros(g["estimated_cost_eur"].sum()))
    c3.metric("Causes non précisées", f"{F.unknown_share(g)} %")
    if "maintenance_suspectee" in g:
        c4.metric("Maintenance suspectée", f"{F.share(int((g['maintenance_suspectee'] == 'probable').sum()), len(g))} %",
                  help="Hypothèse du LLM d'après la conséquence ; indicateur secondaire, hors critères de comparaison.")
    st.markdown(f"**Ce qui distingue ce site :** {s['caracterisation']}")
    if s["points_distinctifs"]:
        st.markdown("**Points distinctifs :**\n" + "\n".join(f"- {p}" for p in s["points_distinctifs"]))
    st.markdown(f"**Causes dominantes :** {s['causes_dominantes']}")

    names = theme_names()
    left, right = st.columns(2)
    with left:
        st.markdown("##### Critères mesurés : ce site face aux autres")
        st.plotly_chart(fig_site_criteria(same_type, site, names), width="stretch")
        st.caption(f"« {MAINTENANCE_INDICATOR} » est un indicateur secondaire. Un thème de cause contributive compte "
                   "parmi les critères du site où il apparaît.")
    with right:
        st.markdown("##### Causes du site")
        st.plotly_chart(fig_site_causes(g), width="stretch")

    st.markdown("#### Mesures du plan pour ce site")
    st.caption("Une mesure par cause connue du site qui figure dans le plan d'actions (voir la synthèse exécutive).")
    mine = site_measures(decisions, site)
    if not mine:
        st.info("Aucune cause connue de ce site n'atteint le seuil du plan.")
    for m in mine:
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**{m['id']} · {m['cause']}** → {m.get('consequence_principale') or '—'}  \n"
                       f"*{str(m['type']).capitalize()} ({m['portee']}) :* {m['mesure']}  \n"
                       f"*Mécanisme :* {m['mecanisme'] or '—'}")
            b.metric("Coût par an", euros(m["cout_par_an"]), f"{m['part_cout_pct']} % du coût des causes connues",
                     delta_color="off")
            if m["a_definir"]:
                st.warning("Mesure à définir : les faits ne permettent pas de la rédiger.")

    with st.expander("Causes isolées et arrêts nominaux, par segment"):
        isolated_section(decisions, names)


def isolated_section(decisions: dict, names: dict[str, str]) -> None:
    """Causes isolées (phrases non rattachées à un thème) et arrêts nominaux, segment par segment."""
    isoles = decisions.get("isoles") or {}
    nominaux = decisions.get("nominaux") or {}
    for a in current().segments:
        iso = isoles.get(a) or {"pct": 0.0, "phrases": 0, "exemples": []}
        st.markdown(f"**{current().label(a).capitalize()}** — causes isolées : {iso['pct']} % des phrases de cause "
                    f"({iso['phrases']} phrases), non regroupées par l'agent.")
        st.markdown("\n".join(f"- « {e} »" for e in iso["exemples"]) or "- aucune")
        nom = nominaux.get(a) or {"theme_ids": [], "justification": ""}
        names_nom = [names.get(t, t) for t in nom["theme_ids"]]
        st.markdown(f"*Arrêts nominaux :* {', '.join(names_nom) or 'aucun thème'}. *Justification :* {nom['justification']}")


# ------------------------------------------------------------------------------------------------
# Onglet 5 : explorateur et vérification
# ------------------------------------------------------------------------------------------------

def tab_explorer(incidents: pd.DataFrame, review: dict | None) -> None:
    st.subheader("Explorateur & recherche d'incidents")
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("🔎 Recherche dans les rapports")
    assets = c2.multiselect("Segment", current().segments, default=current().segments, format_func=current().label)
    sites = c3.multiselect("Site", sorted(incidents["site"].unique()))
    types = c4.multiselect("Type de cause", sorted(incidents["cause_type"].dropna().unique()))
    df = incidents[incidents["asset_type"].isin(assets)]
    if query.strip():
        df = df[df["text"].str.contains(query.strip(), case=False, na=False, regex=False)]
    if sites:
        df = df[df["site"].isin(sites)]
    if types:
        df = df[df["cause_type"].isin(types)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Résultats", f"{len(df)} / {len(incidents)}")
    m2.metric("Impact financier", euros(df["estimated_cost_eur"].sum()))
    m3.metric("Arrêt cumulé", f"{df['downtime_hours'].sum():,.0f} h".replace(",", " "))
    m4.metric("Thèmes de causes touchés", df.loc[~df["theme_cause"].isin(F.NOT_A_CAUSE), "theme_cause"].nunique())
    cols = ["incident_id", "site", "date", "severity", "estimated_cost_eur", "theme_cause", "cause_type",
            "theme_consequence", "maintenance_suspectee"]
    st.dataframe(df[[c for c in cols if c in df]], hide_index=True, width="stretch", height=320)
    st.download_button("📥 Télécharger la sélection (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "incidents_selection.csv", "text/csv")
    if len(df):
        inc = st.selectbox("Incident à lire", df["incident_id"])
        incident_reader(df[df["incident_id"] == inc].iloc[0])

    st.divider()
    st.subheader("Vérification")
    if review is None:
        st.info("Pas de vérification : lancer `python main.py --steps verification`.")
        return
    verdict = review["verdict"]
    st.markdown(f"**Verdict : {VERDICTS.get(verdict['verdict'], verdict['verdict'])}** — {verdict['synthese']}")
    checks = pd.DataFrame(review["controles"])
    st.caption(f"Contrôles automatiques : {int(checks['ok'].sum())} réussis sur {len(checks)}.")
    checks["ok"] = checks["ok"].map({True: "✅ ok", False: "⛔ échec"})
    st.dataframe(checks, hide_index=True, width="stretch")
    findings = pd.DataFrame(review["constats"])
    if len(findings):
        problems = findings[findings["gravite"] != "conforme"]
        st.markdown(f"**Constats du vérificateur** : {len(problems)} problèmes, {len(findings) - len(problems)} points conformes")
        st.dataframe(findings, hide_index=True, width="stretch")


def main() -> None:
    incidents, sheets, decisions, review, report_md = load()
    sidebar(incidents, sheets, review, report_md)
    st.title("Rapport d'analyse des incidents")
    st.caption(current().domain)
    st.markdown("Causes et conséquences de chaque incident extraites par le LLM, regroupées en thèmes par l'agent, "
                "croisées dans une heatmap et priorisées par coût : de la synthèse au plan d'actions.")
    tabs = st.tabs(["📊 Synthèse exécutive", "🔥 Heatmap causes × conséquences", "🔍 Pareto & fiches diagnostic",
                    "🏭 Sites & mesures", "📋 Explorateur & vérification"])
    with tabs[0]:
        tab_executive(incidents, sheets, decisions)
    with tabs[1]:
        tab_heatmap(incidents, decisions)
    with tabs[2]:
        tab_sheets(incidents, sheets)
    with tabs[3]:
        tab_sites(incidents, decisions)
    with tabs[4]:
        tab_explorer(incidents, review)


main()
