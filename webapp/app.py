"""Webapp Streamlit : rapport actionnable, heatmap causes x conséquences, criticité Pareto des causes.

Lecture seule des sorties de la pipeline (aucun appel LLM) :
    outputs/2_analyse/      incidents_analyse.csv, fiches_causes.csv, decisions.json, rapport_analyse.md
    outputs/3_verification/ verification.json

Lancement :
    python main.py --steps webapp       # ou : streamlit run webapp/app.py (depuis la racine du repo)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis_agent import facts as F  # noqa: E402
from analysis_agent.config import ASSET_LABELS, ASSET_TYPES, SITE_PROFILES  # noqa: E402

ANALYSIS = ROOT / "outputs" / "2_analyse"
REVIEW = ROOT / "outputs" / "3_verification"

# Criticité : palette de statut (toujours accompagnée de son libellé, jamais la couleur seule).
TIER_COLORS = {"Critique": "#d03b3b", "Élevée": "#ec835a", "Moyenne": "#fab219", "Faible": "#0ca30c"}
TIERS = list(TIER_COLORS)
# Rampe séquentielle bleue (une teinte, clair -> foncé) pour la heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METRICS = {"Coût total (€)": "cout_total", "Nombre d'incidents": "incidents", "Arrêt total (h)": "arret_total_h"}

st.set_page_config(page_title="Incidents PV & éolien | Rapport actionnable", page_icon="⚡", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None, str]:
    path = ANALYSIS / "incidents_analyse.csv"
    if not path.exists():
        st.error(f"{path} introuvable : lancer d'abord `python main.py --steps analyse`.")
        st.stop()
    incidents = pd.read_csv(path)
    for col in ("estimated_cost_eur", "downtime_hours"):
        incidents[col] = pd.to_numeric(incidents[col], errors="coerce").fillna(0.0)
    for col in ("theme_cause", "theme_consequence"):
        incidents[col] = incidents[col].fillna(F.UNCLASSIFIED)
    sheets_path = ANALYSIS / "fiches_causes.csv"
    sheets = pd.read_csv(sheets_path) if sheets_path.exists() else pd.DataFrame()
    decisions = json.loads((ANALYSIS / "decisions.json").read_text(encoding="utf-8"))
    review_path = REVIEW / "verification.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
    report_path = ANALYSIS / "rapport_analyse.md"
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return incidents, sheets, decisions, review, report_md


def euros(x: float) -> str:
    return F.euros(x)


def asset_picker(key: str) -> str:
    return st.radio("Type d'installation", ASSET_TYPES, format_func=lambda a: ASSET_LABELS[a].capitalize(),
                    horizontal=True, key=key)


# ------------------------------------------------------------------------------------------------
# Onglet 1 : rapport actionnable
# ------------------------------------------------------------------------------------------------

def tab_report(incidents: pd.DataFrame, decisions: dict, report_md: str) -> None:
    report = decisions.get("rapport") or {}
    st.subheader("Synthèse")
    st.write(report.get("synthese", "Pas de rapport : l'analyse n'a pas terminé."))
    if report_md:
        st.download_button("📥 Télécharger le rapport complet (Markdown)", report_md, "rapport_analyse.md", "text/markdown")

    st.subheader("Priorités du parc")
    st.caption("Couples (site, cause connue) au plus fort coût, hors arrêts de sécurité nominaux. Le coût par an "
               "est la borne haute de l'économie possible si la mesure supprime la cause.")
    fleet = F.fleet_priorities(incidents)
    actions = {(s["site"], p["cause"]): p["action"] for s in report.get("sites", []) for p in s["plan"]}
    fleet["mesure préventive"] = [actions.get((r.site, r.cause), "hors des 3 causes prioritaires du site")
                                  for r in fleet.itertuples()]
    fleet["coût total"] = fleet["cout_total"].map(euros)
    fleet["coût par an"] = fleet["cout_par_an"].map(euros)
    st.dataframe(fleet[["site", "cause", "incidents", "coût total", "coût par an", "mesure préventive"]],
                 hide_index=True, width="stretch")

    st.subheader("Plan par site")
    sites = report.get("sites", [])
    if not sites:
        return
    site = st.selectbox("Site", [s["site"] for s in sites])
    s = next(x for x in sites if x["site"] == site)
    g = incidents[incidents["site"] == site]
    asset_type = g["asset_type"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profil attribué", s["profil"])
    c2.metric("Incidents", len(g))
    c3.metric("Coût total", euros(g["estimated_cost_eur"].sum()))
    if "maintenance_suspectee" in g:
        c4.metric("Maintenance suspectée (probable)", f"{F.share(int((g['maintenance_suspectee'] == 'probable').sum()), len(g))} %",
                  help="Hypothèse du LLM déduite de la conséquence ; indicateur secondaire, hors attribution des profils.")
    st.caption(f"Profil {s['profil']} : {SITE_PROFILES[asset_type][s['profil']]}")
    st.markdown(f"**Pourquoi ce profil :** {s['justification_profil']}")
    st.markdown(f"**Causes dominantes :** {s['causes_dominantes']}")
    plan = pd.DataFrame(s["plan"])
    if len(plan):
        plan["coût total"] = plan["cout_total"].map(euros)
        plan["part du coût du site"] = plan["part_cout_pct"].astype(str) + " %"
        cols = ["priorite", "cause", "consequence_principale", "type", "action", "mecanisme", "incidents", "coût total",
                "part du coût du site"]
        st.dataframe(plan[[c for c in cols if c in plan]].rename(columns={
            "priorite": "priorité", "consequence_principale": "conséquence principale", "action": "mesure préventive",
            "mecanisme": "mécanisme"}), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------
# Onglet 2 : heatmap causes x conséquences
# ------------------------------------------------------------------------------------------------

def tab_heatmap(incidents: pd.DataFrame, decisions: dict) -> None:
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        asset_type = asset_picker("heatmap_asset")
    sub = incidents[incidents["asset_type"] == asset_type]
    with c2:
        site = st.selectbox("Site", ["Tous les sites", *sorted(sub["site"].unique())], key="heatmap_site")
    with c3:
        metric_label = st.selectbox("Couleur", list(METRICS), key="heatmap_metric")
    if site != "Tous les sites":
        sub = sub[sub["site"] == site]
    matrix = F.heatmap(sub, METRICS[metric_label])
    if matrix.empty:
        st.info("Aucune cause connue sur ce périmètre.")
        return
    counts = F.heatmap(sub, "incidents").reindex(index=matrix.index, columns=matrix.columns, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        text=counts.to_numpy(), texttemplate="%{text}", textfont={"size": 11},
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        colorbar={"title": metric_label}, xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b> → %{x}<br>" + metric_label + " : %{z:,.0f}<br>incidents : %{text}<extra></extra>"))
    fig.update_layout(height=max(380, 34 * len(matrix) + 160), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis={"title": "Thème de conséquence", "tickangle": -35}, yaxis={"title": "Thème de cause",
                                                                                        "autorange": "reversed"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Causes connues seulement ; arrêts de sécurité nominaux exclus. Chiffre de chaque case : nombre d'incidents.")

    choice = (decisions.get("chaines_prioritaires") or {}).get(asset_type)
    if choice:
        auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
        st.subheader("Chaînes prioritaires " + ("(choix automatique par le code)" if auto else "(choisies par l'agent)"))
        chains = pd.DataFrame(choice["chaines"])
        chains["coût total"] = chains["cout_total"].map(euros)
        chains["part du coût"] = chains["part_cout_pct"].astype(str) + " %"
        chains["maintenance suspectée"] = chains["maintenance_suspectee_pct"].astype(str) + " %"
        st.dataframe(chains[["cause", "consequence", "incidents", "coût total", "part du coût", "maintenance suspectée"]]
                     .rename(columns={"consequence": "conséquence"}), hide_index=True, width="stretch")
        st.markdown(f"*Justification :* {choice['justification']}")

    st.subheader("Incidents d'une case")
    stats = F.chain_stats(sub)
    if len(stats):
        pair = st.selectbox("Case (cause → conséquence)", stats["id"],
                            format_func=lambda i: " → ".join(stats.loc[stats["id"] == i, ["cause", "consequence"]].iloc[0]))
        cause_id, consequence_id = pair.split(">")
        cell = sub[(sub["theme_cause_id"] == cause_id) & (sub["theme_consequence_id"] == consequence_id)]
        st.dataframe(cell[["incident_id", "site", "date", "severity", "estimated_cost_eur", "downtime_hours",
                           "cause_racine", "consequence", "maintenance_suspectee"]
                          if "maintenance_suspectee" in cell else
                          ["incident_id", "site", "date", "severity", "estimated_cost_eur", "downtime_hours",
                           "cause_racine", "consequence"]], hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------
# Onglet 3 : criticité Pareto et fiches des causes
# ------------------------------------------------------------------------------------------------

def tab_pareto(sheets: pd.DataFrame) -> None:
    if sheets.empty:
        st.info("Pas de fiches de causes : relancer l'analyse.")
        return
    asset_type = asset_picker("pareto_asset")
    sub = sheets[sheets["asset_type"] == asset_type].sort_values("rang")
    c1, c2, c3 = st.columns(3)
    c1.metric("Thèmes de causes connues", len(sub))
    c2.metric("Thèmes critiques", int((sub["criticite"] == "Critique").sum()),
              help="Thèmes qui concentrent les 70 premiers % du coût (le premier est toujours critique).")
    c3.metric("Coût des causes connues", euros(sub["cout_total"].sum()))

    # Une seule échelle : coût total par cause, part cumulée en étiquette (pas de double axe).
    fig = px.bar(sub.iloc[::-1], x="cout_total", y="cause", orientation="h", color="criticite",
                 color_discrete_map=TIER_COLORS, category_orders={"criticite": TIERS},
                 text=sub.iloc[::-1]["cumul_cout_pct"].map(lambda v: f"cumul {v:.0f} %"),
                 labels={"cout_total": "Coût total (€)", "cause": "", "criticite": "Criticité"},
                 hover_data={"incidents": True, "part_cout_pct": True, "consequence_principale": True})
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(height=max(320, 30 * len(sub) + 120), margin={"l": 10, "r": 40, "t": 20, "b": 10},
                      bargap=0.25, legend={"orientation": "h", "y": 1.08})
    st.plotly_chart(fig, width="stretch")

    table = sub.assign(**{"coût total": sub["cout_total"].map(euros), "part du coût": sub["part_cout_pct"].astype(str) + " %",
                          "maintenance suspectée": sub["maintenance_suspectee_pct"].astype(str) + " %"})
    st.dataframe(table[["rang", "criticite", "cause", "incidents", "coût total", "part du coût", "consequence_principale",
                        "maintenance suspectée"]].rename(columns={"criticite": "criticité",
                                                                  "consequence_principale": "conséquence principale"}),
                 hide_index=True, width="stretch")

    st.subheader("Fiche d'une cause")
    cause = st.selectbox("Cause", sub["cause"], format_func=lambda c: f"[{sub.loc[sub['cause'] == c, 'criticite'].iat[0]}] {c}")
    f = sub[sub["cause"] == cause].iloc[0]
    st.markdown(f"**{f['cause']}** — criticité **{f['criticite']}** · facteur dominant : {f['facteur'] or 'non précisé'}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidents", int(f["incidents"]))
    m2.metric("Coût total", euros(f["cout_total"]))
    m3.metric("Coût moyen", euros(f["cout_moyen"]))
    m4.metric("Maintenance suspectée", f"{f['maintenance_suspectee_pct']} %")
    left, right = st.columns(2)
    with left:
        st.info(f"**Ce qui se passe :** {f['description']}")
        st.warning(f"**Conséquences :** {f['consequences']}")
    with right:
        st.success(f"**Mesure préventive ({f['type']}) :** {f['mesure']}\n\n**Mécanisme :** {f['mecanisme']}")


# ------------------------------------------------------------------------------------------------
# Onglet 4 : explorateur et vérification
# ------------------------------------------------------------------------------------------------

def tab_explorer(incidents: pd.DataFrame, review: dict | None) -> None:
    st.subheader("Explorateur d'incidents")
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("Recherche dans les rapports")
    assets = c2.multiselect("Type", ASSET_TYPES, default=ASSET_TYPES, format_func=lambda a: ASSET_LABELS[a])
    sites = c3.multiselect("Site", sorted(incidents["site"].unique()))
    types = c4.multiselect("Type de cause", sorted(incidents["cause_type"].dropna().unique()))
    df = incidents[incidents["asset_type"].isin(assets)]
    if query.strip():
        df = df[df["text"].str.contains(query.strip(), case=False, na=False, regex=False)]
    if sites:
        df = df[df["site"].isin(sites)]
    if types:
        df = df[df["cause_type"].isin(types)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Incidents", f"{len(df)} / {len(incidents)}")
    m2.metric("Coût", euros(df["estimated_cost_eur"].sum()))
    m3.metric("Arrêt cumulé", f"{df['downtime_hours'].sum():,.0f} h".replace(",", " "))
    cols = ["incident_id", "site", "date", "severity", "estimated_cost_eur", "theme_cause", "cause_type",
            "theme_consequence", "maintenance_suspectee"]
    st.dataframe(df[[c for c in cols if c in df]], hide_index=True, width="stretch", height=320)
    st.download_button("📥 Télécharger la sélection (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "incidents_selection.csv", "text/csv")
    if len(df):
        inc = st.selectbox("Incident à lire", df["incident_id"])
        r = df[df["incident_id"] == inc].iloc[0]
        with st.expander("Rapport et analyse", expanded=True):
            st.write(r["text"])
            st.markdown(f"**Cause** ({r['cause_type']}) : {r['cause_racine']}  \n"
                        f"**Indice cité** : {r.get('indice') or '—'}  \n"
                        f"**Conséquence** : {r['consequence']} ({r['categorie_consequence']})  \n"
                        f"**Maintenance suspectée** : {r.get('maintenance_suspectee', '—')} — "
                        f"{r.get('justification_maintenance') or ''}  \n"
                        f"**Action corrective proposée** : {r['action_corrective']}")

    st.subheader("Vérification")
    if review is None:
        st.info("Pas de vérification : lancer `python main.py --steps verification`.")
        return
    verdict = review["verdict"]
    st.markdown(f"**Verdict : {verdict['verdict']}** — {verdict['synthese']}")
    checks = pd.DataFrame(review["controles"])
    checks["ok"] = checks["ok"].map({True: "✅ ok", False: "⛔ échec"})
    st.dataframe(checks, hide_index=True, width="stretch")
    findings = pd.DataFrame(review["constats"])
    if len(findings):
        problems = findings[findings["gravite"] != "conforme"]
        st.markdown(f"**Constats** : {len(problems)} problèmes, {len(findings) - len(problems)} points conformes")
        st.dataframe(findings, hide_index=True, width="stretch")


def main() -> None:
    incidents, sheets, decisions, review, report_md = load()
    st.title("Incidents PV & éolien : rapport actionnable")
    st.caption("Causes et conséquences extraites par le LLM, regroupées en thèmes par l'agent ; tous les chiffres sont "
               "calculés à partir des données (outputs/2_analyse/).")
    tabs = st.tabs(["📋 Rapport actionnable", "🔥 Heatmap causes × conséquences", "📊 Pareto & fiches des causes",
                    "🔎 Explorateur & vérification"])
    with tabs[0]:
        tab_report(incidents, decisions, report_md)
    with tabs[1]:
        tab_heatmap(incidents, decisions)
    with tabs[2]:
        tab_pareto(sheets)
    with tabs[3]:
        tab_explorer(incidents, review)


main()
