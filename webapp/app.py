"""Webapp Streamlit : le rapport d'analyse en version interactive, avec graphiques.

Reprend la mise en page de la branche octave (synthèse exécutive avec Pareto, matrice de risque et
coûts par site ; fiche diagnostic par cause ; explorateur d'incidents) et y ajoute la heatmap causes x
conséquences, les profils de site et les plans de remédiation du workflow d'analyse.

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
from analysis_agent.config import ASSET_LABELS, ASSET_TYPES, MAINTENANCE_INDICATOR, SITE_PROFILES  # noqa: E402

ANALYSIS = ROOT / "outputs" / "2_analyse"
REVIEW = ROOT / "outputs" / "3_verification"

# Criticité : palette de statut, toujours accompagnée de son libellé (jamais la couleur seule).
TIER_COLORS = {"Critique": "#d03b3b", "Élevée": "#ec835a", "Moyenne": "#fab219", "Faible": "#0ca30c"}
TIERS = list(TIER_COLORS)
# Type d'installation : deux premières couleurs catégorielles, dans un ordre fixe.
ASSET_COLORS = {"photovoltaïque": "#2a78d6", "éolien": "#eb6834"}
HIGHLIGHT, MUTED = "#2a78d6", "#b4b2a9"
# Rampe séquentielle bleue (une teinte, clair -> foncé) pour la heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METRICS = {"Coût total (€)": "cout_total", "Nombre d'incidents": "incidents", "Arrêt total (h)": "arret_total_h"}
VERDICTS = {"valide": "✅ validé", "valide_avec_reserves": "⚠️ validé avec réserves", "a_refaire": "⛔ à refaire"}

st.set_page_config(page_title="Incidents PV & éolien | Rapport interactif", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")


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
    incidents["type"] = incidents["asset_type"].map(ASSET_LABELS)
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


def layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    """Mise en forme commune : marges serrées, grille discrète, légende horizontale au-dessus."""
    fig.update_layout(height=height, margin={"l": 10, "r": 30, "t": 40, "b": 10},
                      legend={"orientation": "h", "y": 1.12, "x": 0, "title": None}, **kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def hbars(fig: go.Figure, order: list[str], values) -> go.Figure:
    """Barres horizontales : ordre imposé (du haut vers le bas) et place à droite pour les étiquettes."""
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
    fig.update_xaxes(range=[0, 1.25 * max(float(pd.Series(values).max()), 1.0)])
    return fig


def n_incidents(n: int) -> str:
    return f"{n} incident" + ("s" if n > 1 else "")


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
    st.subheader("Vue d'ensemble du parc")
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
        by_site = (incidents.groupby(["site", "type"])["estimated_cost_eur"].sum().reset_index()
                   .sort_values("estimated_cost_eur", ascending=False))
        fig = px.bar(by_site, x="estimated_cost_eur", y="site", color="type", orientation="h",
                     color_discrete_map=ASSET_COLORS, text=by_site["estimated_cost_eur"].map(euros),
                     labels={"estimated_cost_eur": "Coût total (€)", "site": "", "type": "Type"})
        hbars(fig, list(by_site["site"]), by_site["estimated_cost_eur"])
        st.plotly_chart(layout(fig, 360, bargap=0.35), width="stretch")
    with right:
        st.markdown("#### Matrice de risque des causes")
        if len(sheets):
            fig = px.scatter(sheets.assign(type=sheets["asset_type"].map(ASSET_LABELS)),
                             x="incidents", y="cout_moyen", size="arret_total_h", color="criticite",
                             color_discrete_map=TIER_COLORS, category_orders={"criticite": TIERS}, facet_col="type",
                             hover_name="cause", size_max=30,
                             hover_data={"cout_total": ":,.0f", "arret_total_h": ":,.0f", "incidents": True,
                                         "cout_moyen": ":,.0f", "criticite": True, "type": True},
                             labels={"incidents": "Fréquence (incidents)", "cout_moyen": "Coût moyen (€)",
                                     "arret_total_h": "Arrêt total (h)", "criticite": "Criticité", "type": "Type"})
            fig.update_traces(marker={"line": {"width": 2, "color": "white"}, "opacity": 0.85})
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].capitalize()))
            layout(fig, 380).update_layout(legend={"y": -0.28})  # sous le graphique : les titres des panneaux sont en haut
            st.plotly_chart(fig, width="stretch")
            st.caption("Une bulle par thème de cause ; taille = arrêt total. En haut à droite : fréquent ET cher.")

    st.markdown("#### Priorités du parc")
    st.caption("Couples (site, cause connue) au plus fort coût, hors arrêts de sécurité nominaux. Le coût par an "
               "est la borne haute de l'économie possible si la mesure supprime la cause.")
    fleet = F.fleet_priorities(incidents)
    actions = {(s["site"], p["cause"]): p["action"] for s in report.get("sites", []) for p in s["plan"]}
    fleet["mesure préventive"] = [actions.get((r.site, r.cause), "hors des 3 causes prioritaires du site")
                                  for r in fleet.itertuples()]
    fleet["label"] = fleet["site"] + " · " + fleet["cause"]
    fig = px.bar(fleet, x="cout_par_an", y="label", orientation="h", text=fleet["cout_par_an"].map(euros),
                 hover_data={"incidents": True, "mesure préventive": True, "label": False},
                 labels={"cout_par_an": "Coût par an (€)", "label": ""})
    fig.update_traces(marker_color=HIGHLIGHT)
    hbars(fig, list(fleet["label"]), fleet["cout_par_an"])
    st.plotly_chart(layout(fig, 40 * len(fleet) + 80, bargap=0.35), width="stretch")
    fleet["coût total"] = fleet["cout_total"].map(euros)
    fleet["coût par an"] = fleet["cout_par_an"].map(euros)
    st.dataframe(fleet[["site", "cause", "incidents", "coût total", "coût par an", "mesure préventive"]],
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
    matrix = F.heatmap(sub, METRICS[metric_label])
    if matrix.empty:
        st.info("Aucune cause connue sur ce périmètre.")
        return
    counts = F.heatmap(sub, "incidents").reindex(index=matrix.index, columns=matrix.columns, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        text=counts.where(counts > 0, "").to_numpy(), texttemplate="%{text}", textfont={"size": 11},
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        colorbar={"title": metric_label}, xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b> → %{x}<br>" + metric_label + " : %{z:,.0f}<br>incidents : %{text}<extra></extra>"))
    fig.update_layout(height=max(380, 34 * len(matrix) + 160), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis={"title": "Thème de conséquence", "tickangle": -35},
                      yaxis={"title": "Thème de cause", "autorange": "reversed"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Causes connues seulement ; arrêts de sécurité nominaux exclus. Chiffre de chaque case : nombre d'incidents.")

    choice = (decisions.get("chaines_prioritaires") or {}).get(asset_type)
    if choice:
        auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
        st.subheader("Chaînes prioritaires " + ("(choix automatique par le code)" if auto else "(choisies par l'agent)"))
        chains = pd.DataFrame(choice["chaines"])
        chains["chaîne"] = chains["cause"] + " → " + chains["consequence"]
        fig = px.bar(chains, x="cout_total", y="chaîne", orientation="h", text=chains["cout_total"].map(euros),
                     hover_data={"incidents": True, "part_cout_pct": True, "maintenance_suspectee_pct": True, "chaîne": False},
                     labels={"cout_total": "Coût total (€)", "chaîne": "", "part_cout_pct": "part du coût (%)",
                             "maintenance_suspectee_pct": "maintenance suspectée (%)"})
        fig.update_traces(marker_color=HIGHLIGHT)
        hbars(fig, list(chains["chaîne"]), chains["cout_total"])
        st.plotly_chart(layout(fig, 50 * len(chains) + 80, bargap=0.35), width="stretch")
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
    # Une seule échelle : coût total par cause, part cumulée en étiquette (pas de double axe).
    fig = px.bar(sub, x="cout_total", y="cause", orientation="h", color="criticite",
                 color_discrete_map=TIER_COLORS, category_orders={"criticite": TIERS},
                 text=sub.apply(lambda r: f"{euros(r['cout_total'])} · cumul {r['cumul_cout_pct']:.0f} %", axis=1),
                 labels={"cout_total": "Coût total (€)", "cause": "", "criticite": "Criticité"},
                 hover_data={"incidents": True, "part_cout_pct": True, "consequence_principale": True})
    hbars(fig, list(sub["cause"]), 1.15 * sub["cout_total"])
    st.plotly_chart(layout(fig, max(320, 32 * len(sub) + 120), bargap=0.3), width="stretch")
    tier_filter = st.multiselect("Filtrer par criticité", TIERS, default=["Critique", "Élevée"])
    shown = sub[sub["criticite"].isin(tier_filter)] if tier_filter else sub
    table = shown.assign(**{"coût total": shown["cout_total"].map(euros), "part du coût": shown["part_cout_pct"].astype(str) + " %",
                            "maintenance suspectée": shown["maintenance_suspectee_pct"].astype(str) + " %"})
    st.dataframe(table[["rang", "criticite", "cause", "incidents", "coût total", "part du coût", "consequence_principale",
                        "maintenance suspectée", "mesure"]]
                 .rename(columns={"criticite": "criticité", "consequence_principale": "conséquence principale",
                                  "mesure": "mesure préventive"}), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Fiche diagnostic d'une cause")
    cause = st.selectbox("Cause", sub["cause"],
                         format_func=lambda c: f"[{sub.loc[sub['cause'] == c, 'criticite'].iat[0]}] {c}")
    f = sub[sub["cause"] == cause].iloc[0]
    color = TIER_COLORS.get(f["criticite"], "#64748B")
    st.markdown(f"""<div style="padding:16px 18px;border-radius:8px;border-left:6px solid {color};
        background-color:rgba(128,128,128,0.08);margin-bottom:16px;">
        <h3 style="margin-top:0;">{f['cause']}</h3>{tier_badge(f['criticite'])}
        <span style="margin-left:10px;font-size:0.9rem;opacity:0.8;">Facteur dominant : {f['facteur'] or 'non précisé'}
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
        st.markdown("#### 🛡️ Mesure préventive")
        st.success(f"**{str(f['type']).capitalize()} :** {f['mesure']}\n\n**Mécanisme :** {f['mecanisme']}")

    g = incidents[(incidents["asset_type"] == asset_type) & (incidents["theme_cause"] == cause)]
    left, right = st.columns(2)
    with left:
        chains = F.chain_stats(incidents[incidents["asset_type"] == asset_type])
        chains = chains[chains["cause"] == cause].head(6)
        if len(chains):
            st.markdown("##### Conséquences de cette cause")
            fig = px.bar(chains, x="cout_total", y="consequence", orientation="h", text=chains["incidents"].map(n_incidents),
                         labels={"cout_total": "Coût total (€)", "consequence": ""})
            fig.update_traces(marker_color=HIGHLIGHT)
            hbars(fig, list(chains["consequence"]), chains["cout_total"])
            st.plotly_chart(layout(fig, 40 * len(chains) + 110, bargap=0.35), width="stretch")
    with right:
        st.markdown("##### Répartition par site")
        per_site = (g.groupby("site").agg(incidents=("incident_id", "count"), cout=("estimated_cost_eur", "sum"))
                    .reset_index().sort_values("cout", ascending=False))
        fig = px.bar(per_site, x="cout", y="site", orientation="h", text=per_site["incidents"].map(n_incidents),
                     labels={"cout": "Coût total (€)", "site": ""})
        fig.update_traces(marker_color=HIGHLIGHT)
        hbars(fig, list(per_site["site"]), per_site["cout"])
        st.plotly_chart(layout(fig, 40 * len(per_site) + 110, bargap=0.35), width="stretch")

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
                    f"**Conséquence** : {r['consequence']} ({r['categorie_consequence']})  \n"
                    f"**Maintenance suspectée** : {r.get('maintenance_suspectee', '—')}"
                    + (f" — {r['justification_maintenance']}" if pd.notna(r.get("justification_maintenance"))
                       and r.get("justification_maintenance") else "") + "  \n"
                    f"**Action corrective proposée** : {r['action_corrective']}")
        st.caption(f"Date : {r.get('date', '—')} · Technicien : {r.get('technician_id', '—')} · "
                   f"Format source : {r.get('document_format', '—')}")


# ------------------------------------------------------------------------------------------------
# Onglet 4 : profils et plans par site
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
        <h3 style="margin-top:0;">{site} — profil {s['profil']}{' (par élimination)' if s.get('par_elimination') else ''}</h3>
        <span style="opacity:0.8;">{SITE_PROFILES[asset_type][s['profil']]}</span></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", len(g))
    c2.metric("Coût total", euros(g["estimated_cost_eur"].sum()))
    c3.metric("Causes non précisées", f"{F.unknown_share(g)} %")
    if "maintenance_suspectee" in g:
        c4.metric("Maintenance suspectée", f"{F.share(int((g['maintenance_suspectee'] == 'probable').sum()), len(g))} %",
                  help="Hypothèse du LLM d'après la conséquence ; indicateur secondaire, hors attribution des profils.")
    st.markdown(f"**Pourquoi ce profil :** {s['justification_profil']}")
    st.markdown(f"**Causes dominantes :** {s['causes_dominantes']}")

    left, right = st.columns(2)
    with left:
        indicators = F.profile_indicators(same_type)
        rows = [{"critère": c, "site": other, "valeur": v} for c, d in indicators.items() if d["unite"] == "%"
                for other, v in d["valeurs"].items()]
        crit = pd.DataFrame(rows)
        crit = crit[crit.groupby("critère")["valeur"].transform("max") > 0]
        others = sorted(set(crit["site"]) - {site})
        st.markdown("##### Critères des profils : ce site face aux autres")
        fig = px.bar(crit, x="valeur", y="critère", color="site", orientation="h", barmode="group",
                     category_orders={"site": [site, *others]},
                     color_discrete_map={site: HIGHLIGHT, **dict(zip(others, [MUTED, "#d3d1c7"]))},
                     labels={"valeur": "% des incidents du site", "critère": "", "site": "Site"})
        fig.update_traces(marker_line_width=0)
        fig.update_yaxes(categoryorder="array", categoryarray=list(dict.fromkeys(crit["critère"]))[::-1])
        st.plotly_chart(layout(fig, 34 * crit["critère"].nunique() + 160, bargap=0.25), width="stretch")
        st.caption(f"« {MAINTENANCE_INDICATOR} » est un indicateur secondaire : il n'entre pas dans l'attribution.")
    with right:
        causes = (100 * g["theme_cause"].value_counts(normalize=True)).round(1).head(8).reset_index()
        causes.columns = ["cause", "part"]
        causes["série"] = causes["cause"].map(lambda c: "non exploitable" if c in F.NOT_A_CAUSE else "cause connue")
        st.markdown("##### Causes du site")
        fig = px.bar(causes, x="part", y="cause", orientation="h", color="série",
                     color_discrete_map={"cause connue": HIGHLIGHT, "non exploitable": MUTED},
                     text=causes["part"].map(lambda v: f"{v} %"), labels={"part": "% des incidents du site", "cause": "",
                                                                         "série": ""})
        hbars(fig, list(causes["cause"]), causes["part"])
        st.plotly_chart(layout(fig, 34 * len(causes) + 160, bargap=0.3), width="stretch")

    st.markdown("#### Plan de remédiation")
    st.caption("Mesures préventives pour les causes connues les plus coûteuses du site, hors arrêts de sécurité "
               "nominaux ; priorité selon leur part du coût du site (≥ 25 % haute, ≥ 10 % moyenne).")
    plan = pd.DataFrame(s["plan"])
    if len(plan):
        for p in plan.itertuples():
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.markdown(f"**{p.cause}** → {getattr(p, 'consequence_principale', '') or '—'}  \n"
                           f"*{str(p.type).capitalize()} :* {p.action}  \n*Mécanisme :* {p.mecanisme}")
                b.metric(f"Priorité {p.priorite}", euros(p.cout_total), f"{p.part_cout_pct} % du coût du site",
                         delta_color="off")


# ------------------------------------------------------------------------------------------------
# Onglet 5 : explorateur et vérification
# ------------------------------------------------------------------------------------------------

def tab_explorer(incidents: pd.DataFrame, review: dict | None) -> None:
    st.subheader("Explorateur & recherche d'incidents")
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("🔎 Recherche dans les rapports")
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
    st.title("Rapport d'analyse des incidents PV & éolien")
    st.markdown("Causes et conséquences de chaque incident extraites par le LLM, regroupées en thèmes par l'agent, "
                "croisées dans une heatmap et priorisées par coût : de la synthèse au plan d'actions par site.")
    tabs = st.tabs(["📊 Synthèse exécutive", "🔥 Heatmap causes × conséquences", "🔍 Pareto & fiches diagnostic",
                    "🏭 Profils & plans par site", "📋 Explorateur & vérification"])
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
