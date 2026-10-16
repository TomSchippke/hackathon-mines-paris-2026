"""Données et graphiques du rapport interactif, partagés par la webapp (app.py) et l'export HTML
(export_html.py) : les deux affichent exactement les mêmes graphiques.

Rien ici ne dépend de Streamlit ; tout est lu dans les sorties de la pipeline (aucun appel LLM).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis_agent import facts as F  # noqa: E402
from analysis_agent.config import SCOPE_SEGMENT, SCOPE_SITE  # noqa: E402
from framing.cadrage import current  # noqa: E402

ANALYSIS = ROOT / "outputs" / "2_analyse"
REVIEW = ROOT / "outputs" / "3_verification"

# Criticité : palette de statut, toujours accompagnée de son libellé (jamais la couleur seule).
TIER_COLORS = {"Critique": "#d03b3b", "Élevée": "#ec835a", "Moyenne": "#fab219", "Faible": "#0ca30c"}
TIERS = list(TIER_COLORS)
# Segments (libellés du cadrage) : couleurs catégorielles, dans l'ordre des segments.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#8f5bd6", "#d6a21b", "#d6457a"]
# Portée d'une mesure : deux couleurs catégorielles, toujours légendées.
PORTEE_COLORS = {SCOPE_SITE: CATEGORICAL[0], SCOPE_SEGMENT: CATEGORICAL[1]}


def segment_colors() -> dict[str, str]:
    c = current()
    return {c.label(a): CATEGORICAL[i % len(CATEGORICAL)] for i, a in enumerate(c.segments)}


HIGHLIGHT, MUTED, MUTED_2 = "#2a78d6", "#b4b2a9", "#d3d1c7"
# Rampe séquentielle bleue (une teinte, clair -> foncé) pour la heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METRICS = {"Coût total (€)": "cout_total", "Nombre d'incidents": "incidents", "Arrêt total (h)": "arret_total_h"}
VERDICTS = {"valide": "✅ validé", "valide_avec_reserves": "⚠️ validé avec réserves", "a_refaire": "⛔ à refaire"}


class MissingOutputs(FileNotFoundError):
    """L'analyse n'a pas encore produit ses sorties."""


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None, str]:
    """(incidents, fiches des causes, decisions.json, verification.json ou None, rapport Markdown)."""
    path = ANALYSIS / "incidents_analyse.csv"
    if not path.exists():
        raise MissingOutputs(f"{path} introuvable : lancer d'abord `python main.py --steps analyse`.")
    incidents = pd.read_csv(path)
    for col in ("estimated_cost_eur", "downtime_hours"):
        incidents[col] = pd.to_numeric(incidents[col], errors="coerce").fillna(0.0)
    for col in ("theme_cause", "theme_consequence"):
        incidents[col] = incidents[col].fillna(F.UNCLASSIFIED)
    incidents["type"] = incidents["asset_type"].map(current().label)
    sheets_path = ANALYSIS / "fiches_causes.csv"
    sheets = pd.read_csv(sheets_path) if sheets_path.exists() else pd.DataFrame()
    if "sites_avec_mesure" in sheets:
        sheets["sites_avec_mesure"] = sheets["sites_avec_mesure"].map(sites_text)
    decisions = json.loads((ANALYSIS / "decisions.json").read_text(encoding="utf-8"))
    review_path = REVIEW / "verification.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
    report_path = ANALYSIS / "rapport_analyse.md"
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return incidents, sheets, decisions, review, report_md


euros = F.euros


def n_incidents(n: int) -> str:
    return f"{n} incident" + ("s" if n > 1 else "")


def theme_names() -> dict[str, str]:
    """Nom de chaque thème (causes et conséquences) d'après themes.csv, par identifiant. Indispensable pour
    un thème qui n'apparaît que comme cause contributive (absent des causes racines de la table)."""
    path = ANALYSIS / "themes.csv"
    if not path.exists():
        return {}
    themes = pd.read_csv(path)
    return dict(zip(themes["theme_id"], themes["name"]))


def sites_text(value) -> str:
    """Liste de sites, telle qu'écrite dans fiches_causes.csv (« ['S1', 'S2'] ») ou dans decisions.json."""
    sites = ast.literal_eval(value) if isinstance(value, str) else (value if isinstance(value, list) else [])
    return ", ".join(sites) or "aucun"


def measures(decisions: dict) -> list[dict]:
    """Mesures du plan d'actions (rapport["mesures"]), classées par coût par an."""
    return (decisions.get("rapport") or {}).get("mesures", [])




def site_measures(decisions: dict, site: str) -> list[dict]:
    """Mesures du plan qui concernent ce site (par identifiant, dans l'ordre du plan)."""
    return [m for m in measures(decisions) if m["site"] == site]


def plan_table(decisions: dict) -> pd.DataFrame:
    """Plan d'actions : une ligne par mesure (couple site × cause), avec son statut."""
    plan = pd.DataFrame(measures(decisions))
    if plan.empty:
        return plan
    return plan.assign(statut=plan["a_definir"].map({True: "À définir", False: "rédigée"}))


# ------------------------------------------------------------------------------------------------
# Mise en forme commune
# ------------------------------------------------------------------------------------------------

def layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    """Marges serrées, grille discrète, légende horizontale au-dessus."""
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


def simple_hbar(df: pd.DataFrame, x: str, y: str, text, labels: dict, height_per_bar: int = 40) -> go.Figure:
    """Barres horizontales d'une seule série, dans l'ordre de `df`."""
    fig = px.bar(df, x=x, y=y, orientation="h", text=text, labels=labels)
    fig.update_traces(marker_color=HIGHLIGHT)
    hbars(fig, list(df[y]), df[x])
    return layout(fig, height_per_bar * len(df) + 90, bargap=0.35)


# ------------------------------------------------------------------------------------------------
# Graphiques
# ------------------------------------------------------------------------------------------------

def fig_cost_by_site(incidents: pd.DataFrame) -> go.Figure:
    by_site = (incidents.groupby(["site", "type"])["estimated_cost_eur"].sum().reset_index()
               .sort_values("estimated_cost_eur", ascending=False))
    fig = px.bar(by_site, x="estimated_cost_eur", y="site", color="type", orientation="h",
                 color_discrete_map=segment_colors(), text=by_site["estimated_cost_eur"].map(euros),
                 labels={"estimated_cost_eur": "Coût total (€)", "site": "", "type": "Type"})
    hbars(fig, list(by_site["site"]), by_site["estimated_cost_eur"])
    return layout(fig, 360, bargap=0.35)


def fig_risk_matrix(sheets: pd.DataFrame) -> go.Figure:
    """Une bulle par thème de cause : fréquence x coût moyen, taille = arrêt total, un panneau par type."""
    fig = px.scatter(sheets.assign(type=sheets["asset_type"].map(current().label)),
                     x="incidents", y="cout_moyen", size="arret_total_h", color="criticite",
                     color_discrete_map=TIER_COLORS, category_orders={"criticite": TIERS}, facet_col="type",
                     hover_name="cause", size_max=30,
                     hover_data={"cout_total": ":,.0f", "arret_total_h": ":,.0f", "incidents": True,
                                 "cout_moyen": ":,.0f", "criticite": True, "type": False},
                     labels={"incidents": "Fréquence (incidents)", "cout_moyen": "Coût moyen (€)",
                             "arret_total_h": "Arrêt total (h)", "criticite": "Criticité", "type": "Type"})
    fig.update_traces(marker={"line": {"width": 2, "color": "white"}, "opacity": 0.85})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].capitalize()))
    return layout(fig, 380).update_layout(legend={"y": -0.28})  # sous le graphique : les titres des panneaux sont en haut


def fig_plan(plan: pd.DataFrame) -> go.Figure:
    """Plan d'actions : coût par an de chaque mesure (barres, couleur = portée) et couverture cumulée du coût
    (ligne, axe du haut). Une mesure « À définir » est marquée dans son libellé."""
    labels = [f"{r.id} · {r.site} · {r.cause}" + (" (à définir)" if r.a_definir else "") for r in plan.itertuples()]
    plan = plan.assign(libelle=labels)
    fig = px.bar(plan, x="cout_par_an", y="libelle", orientation="h", color="portee",
                 color_discrete_map=PORTEE_COLORS, category_orders={"portee": list(PORTEE_COLORS)},
                 text=plan["cout_par_an"].map(euros),
                 hover_data={"incidents": True, "mesure": True, "libelle": False, "portee": False},
                 labels={"cout_par_an": "Coût par an (€)", "libelle": "", "portee": "Portée", "mesure": "Mesure"})
    hbars(fig, labels, plan["cout_par_an"])
    fig.add_trace(go.Scatter(x=plan["cumul_pct"], y=labels, xaxis="x2", mode="lines+markers", name="Couverture cumulée",
                             line={"color": MUTED, "width": 2}, marker={"size": 6, "color": MUTED},
                             hovertemplate="%{y}<br>couverture cumulée : %{x} %<extra></extra>"))
    fig.update_layout(xaxis2={"overlaying": "x", "side": "top", "range": [0, 105], "ticksuffix": " %",
                              "showgrid": False, "zeroline": False, "title": "Couverture cumulée du coût (%)"})
    return layout(fig, 40 * len(plan) + 140, bargap=0.35).update_layout(legend={"y": 1.22}, margin={"t": 80})


def fig_heatmap(sub: pd.DataFrame, metric_label: str = "Coût total (€)") -> go.Figure | None:
    """Heatmap thèmes de causes connues x thèmes de conséquences ; texte = nombre d'incidents."""
    matrix = F.heatmap(sub, METRICS[metric_label])
    if matrix.empty:
        return None
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
    return fig


def chains_frame(choice: dict) -> pd.DataFrame:
    chains = pd.DataFrame(choice["chaines"])
    chains["chaîne"] = chains["cause"] + " → " + chains["consequence"]
    return chains


def fig_chains(chains: pd.DataFrame) -> go.Figure:
    """Chaînes prioritaires, dans l'ordre de priorité choisi par l'agent."""
    fig = px.bar(chains, x="cout_total", y="chaîne", orientation="h", text=chains["cout_total"].map(euros),
                 hover_data={"incidents": True, "part_cout_pct": True, "maintenance_suspectee_pct": True, "chaîne": False},
                 labels={"cout_total": "Coût total (€)", "chaîne": "", "part_cout_pct": "part du coût (%)",
                         "maintenance_suspectee_pct": "maintenance suspectée (%)"})
    fig.update_traces(marker_color=HIGHLIGHT)
    hbars(fig, list(chains["chaîne"]), chains["cout_total"])
    return layout(fig, 50 * len(chains) + 80, bargap=0.35)


def fig_pareto(sub: pd.DataFrame) -> go.Figure:
    """Une seule échelle : coût total par cause, part cumulée en étiquette (pas de double axe)."""
    fig = px.bar(sub, x="cout_total", y="cause", orientation="h", color="criticite",
                 color_discrete_map=TIER_COLORS, category_orders={"criticite": TIERS},
                 text=sub.apply(lambda r: f"{euros(r['cout_total'])} · cumul {r['cumul_cout_pct']:.0f} %", axis=1),
                 labels={"cout_total": "Coût total (€)", "cause": "", "criticite": "Criticité"},
                 hover_data={"incidents": True, "part_cout_pct": True, "consequence_principale": True})
    hbars(fig, list(sub["cause"]), 1.15 * sub["cout_total"])
    return layout(fig, max(320, 32 * len(sub) + 120), bargap=0.3)


def fig_cause_consequences(incidents: pd.DataFrame, asset_type: str, cause: str) -> go.Figure | None:
    chains = F.chain_stats(incidents[incidents["asset_type"] == asset_type])
    chains = chains[chains["cause"] == cause].head(6)
    if chains.empty:
        return None
    return simple_hbar(chains, "cout_total", "consequence", chains["incidents"].map(n_incidents),
                       {"cout_total": "Coût total (€)", "consequence": ""})


def fig_cause_sites(cause_incidents: pd.DataFrame) -> go.Figure:
    per_site = (cause_incidents.groupby("site").agg(incidents=("incident_id", "count"), cout=("estimated_cost_eur", "sum"))
                .reset_index().sort_values("cout", ascending=False))
    return simple_hbar(per_site, "cout", "site", per_site["incidents"].map(n_incidents),
                       {"cout": "Coût total (€)", "site": ""})


def fig_site_criteria(same_type: pd.DataFrame, site: str, names: dict[str, str]) -> go.Figure:
    """Critères mesurés (en %) : le site mis en avant face aux autres sites du même segment."""
    indicators = F.site_indicators(same_type, names)
    crit = pd.DataFrame([{"critère": c, "site": s, "valeur": v} for c, d in indicators.items() if d["unite"] == "%"
                         for s, v in d["valeurs"].items()])
    crit = crit[crit.groupby("critère")["valeur"].transform("max") > 0]
    others = sorted(set(crit["site"]) - {site})
    fig = px.bar(crit, x="valeur", y="critère", color="site", orientation="h", barmode="group",
                 category_orders={"site": [site, *others]},
                 color_discrete_map={site: HIGHLIGHT, **{o: (MUTED, MUTED_2)[i % 2] for i, o in enumerate(others)}},
                 labels={"valeur": "% des incidents du site", "critère": "", "site": "Site"})
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(categoryorder="array", categoryarray=list(dict.fromkeys(crit["critère"]))[::-1])
    return layout(fig, 34 * crit["critère"].nunique() + 160, bargap=0.25)


def fig_site_causes(site_incidents: pd.DataFrame) -> go.Figure:
    causes = (100 * site_incidents["theme_cause"].value_counts(normalize=True)).round(1).head(8).reset_index()
    causes.columns = ["cause", "part"]
    causes["série"] = causes["cause"].map(lambda c: "non exploitable" if c in F.NOT_A_CAUSE else "cause connue")
    fig = px.bar(causes, x="part", y="cause", orientation="h", color="série",
                 color_discrete_map={"cause connue": HIGHLIGHT, "non exploitable": MUTED},
                 text=causes["part"].map(lambda v: f"{v} %"),
                 labels={"part": "% des incidents du site", "cause": "", "série": ""})
    hbars(fig, list(causes["cause"]), causes["part"])
    return layout(fig, 34 * len(causes) + 160, bargap=0.3)
