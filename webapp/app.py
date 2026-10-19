"""Application web Streamlit : plan d'action et exploration des incidents.

Même rendu que le rapport statique `outputs/plan_action.html` (palette, typographie, fiches dépliables),
avec en plus les vues interactives : Pareto, explorateur d'incidents, chronologie des équipements.

    streamlit run webapp/app.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(
    page_title="Plan d'action fiabilité PV & éolien",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Chemins des données
SEG_DIR = Path("outputs/segmentation")
SUMMARY_PATH = SEG_DIR / "clusters_merged.csv"
FINE_PATH = SEG_DIR / "clusters_fine.csv"
CONCENTRATION_PATH = SEG_DIR / "concentration.csv"
PRECURSOR_PAIRS_PATH = SEG_DIR / "precursor_pairs.csv"
PRECURSOR_EVIDENCE_PATH = SEG_DIR / "precursor_evidence.csv"
INCIDENTS_PATH = SEG_DIR / "incidents_segmented.csv"
EQUIPMENT_SUMMARY_PATH = Path("outputs/equipment/equipment_summary.csv")
EQUIPMENT_TIMELINE_PATH = Path("outputs/equipment/incidents_with_timeline.csv")
TECHNICIAN_PERF_PATH = Path("outputs/equipment/technician_performance.csv")
RUN_CMD = "python main.py --steps segmentation equipment"
ISOLATED_ID = "isole_atypique"
UNKNOWN_ID = "cause_non_precisee"  # incidents dont le rapport ne dit pas la cause (jamais clusterisés)
WEAK = "à vérifier"  # niveau de preuve d'une concentration significative mais faible (concentration.csv)
RECURRENCE_WINDOWS = [7, 15, 30, 60]  # fenêtres calculées par equipment_analysis

TIERS = ["Critique", "Élevée", "Moyenne", "Faible"]
PART = {"solar": "Photovoltaïque", "wind": "Éolien"}
DIM = {"site": "Site", "equipment_id": "Équipement", "period": "Trimestre"}
SCOPE = {"fleet": "Flotte", "site": "Site", "equipment": "Équipement"}
SEVERITIES = ["sévère", "modérée", "mineure"]
MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]

# Palette du rapport HTML (mode clair)
INK, MUTED, LINE, SUNKEN, ACCENT = "#18222a", "#5a6670", "#d6dde1", "#e9edef", "#1d6a85"
TIER_COLORS = {"Critique": "#b8322a", "Élevée": "#c96a14", "Moyenne": "#9c7a10", "Faible": "#4f7560"}
PART_COLORS = {"wind": ACCENT, "solar": "#c79a2e"}

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap");
:root {
  --bg: #f3f5f6; --surface: #ffffff; --sunken: #e9edef; --ink: #18222a; --muted: #5a6670; --line: #d6dde1;
  --accent: #1d6a85; --accent-soft: #dcebf1;
  --crit: #b8322a; --elev: #c96a14; --moy: #9c7a10; --faible: #4f7560;
  --font-display: "Barlow Condensed", "Arial Narrow", "Helvetica Neue", sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
/* Habillage Streamlit */
html, body, .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stMultiSelect, button { font-family: var(--font-body); }
.stApp { background: var(--bg); }
.block-container { max-width: 1100px; padding-top: 2.2rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 10px 12px; }
.stTabs [data-baseweb="tab"] p { font: 600 15px var(--font-body); color: var(--muted); }
.stTabs [aria-selected="true"] p { color: var(--ink); }
[data-testid="stWidgetLabel"] p { font-size: 13px; color: var(--muted); }
[data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; }

/* Composants du rapport */
.eyebrow { font-family: var(--font-data); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.rp h1, .rp h2, .rp h3 { font-family: var(--font-display); font-weight: 700; line-height: 1.1; margin: 0; color: var(--ink); padding: 0; }
.rp h1 { font-size: clamp(30px, 6vw, 44px); letter-spacing: 0.01em; }
.rp h2 { font-size: 24px; }
.rp-header { display: grid; gap: 12px; margin-bottom: 8px; }
.lede { color: var(--muted); max-width: 70ch; margin: 0; font-size: 15px; line-height: 1.5; }
.figures { display: flex; flex-wrap: wrap; gap: 10px 28px; padding: 6px 0 4px; }
.fig { display: grid; }
.fig b { font-family: var(--font-data); font-size: 20px; font-weight: 500; font-variant-numeric: tabular-nums; color: var(--ink); }
.fig span { font-size: 13px; color: var(--muted); }
.section { display: grid; gap: 6px; margin: 18px 0 6px; }
.note { font-size: 13px; color: var(--muted); max-width: 75ch; margin: 0; line-height: 1.5; }
.count { font-size: 13px; color: var(--muted); }

.rp { color: var(--ink); font-size: 15px; line-height: 1.5; }
.list { display: grid; gap: 10px; }
details.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
details.card > summary {
  list-style: none; cursor: pointer; padding: 14px 16px;
  display: grid; grid-template-columns: 2.4em minmax(0, 1fr); gap: 4px 12px;
}
details.card > summary::-webkit-details-marker { display: none; }
details.card > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.rank { font-family: var(--font-data); font-size: 13px; color: var(--muted); padding-top: 3px; }
.head { display: grid; gap: 6px; min-width: 0; }
.title-row { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.card-title { font-family: var(--font-display); font-size: 21px; font-weight: 600; line-height: 1.15; }
.tier { font: 600 11px var(--font-data); letter-spacing: 0.06em; text-transform: uppercase;
  border: 1.5px solid currentColor; border-radius: 4px; padding: 1px 6px; white-space: nowrap; }
.t-Critique { color: var(--crit); } .t-Élevée { color: var(--elev); } .t-Moyenne { color: var(--moy); } .t-Faible { color: var(--faible); }
.meta { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.meta b { color: var(--ink); font-weight: 600; }
.bar { height: 5px; background: var(--sunken); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 3px; }
.b-Critique { background: var(--crit); } .b-Élevée { background: var(--elev); } .b-Moyenne { background: var(--moy); } .b-Faible { background: var(--faible); }
.body { border-top: 1px solid var(--line); padding: 16px; display: grid; gap: 18px; }
.sec { display: grid; gap: 8px; min-width: 0; }
.sec h3 { font-size: 16px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.sec p { margin: 0; max-width: 70ch; }
.sec ul { margin: 0; padding-left: 1.2em; display: grid; gap: 4px; }
.rec { display: grid; gap: 4px; padding: 10px 12px; background: var(--sunken); border-radius: 6px; }
.rec-top { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.scope { font: 500 11px var(--font-data); text-transform: uppercase; letter-spacing: 0.06em; padding: 1px 6px;
  border-radius: 4px; background: var(--accent-soft); color: var(--accent); }
.scope.fleet { background: transparent; color: var(--muted); border: 1px solid var(--line); }
.targets { font-family: var(--font-data); font-size: 13px; }
.why { font-size: 13px; color: var(--muted); }
.weak { font: 500 11px var(--font-data); text-transform: uppercase; letter-spacing: 0.04em; color: var(--moy);
  border: 1px solid currentColor; border-radius: 4px; padding: 0 5px; white-space: nowrap; }
.quote { font-size: 13px; color: var(--muted); border-left: 2px solid var(--line); padding-left: 10px; font-style: italic; }
.nominal { font: 600 11px var(--font-data); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  border: 1.5px dashed currentColor; border-radius: 4px; padding: 1px 6px; white-space: nowrap; }
.modes { display: flex; flex-wrap: wrap; gap: 8px; }
.mode { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; display: grid; gap: 2px; flex: 1 1 220px; min-width: 0; }
.mode small { color: var(--muted); font-family: var(--font-data); font-size: 12px; }

.tablebox { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
.rp table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 0; }
.rp th, .rp td { text-align: left; padding: 8px 10px; border: 0; border-bottom: 1px solid var(--line); vertical-align: top; }
.rp th { font: 500 12px var(--font-data); color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; background: none; }
.rp td.num, .rp th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--font-data); white-space: nowrap; }
.rp tr:last-child td { border-bottom: 0; }
.rp .mini th, .rp .mini td { padding: 5px 8px; }

.prose { display: grid; gap: 12px; max-width: 70ch; }
.prose p { margin: 0; }
.callout { border-left: 3px solid var(--accent); padding: 10px 14px; background: var(--surface); border-radius: 0 6px 6px 0; }
.report { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: grid; gap: 8px; }
.report p { margin: 0; max-width: 75ch; }
.footer { font-size: 12px; color: var(--muted); margin-top: 32px; }
@media (prefers-reduced-motion: no-preference) { details.card[open] .body { animation: open .18s ease-out; } }
@keyframes open { from { opacity: .4; transform: translateY(-3px); } to { opacity: 1; transform: none; } }
"""

# Gabarit Plotly aux couleurs du rapport (fond transparent, grille discrète)
pio.templates["rapport"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=13, color=INK),
    title=dict(font=dict(family="Barlow Condensed, Arial Narrow, sans-serif", size=20, color=INK), x=0, xanchor="left"),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", separators=", ",
    colorway=[ACCENT, "#c79a2e", "#4f7560", "#b8322a", "#7a5c99", "#5a6670"],
    xaxis=dict(gridcolor=LINE, linecolor=LINE, zeroline=False, title=dict(font=dict(color=MUTED, size=12))),
    yaxis=dict(gridcolor=LINE, linecolor=LINE, zeroline=False, title=dict(font=dict(color=MUTED, size=12))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=dict(text=""), font=dict(size=12, color=MUTED)),
    hoverlabel=dict(font=dict(family="IBM Plex Sans, sans-serif", size=12), bgcolor="#ffffff", bordercolor=LINE),
    margin=dict(l=10, r=10, t=50, b=10),
))
pio.templates.default = "plotly_white+rapport"
RED_SCALE = [[0, SUNKEN], [0.5, "#c96a14"], [1, "#b8322a"]]


# ---------------------------------------------------------------------------
# Formatage (conventions françaises, comme le rapport HTML)
# ---------------------------------------------------------------------------
def num(v: float, d: int = 0) -> str:
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def eur(v: float) -> str:
    return f"{num(v)} €"


def keur(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.2f}".rstrip("0").rstrip(".").replace(".", ",") + " M€"
    return f"{num(v / 1000)} k€"


def month(d: pd.Timestamp) -> str:
    return f"{MONTHS[d.month - 1]} {d.year}"


def figures(items: list[tuple[str, str]]) -> str:
    return '<div class="figures">' + "".join(f"<div class='fig'><b>{b}</b><span>{escape(s)}</span></div>"
                                             for b, s in items) + "</div>"


def section(title: str, note: str = "") -> None:
    st.html(f"<div class='rp section'><h2>{escape(title)}</h2>"
            + (f"<p class='note'>{escape(note)}</p>" if note else "") + "</div>")


def chart(fig: go.Figure) -> None:
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données de segmentation avec mise en cache Streamlit."""
    if not SUMMARY_PATH.exists() or not INCIDENTS_PATH.exists():
        st.error(f"Fichiers introuvables ({SUMMARY_PATH} ou {INCIDENTS_PATH}). Exécutez d'abord : `{RUN_CMD}`")
        st.stop()

    summary_df = pd.read_csv(SUMMARY_PATH, keep_default_na=False, na_values=[""])
    summary_df["partition"] = summary_df["partition"].fillna("")
    # Alias de compatibilité : l'identifiant de cluster est l'id fusionné.
    summary_df["cluster_id"] = summary_df["merged_cluster_id"]
    incidents_df = pd.read_csv(INCIDENTS_PATH)

    for col in ["frequency", "criticality_rank"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    for col in ["mean_cost_eur", "total_cost_eur", "mean_downtime_hours", "total_downtime_hours"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0)
    for col in ["estimated_cost_eur", "downtime_hours"]:
        if col in incidents_df.columns:
            incidents_df[col] = pd.to_numeric(incidents_df[col], errors="coerce").fillna(0.0)
    incidents_df["date"] = pd.to_datetime(incidents_df["date"], errors="coerce")
    return summary_df, incidents_df


def parse_json_list(value) -> list:
    """Parse une colonne JSON en liste ; [] si NaN, vide ou invalide."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


@st.cache_data
def load_action_plan_data() -> dict[str, pd.DataFrame]:
    """Charge fins, concentration, précurseurs et preuves (tolère les fichiers vides)."""
    out: dict[str, pd.DataFrame] = {}
    for key, path in {
        "fine": FINE_PATH,
        "concentration": CONCENTRATION_PATH,
        "pairs": PRECURSOR_PAIRS_PATH,
        "evidence": PRECURSOR_EVIDENCE_PATH,
    }.items():
        if not path.exists():
            st.error(f"Fichier introuvable : {path}. Exécutez d'abord : `{RUN_CMD}`")
            out[key] = pd.DataFrame()
            continue
        try:
            out[key] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            out[key] = pd.DataFrame()
    return out


@st.cache_data
def load_equipment_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les données chronologiques d'équipements ou les calcule automatiquement à la volée."""
    if not (EQUIPMENT_SUMMARY_PATH.exists() and EQUIPMENT_TIMELINE_PATH.exists() and TECHNICIAN_PERF_PATH.exists()):
        from equipment_analysis.pipeline import run_equipment_analysis
        res = run_equipment_analysis()
        return res["equipment_df"], res["timeline_df"], res["technician_df"]
    return (pd.read_csv(EQUIPMENT_SUMMARY_PATH), pd.read_csv(EQUIPMENT_TIMELINE_PATH),
            pd.read_csv(TECHNICIAN_PERF_PATH))


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1"])


def _rows(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    return df if df.empty or mask is None else df[mask]


# ---------------------------------------------------------------------------
# Fiches du plan d'action (même contenu que le rapport HTML)
# ---------------------------------------------------------------------------
@st.cache_data
def build_cards(summary_df: pd.DataFrame, incidents_df: pd.DataFrame, data: dict[str, pd.DataFrame]) -> list[dict]:
    """Une fiche par voie de remédiation, triée par criticité."""
    conc, pairs, fine = data["concentration"], data["pairs"], data["fine"]
    if not conc.empty:
        conc = conc[_bool_series(conc, "significant")]
    if not pairs.empty:
        sig = pairs[_bool_series(pairs, "significant")]
        rules = sig[_bool_series(sig, "plausible") & (sig["kind"] != "recurrence")]
        recur = sig[(sig["kind"] == "recurrence") & _bool_series(sig, "is_best_window")]
    else:
        rules = recur = pairs
    severity = pd.crosstab(incidents_df["merged_cluster_id"], incidents_df["severity"])

    cards = []
    rows = summary_df[~summary_df["merged_cluster_id"].isin([ISOLATED_ID, UNKNOWN_ID])].sort_values("criticality_rank")
    for _, r in rows.iterrows():
        cid = r["merged_cluster_id"]
        where = _rows(conc, conc["merged_cluster_id"] == cid if not conc.empty else None).sort_values("lift", ascending=False)
        rr = _rows(rules, ((rules["consequent"] == cid) | (rules["scope_cluster"] == cid)) if not rules.empty else None)
        rc = _rows(recur, (recur["antecedent"] == cid) if not recur.empty else None)
        modes = _rows(fine, fine["merged_cluster_id"] == cid if not fine.empty else None)
        cards.append({
            "id": cid, "rank": int(r["criticality_rank"]), "partition": r["partition"], "tier": r["criticality_tier"],
            "n": int(r["frequency"]), "cost": float(r["total_cost_eur"]), "mean_cost": float(r["mean_cost_eur"]),
            "downtime": float(r["total_downtime_hours"]), "title": str(r["problem_title"]),
            "description": str(r.get("problem_description") or ""), "cause": str(r.get("common_cause") or ""),
            "impact": str(r.get("system_impact") or ""),
            "quotes": [str(q) for q in parse_json_list(r.get("common_cause_quotes"))],
            "cause_source": str(r.get("common_cause_source") or ""),
            "nominal": str(r.get("nominal")).lower() == "true",
            "nominal_reason": str(r.get("nominal_reason") or ""),
            "severity": {k: int(severity.loc[cid, k]) for k in SEVERITIES
                         if cid in severity.index and k in severity.columns and severity.loc[cid, k]},
            "outcomes": [str(o) for o in parse_json_list(r.get("prevented_outcomes"))],
            "recs": [x for x in parse_json_list(r.get("recommendations")) if isinstance(x, dict)],
            "where": [{"dim": w["dimension"], "value": w["value"], "obs": int(w["observed"]),
                       "exp": float(w["expected"]), "lift": float(w["lift"]),
                       "weak": w.get("evidence_level") == WEAK} for _, w in where.iterrows()],
            "rules": [{"rule": p.get("monitoring_rule") if isinstance(p.get("monitoring_rule"), str)
                       else f"{p['antecedent']} → {p['consequent']}",
                       "window": int(p["window_days"]), "lift": float(p["lift"]), "n_eq": int(p["n_equipments"])}
                      for _, p in rr.iterrows()],
            "recurrence": [{"window": int(p["window_days"]), "n_hits": int(p["n_hits"]),
                            "n_eq": int(p["n_equipments"]), "lift": float(p["lift"])} for _, p in rc.iterrows()],
            "modes": [{"title": m["fine_title"], "size": int(m["size"]),
                       "keywords": [k for k in str(m.get("keywords") or "").split(";") if k][:4]}
                      for _, m in modes.sort_values("size", ascending=False).iterrows()],
        })
    return cards


def _target(t: str, weak: list) -> str:
    return escape(str(t)) + (" <span class='weak'>à vérifier</span>" if t in weak else "")


def card_html(c: dict, max_cost: float) -> str:
    sev = ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in c["severity"].items())
    recs = "".join(
        f"<div class='rec'><div class='rec-top'><span class='scope{' fleet' if r.get('scope') == 'fleet' else ''}'>"
        f"{escape(SCOPE.get(r.get('scope'), str(r.get('scope', ''))))}</span>"
        + (f"<span class='targets'>{', '.join(_target(t, r.get('targets_to_verify') or []) for t in r['targets'])}</span>"
           if isinstance(r.get("targets"), list) and r["targets"] else "")
        + f"</div><div>{escape(str(r.get('action', '')))}</div>"
        + (f"<div class='why'>{escape(str(r['rationale']))}</div>" if r.get("rationale") else "") + "</div>"
        for r in c["recs"])
    if c["where"]:
        where = ("<div class='tablebox'><table class='mini'><tr><th>Dimension</th><th>Valeur</th><th class='num'>Observé</th>"
                 "<th class='num'>Attendu</th><th class='num'>Lift</th><th>Preuve</th></tr>"
                 + "".join(f"<tr><td>{DIM.get(w['dim'], escape(w['dim']))}</td><td>{escape(str(w['value']))}</td>"
                           f"<td class='num'>{w['obs']}</td><td class='num'>{num(w['exp'], 1)}</td>"
                           f"<td class='num'>×{num(w['lift'], 1)}</td>"
                           f"<td>{'<span class="weak">à vérifier</span>' if w['weak'] else 'forte'}</td></tr>"
                           for w in c["where"])
                 + "</table></div>"
                 + ("<p class='note'>« À vérifier » : significatif mais avec une preuve faible (q entre 0,05 et 0,10) ; "
                    "à confirmer sur place avant d'engager des moyens.</p>" if any(w["weak"] for w in c["where"]) else ""))
    else:
        where = ("<p class='note'>Aucune concentration significative : ce problème est réparti sur la flotte, "
                 "la remédiation est à mener partout.</p>")
    rules = ("<ul>" + "".join(f"<li>{escape(r['rule'])} <span class='why'>(fenêtre {r['window']} j, lift "
                              f"×{num(r['lift'], 2)}, {r['n_eq']} équipements)</span></li>" for r in c["rules"]) + "</ul>"
             if c["rules"] else
             "<p class='note'>Aucun signal précurseur significatif : sur un même équipement, ces incidents ne sont "
             "pas précédés d'un autre type d'incident plus souvent que par hasard.</p>")
    recur = "".join(f"<p class='note'>Récidive détectée : {r['n_hits']} incidents de ce type surviennent dans les "
                    f"{r['window']} jours suivant un incident identique sur le même équipement ({r['n_eq']} équipements, "
                    f"lift ×{num(r['lift'], 2)}). Vérifier la qualité des réparations.</p>" for r in c["recurrence"])
    modes = "".join(f"<div class='mode'><span>{escape(m['title'])}</span><small>{m['size']} incidents"
                    + (" · " + escape(", ".join(m["keywords"])) if m["keywords"] else "") + "</small></div>"
                    for m in c["modes"])
    outcomes = ("<div class='sec'><h3>Défaillances évitées</h3><ul>"
                + "".join(f"<li>{escape(o)}</li>" for o in c["outcomes"]) + "</ul></div>") if c["outcomes"] else ""
    tier = escape(c["tier"])
    quotes = "".join(f"<p class='quote'>« {escape(q)} »</p>" for q in c["quotes"])
    if c["cause_source"] == "extraction":
        quotes += ("<p class='note'>Cause reprise des rapports : le diagnostic n'a pas cité d'extrait qui justifie "
                   "sa propre formulation.</p>")
    if c["nominal"]:
        recs = ("<p class='note'>Fonctionnement normal : " + escape(c["nominal_reason"] or "le dispositif a réagi comme prévu")
                + ". Aucune action corrective ; à surveiller seulement si la fréquence augmente.</p>")
    badge = "<span class='nominal'>Fonctionnement normal</span>" if c["nominal"] else f"<span class='tier t-{tier}'>{tier}</span>"
    return f"""
<details class="card">
  <summary>
    <span class="rank">#{c['rank']}</span>
    <div class="head">
      <div class="title-row">{badge}<span class="card-title">{escape(c['title'])}</span></div>
      <div class="meta"><span>{PART.get(c['partition'], escape(c['partition']))}</span><span><b>{c['n']}</b> incidents</span>
        <span><b>{eur(c['cost'])}</b></span><span>{num(c['downtime'])} h d'arrêt</span></div>
      <div class="bar" aria-hidden="true"><i class="b-{tier}" style="width:{100 * c['cost'] / max_cost:.1f}%"></i></div>
    </div>
  </summary>
  <div class="body">
    <div class="sec"><h3>Problème et cause</h3><p>{escape(c['description'])}</p><p><b>Cause commune :</b> {escape(c['cause'])}</p>{quotes}
      {f"<p class='note'>Impact : {escape(c['impact'])}</p>" if c['impact'] else ""}
      {f"<p class='note'>Gravité : {sev}. Coût moyen {eur(c['mean_cost'])} par incident.</p>" if sev else ""}</div>
    {outcomes}
    <div class="sec"><h3>Recommandations</h3>{recs or "<p class='note'>Aucune recommandation.</p>"}</div>
    <div class="sec"><h3>Où agir en priorité</h3>{where}</div>
    <div class="sec"><h3>Signaux précurseurs</h3>{rules}{recur}</div>
    <div class="sec"><h3>Modes de défaillance regroupés</h3><div class="modes">{modes}</div></div>
  </div>
</details>"""


# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
def tab_plan(cards: list[dict], n_isolated: int, n_unknown: int) -> None:
    c1, c2 = st.columns([1, 1])
    with c1:
        part_f = st.segmented_control("Type d'installation", ["Tous", *PART.values()], default="Tous", key="ap_part")
    with c2:
        tier_f = st.segmented_control("Criticité", ["Toutes", *TIERS], default="Toutes", key="ap_tier")
    q = st.text_input("Rechercher", "", placeholder="Rechercher un mot, un site, un équipement…",
                      label_visibility="collapsed", key="ap_q").strip().lower()

    rev_part = {v: k for k, v in PART.items()}
    rows = [c for c in cards
            if (part_f in (None, "Tous") or c["partition"] == rev_part.get(part_f))
            and (tier_f in (None, "Toutes") or c["tier"] == tier_f)
            and (not q or q in json.dumps([c["title"], c["description"], c["cause"], c["recs"], c["where"], c["modes"]],
                                          ensure_ascii=False).lower())]
    active, nominal = [c for c in rows if not c["nominal"]], [c for c in rows if c["nominal"]]
    cost = sum(c["cost"] for c in active)
    st.html(f"<span class='count'>{len(active)} fiche{'s' if len(active) > 1 else ''} · {eur(cost)} · "
            f"{n_isolated} incidents isolés · {n_unknown} sans cause précisée dans le rapport</span>")
    max_cost = max((c["cost"] for c in cards), default=1.0) or 1.0
    st.html("<div class='rp list'>" + ("".join(card_html(c, max_cost) for c in active)
                                       or "<p class='note'>Aucune fiche ne correspond à ces filtres.</p>") + "</div>")
    if nominal:
        section("Fonctionnement normal", f"{len(nominal)} groupe{'s' if len(nominal) > 1 else ''} où le dispositif a "
                "réagi comme prévu (protection, mise en sécurité) : pas d'action corrective.")
        st.html("<div class='rp list'>" + "".join(card_html(c, max_cost) for c in nominal) + "</div>")


def tab_overview(summary_df: pd.DataFrame, incidents_df: pd.DataFrame) -> None:
    total_cost = incidents_df["estimated_cost_eur"].sum()
    st.html(figures([
        (f"{num(incidents_df['downtime_hours'].sum())} h", "d'arrêt cumulé"),
        (eur(incidents_df["estimated_cost_eur"].mean()), "coût moyen par incident"),
        (num(len(summary_df)), "groupes (y compris isolés)"),
    ]))

    section("Pareto des coûts", "Chaque barre est une fiche, classée par coût cumulé ; la courbe donne la part "
                                "cumulée du coût total. Le trait pointillé marque 80 %.")
    s = summary_df.sort_values("total_cost_eur", ascending=False).copy()
    s["cum_ratio"] = 100 * s["total_cost_eur"].cumsum() / total_cost
    s["label"] = [f"#{int(r)}" for r in s["criticality_rank"]]
    fig = go.Figure()
    fig.add_bar(x=s["label"], y=s["total_cost_eur"], name="Coût total",
                marker=dict(color=[TIER_COLORS.get(t, MUTED) for t in s["criticality_tier"]]),
                customdata=s["problem_title"], hovertemplate="<b>%{customdata}</b><br>%{y:,.0f} €<extra></extra>")
    fig.add_scatter(x=s["label"], y=s["cum_ratio"], name="Part cumulée du coût", yaxis="y2", mode="lines+markers",
                    line=dict(color=INK, width=2), marker=dict(size=5),
                    hovertemplate="Cumul : %{y:.1f} %<extra></extra>")
    fig.add_hline(y=80, yref="y2", line=dict(color=MUTED, dash="dot", width=1))
    fig.update_layout(height=420, hovermode="x unified", margin=dict(r=50), xaxis=dict(type="category", tickfont=dict(size=11)),
                      yaxis=dict(title="Coût (€)", tickformat="~s"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], tickvals=[0, 20, 40, 60, 80, 100],
                                  ticktext=[f"{v} %" for v in range(0, 101, 20)], showgrid=False))
    chart(fig)

    col1, col2 = st.columns([6, 5])
    with col1:
        section("Fréquence et coût moyen", "Taille des bulles : heures d'arrêt cumulées.")
        fig = px.scatter(summary_df, x="frequency", y="mean_cost_eur", size="total_downtime_hours",
                         color="criticality_tier", color_discrete_map=TIER_COLORS, category_orders={"criticality_tier": TIERS},
                         hover_name="problem_title", size_max=34,
                         hover_data={"frequency": True, "mean_cost_eur": ":,.0f", "total_cost_eur": ":,.0f",
                                     "total_downtime_hours": ":,.0f", "criticality_tier": False},
                         labels={"frequency": "Incidents", "mean_cost_eur": "Coût moyen (€)",
                                 "total_downtime_hours": "Arrêt (h)", "total_cost_eur": "Coût total (€)"})
        fig.update_traces(marker=dict(line=dict(width=1, color="#ffffff"), opacity=0.9))
        fig.update_layout(height=400, xaxis_type="log", yaxis=dict(type="log", tickformat="~s"))
        chart(fig)
    with col2:
        section("Coût par site", "Coût cumulé des incidents de chaque site.")
        site_cost = (incidents_df.groupby(["site", "asset_type"])["estimated_cost_eur"].sum()
                     .reset_index().sort_values("estimated_cost_eur"))
        site_cost["Type"] = site_cost["asset_type"].map(PART).fillna(site_cost["asset_type"])
        fig = px.bar(site_cost, y="site", x="estimated_cost_eur", color="Type", orientation="h",
                     color_discrete_map={PART[k]: v for k, v in PART_COLORS.items()},
                     labels={"estimated_cost_eur": "Coût (€)", "site": ""})
        fig.update_layout(height=400, xaxis_tickformat="~s", bargap=0.35)
        chart(fig)

    section("Toutes les fiches")
    tier_filter = st.multiselect("Criticité", TIERS, default=["Critique", "Élevée"], key="ov_tier", placeholder="Toutes")
    view = summary_df[summary_df["criticality_tier"].isin(tier_filter)] if tier_filter else summary_df
    st.dataframe(
        view[["criticality_rank", "criticality_tier", "partition", "problem_title", "frequency", "total_cost_eur",
              "mean_cost_eur", "total_downtime_hours", "common_cause"]].sort_values("criticality_rank"),
        hide_index=True, width="stretch", height=360,
        column_config={
            "criticality_rank": st.column_config.NumberColumn("Rang", width="small"),
            "criticality_tier": st.column_config.TextColumn("Criticité"),
            "partition": st.column_config.TextColumn("Type"),
            "problem_title": st.column_config.TextColumn("Fiche", width="large"),
            "frequency": st.column_config.NumberColumn("Incidents"),
            "total_cost_eur": st.column_config.NumberColumn("Coût total", format="%.0f €"),
            "mean_cost_eur": st.column_config.NumberColumn("Coût moyen", format="%.0f €"),
            "total_downtime_hours": st.column_config.NumberColumn("Arrêt", format="%.0f h"),
            "common_cause": st.column_config.TextColumn("Cause commune", width="large"),
        },
    )


def tab_incidents(summary_df: pd.DataFrame, incidents_df: pd.DataFrame) -> None:
    titles = summary_df.sort_values("criticality_rank").set_index("cluster_id")["problem_title"].to_dict()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        cluster_f = st.selectbox("Fiche", ["Toutes", *titles], key="in_cluster",
                                 format_func=lambda c: c if c == "Toutes" else titles[c])
    with c2:
        site_f = st.multiselect("Site", sorted(incidents_df["site"].dropna().unique()), key="in_site", placeholder="Tous")
    with c3:
        sev_f = st.multiselect("Gravité", SEVERITIES, key="in_sev", placeholder="Toutes")
    query = st.text_input("Recherche", "", placeholder="Rechercher dans le texte des rapports (ex. foudre, onduleur)",
                          label_visibility="collapsed", key="in_q").strip()

    df = incidents_df
    if cluster_f != "Toutes":
        df = df[df["cluster_id"] == cluster_f]
    if site_f:
        df = df[df["site"].isin(site_f)]
    if sev_f:
        df = df[df["severity"].isin(sev_f)]
    if query:
        df = df[df["text"].str.contains(query, case=False, na=False, regex=False)]

    sources = df["assignment_source"].value_counts().to_dict()
    st.html(figures([
        (f"{num(len(df))} / {num(len(incidents_df))}", "incidents"),
        (keur(df["estimated_cost_eur"].sum()), "coût"),
        (f"{num(df['downtime_hours'].sum())} h", "d'arrêt"),
        (num(df["cluster_id"].nunique()), "fiches touchées"),
    ]) + ("<p class='note'>Affectation : " + " · ".join(f"{n} {escape(str(s))}" for s, n in sources.items()) + "</p>"
          if sources else ""))

    cols = [c for c in ["incident_id", "date", "site", "equipment_id", "severity", "problem_title", "estimated_cost_eur",
                        "downtime_hours", "weather_condition", "assignment_source"] if c in df.columns]
    st.dataframe(df[cols].sort_values("date"), hide_index=True, width="stretch", height=340, column_config={
        "incident_id": st.column_config.NumberColumn("N°", format="%d"),
        "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
        "site": "Site", "equipment_id": "Équipement", "severity": "Gravité",
        "problem_title": st.column_config.TextColumn("Fiche", width="large"),
        "estimated_cost_eur": st.column_config.NumberColumn("Coût", format="%.0f €"),
        "downtime_hours": st.column_config.NumberColumn("Arrêt", format="%.1f h"),
        "weather_condition": "Météo", "assignment_source": "Affectation",
    })
    st.download_button("Télécharger la sélection (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="incidents_selection.csv", mime="text/csv")

    section("Lire un rapport")
    ids = df.sort_values("date")["incident_id"].tolist()
    if not ids:
        st.html("<p class='note'>Aucun incident ne correspond à ces filtres.</p>")
        return
    by_id = df.set_index("incident_id")
    inc_id = st.selectbox("Incident", ids, key="in_id", label_visibility="collapsed",
                          format_func=lambda i: f"N° {i} · {by_id.at[i, 'site']} · {by_id.at[i, 'severity']} · "
                                                f"{by_id.at[i, 'date']:%d/%m/%Y}")
    inc = by_id.loc[inc_id]
    meta = " · ".join(escape(str(x)) for x in [inc.get("equipment_id"), inc.get("technician_id"),
                                                inc.get("weather_condition"), inc.get("document_format")] if pd.notna(x))
    st.html(f"<div class='rp report'><div class='title-row'><span class='tier t-{escape(str(inc['cluster_criticality_tier']))}'>"
            f"{escape(str(inc['cluster_criticality_tier']))}</span><span class='card-title'>{escape(str(inc['problem_title']))}</span></div>"
            f"<div class='meta'><span>{meta}</span><span><b>{eur(inc['estimated_cost_eur'])}</b></span>"
            f"<span>{num(inc['downtime_hours'], 1)} h d'arrêt</span></div><p>{escape(str(inc['text']))}</p></div>")


def tab_equipment() -> None:
    eq_df, tl_df, tech_df = load_equipment_data()
    eq_df = eq_df.sort_values("bad_actor_score", ascending=False)

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        eq_site = st.selectbox("Site", ["Tous", *sorted(eq_df["site"].dropna().unique())], key="eq_site")
    with fc2:
        eq_part = st.selectbox("Type d'installation", ["Tous", *sorted(eq_df["asset_type"].dropna().unique())],
                               key="eq_part", format_func=lambda p: PART.get(p, p))
    with fc3:
        w = st.select_slider("Fenêtre de récidive (jours après l'incident précédent)", RECURRENCE_WINDOWS, value=30,
                             key="eq_rec", help="Un incident survenant moins de N jours après le précédent sur la "
                                                "même machine est compté comme récidive.")

    eq = eq_df
    if eq_site != "Tous":
        eq = eq[eq["site"] == eq_site]
    if eq_part != "Tous":
        eq = eq[eq["asset_type"] == eq_part]
    above = eq[f"recurrence_above_chance_{w}d"].astype(bool)

    tl = tl_df.copy()
    tl["datetime"] = pd.to_datetime(tl["date"])
    tl["is_rec"] = tl["days_since_last_incident"].notna() & (tl["days_since_last_incident"] <= w)
    st.html(figures([
        (num(len(eq)), "équipements"),
        (f"{num(eq['mtbf_days'].mean() if len(eq) else 0, 1)} j", "MTBF moyen"),
        (num(eq[f"recurrences_{w}d"].sum()), f"récidives sous {w} j"),
        (num(eq[f"expected_recurrences_{w}d"].sum()), "attendues au hasard"),
        (num(int(above.sum())), "équipements au-dessus du hasard"),
    ]))
    if not above.any():
        st.html("<div class='rp callout'><p><b>Pas de réparation qui ne tient pas détectée.</b> Une machine qui tombe "
                "souvent en panne a mécaniquement beaucoup d'incidents rapprochés. Comparé à des pannes réparties au "
                f"hasard sur sa période d'activité, aucun équipement ne rechute plus souvent sous {w} jours : les "
                "rechutes rapides s'expliquent par la fréquence des pannes, pas par la qualité des réparations.</p></div>")

    section("Équipements à surveiller", "Score « bad actor » : coût cumulé (40 %), indisponibilité cumulée (30 %) et "
                                        "récidives au-delà du hasard sous 30 jours (30 %). Sans aucun équipement "
                                        "au-dessus du hasard, le score ne repose que sur le coût et l'indisponibilité.")
    top = eq.head(20)
    st.html("<div class='rp tablebox'><table><tr><th>Équipement</th><th>Site</th><th class='num'>Incidents</th>"
            "<th class='num'>Sévères</th><th class='num'>Coût</th><th class='num'>MTBF</th>"
            f"<th class='num'>Récidives {w} j<br>obs. / hasard</th><th class='num'>Score</th>"
            "<th>Problème le plus fréquent</th></tr>"
            + "".join(f"<tr><td class='num' style='text-align:left'>{escape(str(e['equipment_id']))}</td>"
                      f"<td style='white-space:nowrap'>{escape(str(e['site']))}</td><td class='num'>{int(e['total_incidents'])}</td>"
                      f"<td class='num'>{int(e.get('severe_incidents', 0))}</td><td class='num'>{keur(e['total_cost_eur'])}</td>"
                      f"<td class='num'>{num(e['mtbf_days'], 1)} j</td>"
                      f"<td class='num'>{int(e[f'recurrences_{w}d'])} / {num(e[f'expected_recurrences_{w}d'])}"
                      + (" <span class='weak'>au-dessus</span>" if e[f"recurrence_above_chance_{w}d"] else "") + "</td>"
                      f"<td class='num'>{num(e['bad_actor_score'], 1)}</td><td>{escape(str(e.get('most_frequent_problem', '')))}</td></tr>"
                      for _, e in top.iterrows())
            + "</table></div>")

    fig = px.scatter(eq, x="mtbf_days", y="total_cost_eur", size="total_incidents", color="bad_actor_score",
                     color_continuous_scale=RED_SCALE, hover_name="equipment_id",
                     hover_data={"site": True, "total_incidents": True, f"recurrences_{w}d": True,
                                 f"expected_recurrences_{w}d": ":.0f"},
                     labels={"mtbf_days": "MTBF (jours)", "total_cost_eur": "Coût cumulé (€)",
                             "bad_actor_score": "Score", "total_incidents": "Incidents",
                             f"recurrences_{w}d": f"Récidives {w} j", f"expected_recurrences_{w}d": "Attendues au hasard"},
                     title="MTBF et coût cumulé par équipement")
    fig.update_traces(marker=dict(line=dict(width=1, color="#ffffff")))
    fig.update_layout(height=400, yaxis_tickformat="~s", coloraxis_colorbar=dict(thickness=10, len=0.6))
    chart(fig)

    section("Chronologie d'un équipement")
    if eq.empty:
        st.html("<p class='note'>Aucun équipement pour ces filtres.</p>")
    else:
        by_eq = eq.set_index("equipment_id")
        selected = st.selectbox("Équipement", eq["equipment_id"].tolist(), key="eq_id", label_visibility="collapsed",
                                format_func=lambda x: f"{x} · {by_eq.at[x, 'site']} · {by_eq.at[x, 'total_incidents']} incidents")
        info = by_eq.loc[selected]
        line = tl[tl["equipment_id"] == selected].sort_values("datetime").copy()
        st.html(figures([
            (num(info["total_incidents"]), "incidents"),
            (f"{num(info['mtbf_days'], 1)} j", "MTBF"),
            (f"{int(info[f'recurrences_{w}d'])} / {num(info[f'expected_recurrences_{w}d'])}",
             f"récidives sous {w} j (obs. / hasard)"),
            (keur(info["total_cost_eur"]), "coût cumulé"),
            (escape(str(info["most_frequent_technician"])), "technicien le plus fréquent"),
        ]))
        rec_label, first_label = f"Récidive (< {w} j)", "Incident isolé ou premier"
        line["Statut"] = line["is_rec"].map({True: rec_label, False: first_label})
        hover = {c: True for c in ["severity", "technician_id", "problem_title"] if c in line.columns}
        fig = px.scatter(line, x="datetime", y="days_since_last_incident", color="Statut", size="downtime_hours",
                         color_discrete_map={rec_label: TIER_COLORS["Critique"], first_label: ACCENT},
                         hover_data={"datetime": "|%d/%m/%Y", "days_since_last_incident": ":.0f",
                                     "downtime_hours": ":.1f", **hover},
                         labels={"datetime": "Date", "days_since_last_incident": "Jours depuis l'incident précédent",
                                 "downtime_hours": "Arrêt (h)", "severity": "Gravité", "technician_id": "Technicien",
                                 "problem_title": "Fiche"},
                         title=f"{selected} : taille des points = durée d'arrêt")
        fig.add_hline(y=w, line=dict(color=TIER_COLORS["Critique"], dash="dot", width=1),
                      annotation_text=f"seuil {w} j", annotation_position="bottom right",
                      annotation_font=dict(size=11, color=MUTED))
        fig.update_layout(height=400, margin=dict(t=90), legend=dict(y=1.06))
        chart(fig)

    section("Récidive après intervention, par technicien",
            "Chaque technicien est comparé aux autres techniciens de son site (mêmes machines, mêmes conditions) : "
            "part de ses interventions suivies d'un nouvel incident sur la même machine dans la fenêtre choisie.")
    rate, site_rate = f"post_recurrence_rate_{w}d", f"site_recurrence_rate_{w}d"
    differs, q_col = f"differs_from_site_{w}d", f"post_recurrence_q_vs_site_{w}d"
    tech = tech_df.copy()
    tech["Écart"] = tech[rate] - tech[site_rate]
    tech["Verdict"] = [("Plus de récidives" if gap > 0 else "Moins de récidives") if d
                       else ("Pas de comparaison possible" if pd.isna(gap) else "Comme le site")
                       for d, gap in zip(tech[differs].astype(bool), tech["Écart"])]
    if not tech[differs].astype(bool).any():
        st.html("<div class='rp callout'><p><b>Aucun technicien ne se distingue de son site.</b> Les écarts observés "
                "restent dans ce que le hasard produit avec une centaine d'interventions par personne : ce classement "
                "ne permet pas de juger la qualité du travail d'un technicien.</p></div>")
    tech = tech.sort_values("Écart")
    fig = go.Figure(go.Bar(
        x=tech["Écart"], y=tech["technician_id"], orientation="h",
        marker_color=[TIER_COLORS["Critique"] if d else MUTED for d in tech[differs].astype(bool)],
        hovertemplate="%{y}<br>Écart au site : %{x:+.1f} pts<extra></extra>"))
    fig.update_layout(height=440, title="Écart au taux des autres techniciens du site (points)",
                      xaxis=dict(ticksuffix=" pts", zeroline=True, zerolinecolor=INK))
    chart(fig)
    st.dataframe(
        tech[["technician_id", "sites", "total_interventions", rate, site_rate, "Écart", q_col, "Verdict"]]
        .sort_values("Écart", ascending=False),
        hide_index=True, width="stretch", height=420, column_config={
            "technician_id": "Technicien", "sites": "Site", "total_interventions": "Interventions",
            rate: st.column_config.NumberColumn("Taux", format="%.1f %%"),
            site_rate: st.column_config.NumberColumn("Taux du site", format="%.1f %%"),
            "Écart": st.column_config.NumberColumn(format="%+.1f pts"),
            q_col: st.column_config.NumberColumn("q", format="%.2f"),
        })


def tab_method(data: dict[str, pd.DataFrame], n_fine: int, n_isolated: int, n_unknown: int, n_nominal: int,
               per_equipment: float) -> None:
    conc, pairs = data["concentration"], data["pairs"]
    n_conc = int(_bool_series(conc, "significant").sum()) if not conc.empty else 0
    n_weak = int((conc["evidence_level"] == WEAK).sum()) if "evidence_level" in conc else 0
    tested = pairs[pairs["q_value"].notna() & (pairs["kind"] != "recurrence")] if not pairs.empty else pairs
    n_sig = int(_bool_series(tested, "significant").sum()) if not tested.empty else 0
    min_q = tested["q_value"].min() if not tested.empty else float("nan")
    precursors = (f"Aucun n'est significatif (meilleure valeur q : {num(min_q, 2)}). Avec {num(per_equipment)} incidents "
                  "en moyenne par équipement, un tel lien n'est pas détectable s'il est faible."
                  if n_sig == 0 else f"{n_sig} sont significatifs et apparaissent dans les fiches concernées.")
    st.html(f"""
<div class="rp prose">
  <h2>Comment ces fiches sont construites</h2>
  <p>Un LLM lit d'abord la description de chaque rapport, sans aucune autre colonne (ni site, ni gravité, ni
  météo, ni coût), et en extrait la cause racine avec un extrait
  recopié mot pour mot. Le code vérifie que l'extrait figure dans le rapport ; sinon, ou si le rapport ne dit pas
  pourquoi l'incident est arrivé, l'incident est classé « cause non précisée » ({n_unknown} incidents, hors fiches).</p>
  <p>Ce sont ces causes, et non les rapports entiers, qui sont converties en vecteurs sémantiques puis regroupées
  par type d'installation (photovoltaïque, éolien) avec HDBSCAN : la description des dégâts ne sépare plus une même
  cause en plusieurs groupes selon sa gravité. La taille minimale des groupes est choisie automatiquement par le
  score DBCV, ce qui donne {n_fine} causes fines.</p>
  <p>Un agent LLM fusionne ensuite les modes qui relèvent d'une même correction concrète. Une fusion qui dégrade
  fortement la cohérence des groupes doit être confirmée par un second appel critique nommant un seul composant et
  une seule action ; au-delà de 35 % de dégradation, elle est refusée d'office.</p>
  <p>Pour chaque fiche, un test binomial compare la part des incidents de chaque site, équipement et trimestre à leur
  part attendue ({n_conc} concentrations significatives au total, dont {n_weak} « à vérifier » : q entre 0,05 et
  0,10, soit une preuve faible à confirmer sur place). Les recommandations ne peuvent cibler que ces sites ou
  équipements.</p>
  <p>La cause commune de chaque fiche doit être justifiée par un extrait des rapports, vérifié par le code ; à
  défaut, c'est la cause extraite la plus représentative qui est affichée. Le diagnostic peut aussi qualifier un
  groupe de fonctionnement normal (protection déclenchée comme prévu, sans dégât) ; ce n'est accepté que si au moins
  80 % de ses incidents sont au niveau de gravité le plus bas ({n_nominal} groupe{'s' if n_nominal > 1 else ''}
  concerné{'s' if n_nominal > 1 else ''}, listés à part sans recommandation).</p>
  <p>Les récidives d'un équipement sont comparées à ce que le hasard produirait : mêmes nombres d'incidents et même
  période d'activité, dates tirées uniformément (2 000 simulations, correction de Benjamini-Hochberg). Un technicien
  est comparé aux autres techniciens de son site, jamais à la flotte, car chaque site a ses propres machines et
  conditions.</p>
  <div class="callout"><p><b>Signaux précurseurs.</b> {num(len(tested))} enchaînements « incident A puis incident B sur
  le même équipement » ont été testés contre 10 000 permutations aléatoires des historiques. {precursors}</p></div>
  <p>Les {n_isolated} incidents isolés ne ressemblent à aucun groupe et ne figurent dans aucune fiche.</p>
</div>""")


def main() -> None:
    st.html(f"<style>{CSS}</style>")
    summary_df, incidents_df = load_data()
    data = load_action_plan_data()
    cards = build_cards(summary_df, incidents_df, data)
    n_isolated = int(summary_df.loc[summary_df["merged_cluster_id"] == ISOLATED_ID, "frequency"].sum())
    n_unknown = int(summary_df.loc[summary_df["merged_cluster_id"] == UNKNOWN_ID, "frequency"].sum())

    with st.sidebar:
        st.markdown("**Données**")
        st.caption(f"Segmentation : `{SEG_DIR}`  \nÉquipements : `{EQUIPMENT_SUMMARY_PATH.parent}`")
        st.caption(f"Pour régénérer : `{RUN_CMD}`")
        if st.button("Recharger les données"):
            st.cache_data.clear()
            st.rerun()

    total_cost = incidents_df["estimated_cost_eur"].sum()
    crit_cost = sum(c["cost"] for c in cards if c["tier"] == "Critique" and not c["nominal"])
    dates = incidents_df["date"].dropna()
    period = f"{month(dates.min())} – {month(dates.max())} · " if len(dates) else ""
    st.html(f"""
<header class="rp rp-header">
  <span class="eyebrow">{period}{incidents_df['site'].nunique()} sites · {incidents_df['equipment_id'].nunique()} équipements</span>
  <h1>Plan d'action fiabilité</h1>
  <p class="lede">Les incidents des centrales PV et des parcs éoliens regroupés par voie de remédiation : une fiche par
  correction à mener, classée par coût cumulé, avec les sites et équipements où agir en priorité.</p>
  {figures([(num(len(incidents_df)), "incidents analysés"), (keur(total_cost), "coût total estimé"),
            (num(sum(not c["nominal"] for c in cards)), "voies de remédiation"),
            (f"{num(100 * crit_cost / total_cost) if total_cost else 0} %", "du coût dans les fiches critiques")])}
</header>""")

    t_plan, t_overview, t_incidents, t_equipment, t_method = st.tabs(
        ["Plan d'action", "Vue d'ensemble", "Incidents", "Équipements", "Méthode"])
    with t_plan:
        tab_plan(cards, n_isolated, n_unknown)
    with t_overview:
        tab_overview(summary_df, incidents_df)
    with t_incidents:
        tab_incidents(summary_df, incidents_df)
    with t_equipment:
        tab_equipment()
    with t_method:
        tab_method(data, len(data["fine"]), n_isolated, n_unknown, sum(c["nominal"] for c in cards),
                   len(incidents_df) / max(incidents_df["equipment_id"].nunique(), 1))

    st.html("<p class='rp footer'>Généré à partir des sorties de la pipeline <code>agentic_pipeline</code> "
            "(clustering v2) et <code>equipment_analysis</code>.</p>")


if __name__ == "__main__":
    main()
