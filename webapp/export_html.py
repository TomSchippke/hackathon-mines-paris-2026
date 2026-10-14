"""Export du rapport interactif en UN fichier HTML autonome, à envoyer à quelqu'un qui l'ouvre dans
son navigateur (double-clic), sans Python ni connexion : plotly.js est intégré au fichier.

Mêmes graphiques que la webapp (figures.py) : survol, zoom, légendes cliquables. Ce qui demande un
serveur (filtres dynamiques, recherche) est remplacé par des vues dépliables : une heatmap par site,
toutes les fiches de causes, tous les plans par site.

Sortie : outputs/2_analyse/rapport_interactif.html

Usage :
    python main.py --steps export       # ou : python webapp/export_html.py
"""

from __future__ import annotations

import html
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import (ANALYSIS, F, TIER_COLORS, VERDICTS, chains_frame, euros,  # noqa: E402
                     fig_cause_consequences, fig_cause_sites, fig_chains, fig_cost_by_site, fig_fleet, fig_heatmap,
                     fig_pareto, fig_risk_matrix, fig_site_causes, fig_site_criteria, fleet_table, load_outputs)
from analysis_agent.config import ASSET_LABELS, ASSET_TYPES, MAINTENANCE_INDICATOR, SITE_PROFILES  # noqa: E402

OUTPUT = ANALYSIS / "rapport_interactif.html"
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}

CSS = """
:root { --bg:#f6f6f4; --surface:#ffffff; --text:#0b0b0b; --muted:#52514e; --border:#e2e1dc; --accent:#2a78d6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }
header { background:var(--surface); border-bottom:1px solid var(--border); padding:20px 16px 0; }
header h1 { margin:0 0 4px; font-size:1.6rem; }
header p { margin:0 0 12px; color:var(--muted); }
nav { position:sticky; top:0; z-index:10; background:var(--surface); border-bottom:1px solid var(--border);
      display:flex; gap:4px; overflow-x:auto; padding:0 16px; }
nav a { padding:10px 12px; color:var(--muted); text-decoration:none; white-space:nowrap; border-bottom:2px solid transparent; }
nav a:hover { color:var(--text); border-bottom-color:var(--accent); }
main { max-width:1200px; margin:0 auto; padding:0 16px 48px; }
section { margin-top:32px; }
h2 { font-size:1.35rem; margin:0 0 12px; padding-top:8px; }
h3 { font-size:1.1rem; margin:20px 0 8px; }
h4 { font-size:0.95rem; margin:14px 0 6px; color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin:12px 0; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }
.kpi .label { color:var(--muted); font-size:0.85rem; }
.kpi .value { font-size:1.35rem; font-weight:600; margin-top:2px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap:16px; }
.grid2 > * { min-width:0; }  /* sinon un graphique élargit sa colonne au-delà de la page */
.note { color:var(--muted); font-size:0.88rem; margin:6px 0; }
.callout { border-left:4px solid var(--accent); background:var(--surface); padding:12px 16px; border-radius:4px; margin:12px 0; }
.callout.info { border-left-color:#2a78d6; } .callout.warn { border-left-color:#fab219; } .callout.ok { border-left-color:#0ca30c; }
.badge { display:inline-block; color:#fff; padding:2px 9px; border-radius:4px; font-weight:600; font-size:0.78rem; }
.sheet { border-left:6px solid var(--accent); }
details { background:var(--surface); border:1px solid var(--border); border-radius:8px; margin:10px 0; }
details > summary { cursor:pointer; padding:12px 16px; font-weight:600; list-style-position:inside; }
details > .body { padding:0 16px 16px; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:0.88rem; background:var(--surface); }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); vertical-align:top; }
th { color:var(--muted); font-weight:600; }
.plan { display:grid; grid-template-columns:1fr auto; gap:12px; align-items:start; }
.plan .cost { text-align:right; min-width:150px; }
.plan .cost .value { font-size:1.2rem; font-weight:600; }
footer { color:var(--muted); font-size:0.85rem; text-align:center; padding:24px 16px; }
@media (max-width:640px) { .plan { grid-template-columns:1fr; } .plan .cost { text-align:left; } }
"""

# Un graphique dans une section repliée a une largeur nulle au chargement : on le redimensionne à l'ouverture.
# Les graphiques sont dessinés pendant la lecture de la page, avant la mise en place de la grille : on les
# redimensionne une fois la page chargée.
JS = """
window.addEventListener('load', () => document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p)));
document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => {
  if (d.open) d.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
}));
"""


def esc(value) -> str:
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def chart(fig) -> str:
    if fig is None:
        return '<p class="note">Pas de données sur ce périmètre.</p>'
    fig.update_layout(template="plotly_white")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOT_CONFIG,
                       default_width="100%", default_height=f"{fig.layout.height or 400}px")


def table(df: pd.DataFrame) -> str:
    # Montants sur une seule ligne : « 880 731 € » ne doit pas se couper entre les milliers.
    df = df.apply(lambda col: col.map(lambda v: v.replace(" ", "\u00a0") if isinstance(v, str) and v.endswith("€") else v))
    return '<div class="table-wrap">' + df.to_html(index=False, escape=True, border=0, na_rep="") + "</div>"


def kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>'


def badge(tier: str) -> str:
    return f'<span class="badge" style="background:{TIER_COLORS.get(tier, "#64748B")}">CRITICITÉ : {esc(tier.upper())}</span>'


def details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details{" open" if open_ else ""}><summary>{summary}</summary><div class="body">{body}</div></details>'


def limits(report_md: str) -> list[str]:
    """Puces de la section « Limites » du rapport Markdown."""
    match = re.search(r"## \d+\. Limites\n(.*?)(?:\n## |\Z)", report_md, re.S)
    return [line[2:] for line in (match.group(1) if match else "").splitlines() if line.startswith("- ")]


# ------------------------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------------------------

def section_overview(incidents, sheets, decisions) -> str:
    report = decisions.get("rapport") or {}
    fleet = fleet_table(incidents, decisions)
    shown = fleet.assign(**{"coût total": fleet["cout_total"].map(euros), "coût par an": fleet["cout_par_an"].map(euros)})
    return f"""<section id="synthese"><h2>Synthèse exécutive</h2>
<div class="kpis">{kpi("Incidents", f"{len(incidents):,}".replace(",", " "))}{kpi("Impact financier", euros(incidents["estimated_cost_eur"].sum()))}
{kpi("Arrêt cumulé", f"{incidents['downtime_hours'].sum():,.0f} h".replace(",", " "))}{kpi("Coût moyen / incident", euros(incidents["estimated_cost_eur"].mean()))}
{kpi("Causes non précisées", f"{F.unknown_share(incidents)} %")}</div>
<div class="callout info"><strong>Synthèse.</strong> {esc(report.get("synthese", "Pas de rapport."))}</div>
<div class="grid2"><div class="card"><h3>Coût par site</h3>{chart(fig_cost_by_site(incidents))}</div>
<div class="card"><h3>Matrice de risque des causes</h3>{chart(fig_risk_matrix(sheets)) if len(sheets) else ""}
<p class="note">Une bulle par thème de cause ; taille = arrêt total. En haut à droite : fréquent ET cher.</p></div></div>
<div class="card"><h3>Priorités du parc</h3><p class="note">Couples (site, cause connue) au plus fort coût, hors arrêts de
sécurité nominaux. Le coût par an est la borne haute de l'économie possible si la mesure supprime la cause.</p>
{chart(fig_fleet(fleet))}{table(shown[["site", "cause", "incidents", "coût total", "coût par an", "mesure préventive"]])}</div>
</section>"""


def section_heatmap(incidents, decisions) -> str:
    parts = ['<section id="heatmap"><h2>Heatmap causes × conséquences</h2><p class="note">Causes connues seulement ; '
             "arrêts de sécurité nominaux exclus. Couleur = coût total, chiffre = nombre d'incidents. Survolez une "
             "case pour le détail.</p>"]
    for asset_type in ASSET_TYPES:
        sub = incidents[incidents["asset_type"] == asset_type]
        parts.append(f'<div class="card"><h3>{esc(ASSET_LABELS[asset_type].capitalize())} — tous les sites</h3>'
                     f"{chart(fig_heatmap(sub))}")
        choice = (decisions.get("chaines_prioritaires") or {}).get(asset_type)
        if choice:
            auto = choice["justification"].startswith("CHOIX AUTOMATIQUE")
            parts.append(f"<h4>Chaînes prioritaires ({'choix automatique par le code' if auto else 'choisies par l’agent'}, "
                         f"dans l'ordre de priorité)</h4>{chart(fig_chains(chains_frame(choice)))}"
                         f'<p class="note"><em>Justification :</em> {esc(choice["justification"])}</p>')
        for site in sorted(sub["site"].unique()):
            parts.append(details(f"Heatmap du site {esc(site)}", chart(fig_heatmap(sub[sub["site"] == site]))))
        parts.append("</div>")
    return "".join(parts) + "</section>"


def section_sheets(incidents, sheets) -> str:
    if sheets.empty:
        return ""
    parts = ['<section id="fiches"><h2>Pareto & fiches diagnostic des causes</h2><p class="note">Thèmes de causes connues '
             "triés par coût total (arrêts nominaux exclus). Criticité : part cumulée du coût ≤ 70 % critique, ≤ 90 % "
             "élevée, ≤ 98 % moyenne, sinon faible. Maintenance suspectée : hypothèse du LLM d'après la conséquence.</p>"]
    for asset_type in ASSET_TYPES:
        sub = sheets[sheets["asset_type"] == asset_type].sort_values("rang")
        if sub.empty:
            continue
        parts.append(f'<div class="card"><h3>{esc(ASSET_LABELS[asset_type].capitalize())}</h3>{chart(fig_pareto(sub))}</div>')
        for f in sub.itertuples():
            g = incidents[(incidents["asset_type"] == asset_type) & (incidents["theme_cause"] == f.cause)]
            body = (f'<p>{badge(f.criticite)} <span class="note">Facteur dominant : {esc(f.facteur or "non précisé")} · '
                    f"{f.part_cout_pct} % du coût des causes connues</span></p>"
                    f'<div class="kpis">{kpi("Incidents", str(int(f.incidents)))}{kpi("Coût total", euros(f.cout_total))}'
                    f'{kpi("Coût moyen", euros(f.cout_moyen))}{kpi("Maintenance suspectée", f"{f.maintenance_suspectee_pct} %")}</div>'
                    f'<div class="callout info"><strong>Ce qui se passe :</strong> {esc(f.description)}</div>'
                    f'<div class="callout warn"><strong>Conséquences :</strong> {esc(f.consequences)}</div>'
                    f'<div class="callout ok"><strong>{esc(str(f.type).capitalize())} :</strong> {esc(f.mesure)}<br>'
                    f"<strong>Mécanisme :</strong> {esc(f.mecanisme)}</div>"
                    f'<div class="grid2"><div><h4>Conséquences de cette cause</h4>'
                    f"{chart(fig_cause_consequences(incidents, asset_type, f.cause))}</div>"
                    f"<div><h4>Répartition par site</h4>{chart(fig_cause_sites(g))}</div></div>")
            summary = (f'<span class="badge" style="background:{TIER_COLORS.get(f.criticite, "#64748B")}">'
                       f"{esc(f.criticite)}</span> {esc(f.cause)} — {euros(f.cout_total)}, {int(f.incidents)} incidents")
            parts.append(details(summary, body))
    return "".join(parts) + "</section>"


def section_sites(incidents, decisions) -> str:
    sites = (decisions.get("rapport") or {}).get("sites", [])
    parts = ['<section id="sites"><h2>Profils & plans par site</h2><p class="note">Plans : mesures préventives pour les '
             "causes connues les plus coûteuses du site, hors arrêts de sécurité nominaux ; priorité selon leur part du "
             f"coût du site (≥ 25 % haute, ≥ 10 % moyenne). « {esc(MAINTENANCE_INDICATOR)} » est un indicateur "
             "secondaire : il n'entre pas dans l'attribution des profils.</p>"]
    for s in sites:
        g = incidents[incidents["site"] == s["site"]]
        asset_type = g["asset_type"].iloc[0]
        suspected = F.share(int((g["maintenance_suspectee"] == "probable").sum()), len(g)) if "maintenance_suspectee" in g else "—"
        plans = "".join(
            f'<div class="card plan"><div><strong>{esc(p["cause"])}</strong> → {esc(p.get("consequence_principale") or "—")}<br>'
            f'<em>{esc(str(p["type"]).capitalize())} :</em> {esc(p["action"])}<br><em>Mécanisme :</em> {esc(p["mecanisme"])}</div>'
            f'<div class="cost"><div class="note">Priorité {esc(p["priorite"])}</div><div class="value">{euros(p["cout_total"])}</div>'
            f'<div class="note">{p["part_cout_pct"]} % du coût du site</div></div></div>' for p in s["plan"])
        body = (f'<p class="note">Profil {esc(s["profil"])} : {esc(SITE_PROFILES[asset_type][s["profil"]])}</p>'
                f'<div class="kpis">{kpi("Incidents", str(len(g)))}{kpi("Coût total", euros(g["estimated_cost_eur"].sum()))}'
                f'{kpi("Causes non précisées", f"{F.unknown_share(g)} %")}{kpi("Maintenance suspectée", f"{suspected} %")}</div>'
                f'<p><strong>Pourquoi ce profil :</strong> {esc(s["justification_profil"])}</p>'
                f'<p><strong>Causes dominantes :</strong> {esc(s["causes_dominantes"])}</p>'
                f'<div class="grid2"><div><h4>Critères des profils : ce site face aux autres</h4>'
                f'{chart(fig_site_criteria(incidents[incidents["asset_type"] == asset_type], s["site"]))}</div>'
                f'<div><h4>Causes du site</h4>{chart(fig_site_causes(g))}</div></div>'
                f"<h4>Plan de remédiation</h4>{plans}")
        summary = (f'{esc(s["site"])} — profil {esc(s["profil"])}{" (par élimination)" if s.get("par_elimination") else ""}'
                   f' · {euros(g["estimated_cost_eur"].sum())}')
        parts.append(details(summary, body))
    return "".join(parts) + "</section>"


def section_review(review, report_md) -> str:
    parts = ['<section id="verification"><h2>Vérification & limites</h2>']
    if review:
        v = review["verdict"]
        checks = pd.DataFrame(review["controles"])
        parts.append(f'<div class="card"><p><strong>Verdict du vérificateur : {esc(VERDICTS.get(v["verdict"], v["verdict"]))}'
                     f"</strong> — {esc(v['synthese'])}</p><p class=\"note\">Contrôles automatiques : {int(checks['ok'].sum())} "
                     f"réussis sur {len(checks)}.</p></div>")
        checks["ok"] = checks["ok"].map({True: "✅ ok", False: "⛔ échec"})
        parts.append(details("Contrôles automatiques", table(checks)))
        findings = pd.DataFrame(review["constats"])
        if len(findings):
            parts.append(details(f"Constats du vérificateur ({len(findings)})", table(findings)))
    bullets = limits(report_md)
    if bullets:
        parts.append('<div class="card"><h3>Limites</h3><ul>' + "".join(f"<li>{esc(b)}</li>" for b in bullets) + "</ul></div>")
    return "".join(parts) + "</section>"


def build() -> Path:
    incidents, sheets, decisions, review, report_md = load_outputs()
    body = "".join([section_overview(incidents, sheets, decisions), section_heatmap(incidents, decisions),
                    section_sheets(incidents, sheets), section_sites(incidents, decisions), section_review(review, report_md)])
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport incidents PV & éolien</title><style>{CSS}</style>
<script>{get_plotlyjs()}</script></head>
<body><header><h1>Rapport d'analyse des incidents PV & éolien</h1>
<p>Rapport interactif généré le {date.today():%d/%m/%Y} à partir des sorties de la pipeline. Causes et conséquences
extraites par le LLM, regroupées en thèmes par l'agent, croisées dans une heatmap et priorisées par coût. Tous les
chiffres sont calculés à partir des données ; les textes du LLM ont été validés puis relus par un juge.</p></header>
<nav><a href="#synthese">Synthèse</a><a href="#heatmap">Heatmap</a><a href="#fiches">Pareto & fiches</a>
<a href="#sites">Profils & plans</a><a href="#verification">Vérification</a></nav>
<main>{body}</main>
<footer>Graphiques interactifs : survol pour le détail, glisser pour zoomer, double-clic pour revenir, clic sur la
légende pour masquer une série.</footer>
<script>{JS}</script></body></html>"""
    OUTPUT.write_text(page, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Rapport interactif : {path} ({path.stat().st_size / 1e6:.1f} Mo)")
