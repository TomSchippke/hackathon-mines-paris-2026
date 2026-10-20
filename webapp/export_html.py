"""Rapport HTML autonome : le contenu de l'application Streamlit en une seule page, sans serveur.

Mêmes blocs que l'application (synthèse, fiches du plan d'action, vue d'ensemble, équipements, méthode) ;
les vues de filtrage interactif (explorateur d'incidents, chronologie d'un équipement) restent dans
l'application. Plotly est intégré au fichier : la page s'ouvre hors ligne.

    python -m webapp.export_html            # -> outputs/rapport.html
"""

from __future__ import annotations

import argparse
import logging
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from webapp import app as A

# Appelé hors serveur Streamlit : ses avertissements (« No runtime found »...) sont sans objet ici.
# Streamlit règle le niveau de chacun de ses loggers : il faut les baisser un par un.
for _name in [n for n in logging.root.manager.loggerDict if n.startswith("streamlit")]:
    logging.getLogger(_name).setLevel(logging.ERROR)

DEFAULT_OUT = Path("outputs/rapport.html")
RECURRENCE_WINDOW = 30  # fenêtre de récidive affichée (celle par défaut de l'application)

PAGE_CSS = """
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-body); }
.page { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
nav.toc { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line);
  display: flex; flex-wrap: wrap; gap: 4px 18px; padding: 10px 0; margin: 12px 0 8px; }
nav.toc a { font: 600 15px var(--font-body); color: var(--muted); text-decoration: none; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--ink); }
section.part { scroll-margin-top: 56px; padding-top: 8px; }
section.part > h2.part-title { font-family: var(--font-display); font-size: 30px; margin: 28px 0 4px; }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; margin: 10px 0; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 0 16px; }
"""


def chart_html(fig: go.Figure) -> str:
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    return "<div class='chart'>" + fig.to_html(full_html=False, include_plotlyjs=False,
                                               config={"displaylogo": False, "responsive": True}) + "</div>"


def blocks_html(blocks: list) -> str:
    return "".join(chart_html(b) if isinstance(b, go.Figure) else b for b in blocks)


def part(anchor: str, title: str, body: str) -> str:
    return f"<section class='part' id='{anchor}'><h2 class='rp part-title'>{escape(title)}</h2>{body}</section>"


def plan_html(cards: list[dict], n_isolated: int) -> str:
    active, nominal = [c for c in cards if not c["nominal"]], [c for c in cards if c["nominal"]]
    max_cost = max((c["cost"] for c in cards), default=1.0) or 1.0
    out = (f"<span class='count'>{len(active)} fiche{'s' if len(active) > 1 else ''} · "
           f"{A.eur(sum(c['cost'] for c in active))} · {n_isolated} incidents isolés non classés · "
           "cliquer sur une fiche pour l'ouvrir</span>"
           + "<div class='rp list'>" + "".join(A.card_html(c, max_cost) for c in active) + "</div>")
    if nominal:
        out += A.section_html("Fonctionnement normal", f"{len(nominal)} groupe{'s' if len(nominal) > 1 else ''} où le "
                              "dispositif a réagi comme prévu (protection, mise en sécurité) : pas d'action corrective.")
        out += "<div class='rp list'>" + "".join(A.card_html(c, max_cost) for c in nominal) + "</div>"
    return out


def fiches_table_html(summary_df) -> str:
    rows = summary_df.sort_values("criticality_rank")
    return ("<div class='rp tablebox'><table><tr><th class='num'>Rang</th><th>Criticité</th><th>Type</th><th>Fiche</th>"
            "<th class='num'>Incidents</th><th class='num'>Coût total</th><th class='num'>Coût moyen</th>"
            "<th class='num'>Arrêt</th></tr>"
            + "".join(f"<tr><td class='num'>{int(r['criticality_rank'])}</td>"
                      f"<td><span class='tier t-{escape(str(r['criticality_tier']))}'>{escape(str(r['criticality_tier']))}</span></td>"
                      f"<td>{escape(A.PART.get(r['partition'], str(r['partition'])))}</td><td>{escape(str(r['problem_title']))}</td>"
                      f"<td class='num'>{int(r['frequency'])}</td><td class='num'>{A.eur(r['total_cost_eur'])}</td>"
                      f"<td class='num'>{A.eur(r['mean_cost_eur'])}</td><td class='num'>{A.num(r['total_downtime_hours'])} h</td></tr>"
                      for _, r in rows.iterrows())
            + "</table></div>")


def overview_html(summary_df, incidents_df) -> str:
    return (A.overview_figures_html(summary_df, incidents_df)
            + A.section_html("Pareto des coûts", A.PARETO_NOTE) + chart_html(A.fig_pareto(summary_df, incidents_df))
            + "<div class='cols'><div>"
            + A.section_html("Fréquence et coût moyen", "Taille des bulles : heures d'arrêt cumulées.")
            + chart_html(A.fig_frequency_cost(summary_df)) + "</div><div>"
            + A.section_html("Coût par site", "Coût cumulé des incidents de chaque site.")
            + chart_html(A.fig_site_cost(incidents_df)) + "</div></div>"
            + A.section_html("Toutes les fiches") + fiches_table_html(summary_df))


def equipment_html(w: int = RECURRENCE_WINDOW) -> str:
    eq, _, _ = A.load_equipment_data()
    eq = eq.sort_values("bad_actor_score", ascending=False)
    above = eq[f"recurrence_above_chance_{w}d"].astype(bool)
    return (A.figures([
                (A.num(len(eq)), "équipements"),
                (f"{A.num(eq['mtbf_days'].mean() if len(eq) else 0, 1)} j", "MTBF moyen"),
                (A.num(eq[f"recurrences_{w}d"].sum()), f"récidives sous {w} j"),
                (A.num(eq[f"expected_recurrences_{w}d"].sum()), "attendues au hasard"),
                (A.num(int(above.sum())), "équipements au-dessus du hasard"),
            ])
            + A.section_html("Équipements à surveiller", A.BAD_ACTOR_NOTE) + A.bad_actors_html(eq, w)
            + chart_html(A.fig_mtbf_cost(eq, w)))


def build_report() -> str:
    for path in (A.SUMMARY_PATH, A.INCIDENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable. Exécutez d'abord : {A.RUN_CMD}")
    summary_df, incidents_df = A.load_data()
    data = A.load_action_plan_data()
    cards = A.build_cards(summary_df, incidents_df, data)
    n_isolated = int(summary_df.loc[summary_df["merged_cluster_id"] == A.ISOLATED_ID, "frequency"].sum())
    parts = [
        ("synthese", "Synthèse", blocks_html(A.summary_blocks(A.load_overview(summary_df, incidents_df)))),
        ("plan", "Plan d'action", plan_html(cards, n_isolated)),
        ("ensemble", "Vue d'ensemble", overview_html(summary_df, incidents_df)),
        ("equipements", "Équipements", equipment_html()),
        ("methode", "Méthode", A.method_html(*A.method_args(summary_df, incidents_df, data, cards))),
    ]
    nav = "<nav class='toc' aria-label='Sections'>" + "".join(f"<a href='#{a}'>{escape(t)}</a>" for a, t, _ in parts) + "</nav>"
    body = "".join(part(a, t, b) for a, t, b in parts)
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plan d'action fiabilité</title>
<style>{A.CSS}{PAGE_CSS}</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<main class="page">
{A.header_html(incidents_df, cards)}
{nav}
{body}
<p class="rp footer">Généré à partir des sorties de la pipeline <code>agentic_pipeline</code> et <code>equipment_analysis</code>.
L'explorateur d'incidents et la chronologie par équipement sont dans l'application : <code>streamlit run webapp/app.py</code>.</p>
</main>
</body>
</html>"""


def run(out: Path = DEFAULT_OUT) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(), encoding="utf-8")
    print(f"   ✓ {out} ({out.stat().st_size / 1e6:.1f} Mo)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport HTML autonome")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
