"""Graphiques SVG en ligne (aucune bibliothèque JS) : Pareto des voies, coût par site, frise d'escalade.

Les couleurs passent par des classes CSS (`tier-*`, `sev-*`, `cat-*`) définies dans le gabarit :
mêmes jetons pour les graphiques et les badges, variante sombre comprise. Chaque marque porte un
`<title>` (infobulle native) et chaque graphique un `<title>`/`<desc>` pour les lecteurs d'écran.
"""

from __future__ import annotations

import math
from html import escape
from itertools import count

import pandas as pd
from markupsafe import Markup

from .formatting import date_fr, eur, eur_short, pct

_ids = count(1)  # identifiants uniques des <title>/<desc> au sein d'une page


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _nice_step(vmax: float, n_ticks: int = 4) -> float:
    """Pas de graduation « rond » (1, 2, 2,5, 5 × 10^k) donnant environ n_ticks graduations."""
    if vmax <= 0:
        return 1.0
    raw = vmax / n_ticks
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def _open_svg(width: int, height: int, title: str, desc: str, css: str) -> list[str]:
    k = next(_ids)
    return [
        f'<svg class="chart {css}" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-labelledby="svgt{k} svgd{k}">',
        f'<title id="svgt{k}">{escape(title)}</title><desc id="svgd{k}">{escape(desc)}</desc>',
    ]


def _x_ticks(vmax: float, x0: float, scale: float, y_top: float, y_bottom: float) -> list[str]:
    step = _nice_step(vmax)
    out = []
    v = 0.0
    while v <= vmax * 1.0001:
        x = x0 + v * scale
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bottom}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{y_top - 6}" text-anchor="middle">{escape(eur_short(v))}</text>')
        v += step
    return out


def pareto_svg(rows: list[dict]) -> Markup:
    """Barres horizontales du coût par voie (ordre décroissant), couleur = palier, cumul en % à droite.

    `rows` : dicts {title, cost, cum_share, tier, tier_slug, href}.
    """
    if not rows:
        return Markup("")
    W, label_w, x0, x1, row_h, top = 760, 292, 300, 610, 24, 34
    H = top + row_h * len(rows) + 8
    vmax = max(r["cost"] for r in rows) or 1.0
    scale = (x1 - x0) / vmax
    total = rows[-1]["cum_share"]
    out = _open_svg(W, H, "Coût total par voie de remédiation (Pareto)",
                    f"{len(rows)} voies classées par coût total décroissant ; couleur selon le palier de "
                    f"criticité ; la colonne de droite donne la part cumulée du coût de la flotte "
                    f"({pct(total)} au total).", "pareto")
    out += _x_ticks(vmax, x0, scale, top, H - 8)
    out.append(f'<text class="tick" x="{W - 4}" y="{top - 6}" text-anchor="end">cumul</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        w = max(r["cost"] * scale, 2)
        tip = f"{r['title']} : {eur(r['cost'])}, cumul {pct(r['cum_share'])}, palier {r['tier']}"
        out.append(f'<a href="{escape(r["href"])}"><g class="row"><title>{escape(tip)}</title>')
        out.append(f'<rect class="hit" x="0" y="{y}" width="{W}" height="{row_h}"/>')
        out.append(f'<text class="label" x="{label_w}" y="{y + 16}" text-anchor="end">'
                   f'{escape(_truncate(r["title"], 40))}</text>')
        out.append(f'<rect class="bar tier-{r["tier_slug"]}" x="{x0}" y="{y + 5}" width="{w:.1f}" '
                   f'height="14" rx="3"/>')
        out.append(f'<text class="value" x="{x0 + w + 6:.1f}" y="{y + 16}">{escape(eur_short(r["cost"]))}</text>')
        out.append(f'<text class="cum" x="{W - 4}" y="{y + 16}" text-anchor="end">{escape(pct(r["cum_share"]))}</text>')
        out.append("</g></a>")
    out.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{H - 8}"/>')
    out.append("</svg>")
    return Markup("\n".join(out))


def site_svg(rows: list[dict], groups: list[str]) -> Markup:
    """Coût total par site, couleur = type d'actif (`groups` : ordre fixe des types).

    `rows` : dicts {site, group, cost, n}, triés par coût décroissant.
    """
    if not rows:
        return Markup("")
    W, label_w, x0, x1, row_h, top = 760, 192, 200, 640, 28, 34
    H = top + row_h * len(rows) + 8
    vmax = max(r["cost"] for r in rows) or 1.0
    scale = (x1 - x0) / vmax
    out = _open_svg(W, H, "Coût total par site et type d'actif",
                    f"{len(rows)} sites ; couleur selon le type d'actif ({', '.join(groups)}).", "sites")
    out += _x_ticks(vmax, x0, scale, top, H - 8)
    for i, r in enumerate(rows):
        y = top + i * row_h
        w = max(r["cost"] * scale, 2)
        cat = groups.index(r["group"]) + 1 if r["group"] in groups else 0
        tip = f"{r['site']} ({r['group']}) : {eur(r['cost'])}, {r['n']} incidents"
        out.append(f'<g class="row"><title>{escape(tip)}</title>')
        out.append(f'<text class="label" x="{label_w}" y="{y + 18}" text-anchor="end">{escape(_truncate(r["site"], 28))}</text>')
        out.append(f'<rect class="bar cat-{cat}" x="{x0}" y="{y + 6}" width="{w:.1f}" height="16" rx="3"/>')
        out.append(f'<text class="value" x="{x0 + w + 6:.1f}" y="{y + 18}">'
                   f'{escape(eur_short(r["cost"]))} · {r["n"]} inc.</text>')
        out.append("</g>")
    out.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{H - 8}"/>')
    out.append("</svg>")
    return Markup("\n".join(out))


def timeline_svg(points: list[dict], title: str) -> Markup:
    """Frise d'une série en escalade : dates en abscisse, points colorés par sévérité, coûts au-dessus,
    identifiants d'incident (liens vers le PDF s'il existe) en dessous.

    `points` : dicts {id, date, severity, sev_slug, cost, href (ou None)} triés par date.
    """
    if not points:
        return Markup("")
    W, H, pad, y0 = 640, 132, 44, 60
    dates = pd.to_datetime([p["date"] for p in points], errors="coerce")
    lo, hi = dates.min() - pd.Timedelta(days=30), dates.max() + pd.Timedelta(days=30)
    span = max((hi - lo).days, 1)

    def xpos(d) -> float:
        return pad + (W - 2 * pad) * (d - lo).days / span

    out = _open_svg(W, H, title, "Incidents de la série par date ; couleur et libellé selon la sévérité ; "
                    "coût au-dessus de chaque point.", "timeline")
    out.append(f'<line class="axis" x1="{pad - 20}" y1="{y0}" x2="{W - pad + 20}" y2="{y0}"/>')
    for year in range(lo.year, hi.year + 2):
        d = pd.Timestamp(year=year, month=1, day=1)
        if lo <= d <= hi:
            x = xpos(d)
            out.append(f'<line class="grid" x1="{x:.1f}" y1="{y0 - 50}" x2="{x:.1f}" y2="{y0 + 8}"/>')
            out.append(f'<text class="tick" x="{x:.1f}" y="{H - 4}" text-anchor="middle">{year}</text>')
    prev_x, level = -1e9, 0
    for p, d in zip(points, dates):
        x = xpos(d)
        level = 1 - level if x - prev_x < 74 else 0  # décale verticalement les libellés trop proches
        prev_x = x
        tip = f"Incident {p['id']} · {date_fr(p['date'])} · {p['severity']} · {eur(p['cost'])}"
        out.append(f'<g><title>{escape(tip)}</title>')
        out.append(f'<circle class="dot sev-{p["sev_slug"]}" cx="{x:.1f}" cy="{y0}" r="7"/>')
        out.append(f'<text class="value" x="{x:.1f}" y="{y0 - 16 - 16 * level}" text-anchor="middle">'
                   f'{escape(eur_short(p["cost"]))}</text>')
        label = f'<text class="idlabel" x="{x:.1f}" y="{y0 + 26 + 16 * level}" text-anchor="middle">#{escape(str(p["id"]))}</text>'
        out.append(f'<a href="{escape(p["href"])}">{label}</a>' if p.get("href") else label)
        out.append("</g>")
    out.append("</svg>")
    return Markup("\n".join(out))
