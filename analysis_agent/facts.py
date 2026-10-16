"""Faits calculés en Python : tout ce que le rapport affirme vient d'ici.

Le LLM ne fait que reformuler ces faits (writing.py) ; les tableaux du rapport (report.py) les
affichent tels quels. Une seule source par chiffre : tous les pourcentages d'un site sont exprimés en
part de l'ensemble de ses incidents (causes non précisées comprises).

Entrée commune : la table des incidents (AnalysisWorkspace.incidents_table), avec pour chaque incident
son thème de cause, ses thèmes de causes contributives, son thème de conséquence, son composant, son
indicateur d'arrêt nominal et l'analyse du LLM.

Pas de profil de site : chaque site est caractérisé par ce qui le distingue des autres sites de son
segment (site_indicators, distinctive_points). Le plan d'actions (measure_pairs) porte sur les couples
(site, thème de cause) les plus coûteux, tous segments confondus.
"""

from __future__ import annotations

import pandas as pd

from framing.cadrage import current

from .config import (CRITICALITY_TIERS, DBCV_LABELS, DISTINCT_LARGE_GAP, DISTINCT_MIN_GAP, DISTINCT_RATIO,
                     MAINTENANCE_INDICATOR, MAINTENANCE_SUSPECT_LEVELS, PLAN_COVERAGE_PCT, PLAN_MAX_MEASURES,
                     PLAN_MIN_INCIDENTS, SCOPE_SEGMENT, SCOPE_SITE, STABILITY_LABELS, TOP_PAIRS)

UNKNOWN = "Cause non précisée"      # thème des incidents dont la cause est inconnue (exclus du clustering)
UNCERTAIN = "Cause incertaine"      # cause inférée avec une confiance faible (exclue, comptée à part)
UNCLASSIFIED = "non classé"         # incidents que HDBSCAN n'a mis dans aucun cluster
NOT_A_CAUSE = {UNKNOWN, UNCERTAIN, UNCLASSIFIED}
NO_COMPONENT = "non précisé"        # composant absent ou non identifié
ISOLATED_EXAMPLES = 5               # phrases isolées citées en exemple


def nominal_stops(incidents: pd.DataFrame) -> pd.Series:
    """Arrêts de sécurité nominaux : la protection a fonctionné, sans dommage (thème de conséquence
    choisi comme nominal par l'agent, sévérité la plus faible ; indicateur calculé dans la table des
    incidents). Ce n'est pas une cause à supprimer : exclus des plans et de la priorisation, présentés à part."""
    if "nominal" not in incidents:
        return pd.Series(False, index=incidents.index)
    return incidents["nominal"].fillna(False).astype(bool)


def maintenance_suspected(incidents: pd.DataFrame) -> pd.Series:
    """Incidents où le LLM juge un manque de maintenance probable, d'après la conséquence (hypothèse)."""
    return incidents["maintenance_suspectee"].isin(MAINTENANCE_SUSPECT_LEVELS)


def label(value: float, thresholds: list[tuple[float, str]]) -> str:
    return next(text for limit, text in thresholds if value >= limit)


def clustering_quality(summary: dict) -> str:
    """Qualité d'un essai de clustering en mots, d'après des seuils fixes (config.py)."""
    return (f"{summary['nombre_clusters']} clusters, DBCV {summary['dbcv']:.2f} ({label(summary['dbcv'], DBCV_LABELS)}), "
            f"stabilité {summary['stabilite']:.2f} ({label(summary['stabilite'], STABILITY_LABELS)}), "
            f"{summary['non_classes_pct']} % non classés")


def euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def share(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def years(incidents: pd.DataFrame) -> float:
    dates = pd.to_datetime(incidents["date"], errors="coerce")
    return max((dates.max() - dates.min()).days / 365.25, 1.0)


# ------------------------------------------------------------------------------------------------
# Critères des sites (au sein d'un segment)
# ------------------------------------------------------------------------------------------------

def severe_criterion() -> str:
    return f"incidents « {current().most_severe} »"


def theme_names(incidents: pd.DataFrame) -> dict[str, str]:
    """Nom de chaque thème de cause connu, par identifiant (d'après les causes racines de la table)."""
    roots = incidents.loc[~incidents["theme_cause"].isin(NOT_A_CAUSE), ["theme_cause_id", "theme_cause"]]
    return dict(zip(roots["theme_cause_id"], roots["theme_cause"]))


def incident_themes(incidents: pd.DataFrame, names: dict[str, str] | None = None) -> pd.Series:
    """Par incident, l'ensemble des noms de thèmes qui le concernent : sa cause racine et ses causes
    contributives (identifiants traduits en noms). `names` : {theme_id: nom} de tous les thèmes de causes
    (indispensable pour un thème qui n'apparaît que comme cause contributive) ; à défaut, les paires
    (id, nom) des causes racines de la table, et un identifiant sans nom est ignoré."""
    names = {**theme_names(incidents), **(names or {})}
    themes = []
    for root, ids in zip(incidents["theme_cause"], incidents["themes_contributifs_ids"].fillna("")):
        themes.append({root, *(names[i] for i in ids.split("|") if i in names)})
    return pd.Series(themes, index=incidents.index, dtype=object)


def site_indicators(incidents: pd.DataFrame, names: dict[str, str] | None = None) -> dict[str, dict]:
    """Critères mesurés sur chaque site d'un même segment (incidents de ce seul segment) : un critère par
    thème de cause présent dans le segment (part des incidents du site où ce thème est la cause racine
    OU une cause contributive), puis des indicateurs secondaires (jamais critères) : maintenance
    suspectée, part des incidents les plus graves, coût moyen.
    {critère: {"unite": "%" ou "€", "valeurs": {site: valeur}}}. `names` : voir incident_themes."""
    c = current()
    sites = incidents["site"]
    themes = incident_themes(incidents, names)
    causes = sorted({t for ts in themes for t in ts} - NOT_A_CAUSE)
    out = {name: 100 * themes.map(lambda t, n=name: n in t).groupby(sites).mean() for name in causes}
    if "maintenance_suspectee" in incidents:
        out[MAINTENANCE_INDICATOR] = 100 * maintenance_suspected(incidents).groupby(sites).mean()
    out[severe_criterion()] = 100 * (incidents["severity"] == c.most_severe).groupby(sites).mean()
    out["coût moyen par incident"] = incidents.groupby("site")["estimated_cost_eur"].mean()
    return {name: {"unite": "€" if name.startswith("coût") else "%",
                   "valeurs": {s: round(float(v), 1) for s, v in values.items()}}
            for name, values in out.items()}


def rank_text(indicators: dict[str, dict]) -> dict[str, str]:
    """Version compacte pour l'agent : « Est-1 15.0 % (1er) | Est-2 8.3 % (2e) | Ouest-1 0.3 % (3e) »."""
    out = {}
    for criterion, d in indicators.items():
        values = d["valeurs"]
        if len(set(values.values())) == 1:
            out[criterion] = f"égal sur tous les sites ({next(iter(values.values()))} {d['unite']})"
            continue
        ordered = sorted(values.items(), key=lambda kv: -kv[1])
        out[criterion] = " | ".join(f"{s} {v} {d['unite']} ({i}{'er' if i == 1 else 'e'})"
                                    for i, (s, v) in enumerate(ordered, 1))
    return out


def clearly_apart(high: float, low: float) -> bool:
    """`high` dépasse nettement `low` (voir DISTINCT_* dans config.py)."""
    gap = high - low
    return gap >= DISTINCT_MIN_GAP and (high >= DISTINCT_RATIO * low or gap >= DISTINCT_LARGE_GAP)


def position(indicators: dict[str, dict], criterion: str, site: str) -> str | None:
    """« élevé » si le site est nettement en tête sur ce critère, « faible » s'il est nettement en
    queue, None sinon (y compris valeurs à 0, valeurs égales partout, ou segment d'un seul site)."""
    values = indicators[criterion]["valeurs"]
    mine, others = values[site], [v for s, v in values.items() if s != site]
    if not others:
        return None
    if mine > 0 and clearly_apart(mine, max(others)):
        return "élevé"
    if clearly_apart(min(others), mine):
        return "faible"
    return None


def distinctive_points(indicators: dict[str, dict], site: str) -> list[dict]:
    """Critères où `site` se distingue nettement des autres sites du même segment."""
    points = []
    for criterion, d in indicators.items():
        where = position(indicators, criterion, site)
        if where:
            points.append({"critere": criterion, "position": "le plus élevé" if where == "élevé" else "le plus faible",
                           "valeur": d["valeurs"][site], "unite": d["unite"],
                           "autres_sites": sorted(v for s, v in d["valeurs"].items() if s != site)})
    return points


def point_text(p: dict) -> str:
    others = " et ".join(f"{v} {p['unite']}" for v in p["autres_sites"])
    return f"{p['critere']} : {p['valeur']} {p['unite']}, {p['position']} des sites (autres sites : {others})"


# ------------------------------------------------------------------------------------------------
# Causes, coûts
# ------------------------------------------------------------------------------------------------

def cause_stats(site_incidents: pd.DataFrame, actionable_only: bool = False) -> pd.DataFrame:
    """Par thème de cause connue d'un site : incidents, part des incidents du site, coûts, arrêt,
    part du coût total du site. Trié par coût total décroissant.
    `actionable_only` : sans les arrêts de sécurité nominaux."""
    total_n, total_cost = len(site_incidents), site_incidents["estimated_cost_eur"].sum()
    known = site_incidents[~site_incidents["theme_cause"].isin(NOT_A_CAUSE)]
    if actionable_only:
        known = known[~nominal_stops(known)]
    stats = (known.groupby("theme_cause")
             .agg(incidents=("incident_id", "count"), cout_total=("estimated_cost_eur", "sum"),
                  cout_moyen=("estimated_cost_eur", "mean"), arret_moyen=("downtime_hours", "mean"))
             .reset_index().rename(columns={"theme_cause": "cause"}))
    stats["part_incidents_pct"] = stats["incidents"].map(lambda n: share(n, total_n))
    stats["part_cout_pct"] = (100 * stats["cout_total"] / total_cost).round(1)
    return stats.sort_values("cout_total", ascending=False).reset_index(drop=True)


def dominant_causes(site_incidents: pd.DataFrame, n: int = 3) -> list[dict]:
    """Les n causes connues les plus fréquentes du site (hors « non précisée » et « non classé »)."""
    stats = cause_stats(site_incidents).sort_values("incidents", ascending=False).head(n)
    return stats[["cause", "incidents", "part_incidents_pct"]].to_dict(orient="records")


def main_component(incidents: pd.DataFrame) -> str:
    """Composant le plus fréquent parmi des incidents (« non précisé » ignoré)."""
    components = incidents["composant"].dropna()
    components = components[components != NO_COMPONENT]
    return components.mode().iat[0] if len(components) else NO_COMPONENT


def cause_origin(site_incidents: pd.DataFrame, causes: list[str]) -> dict[str, float]:
    """Part des incidents des causes données dont la cause est explicite ou inférée (en %)."""
    sub = site_incidents[site_incidents["theme_cause"].isin(causes)]
    return {t: share(int((sub["cause_type"] == t).sum()), len(sub)) for t in ("explicite", "inférée")}


def structural_losses(incidents: pd.DataFrame) -> pd.DataFrame:
    """Arrêts de sécurité nominaux par site : perte de production structurelle liée à l'exposition."""
    nominal = incidents[nominal_stops(incidents)]
    stats = (nominal.groupby("site").agg(arrets=("incident_id", "count"), cout_total=("estimated_cost_eur", "sum"),
                                         arret_total_h=("downtime_hours", "sum")).reset_index())
    stats["part_incidents_pct"] = [share(n, int((incidents["site"] == s).sum())) for s, n in zip(stats["site"], stats["arrets"])]
    stats["cout_par_an"] = (stats["cout_total"] / years(incidents)).round(0)
    stats["cause_principale"] = [nominal.loc[nominal["site"] == s, "theme_cause"].mode().iat[0] for s in stats["site"]]
    return stats.sort_values("cout_total", ascending=False).reset_index(drop=True)


def typical_actions(incidents: pd.DataFrame, cause: str, n: int = 3) -> list[str]:
    """Actions correctives les plus fréquentes proposées par l'extraction pour une cause."""
    return incidents.loc[incidents["theme_cause"] == cause, "action_corrective"].dropna().value_counts().head(n).index.tolist()


def unknown_share(incidents: pd.DataFrame) -> float:
    return share(int((incidents["theme_cause"] == UNKNOWN).sum()), len(incidents))


def cause_consequence_pairs(incidents: pd.DataFrame) -> pd.DataFrame:
    """Paires (cause connue -> conséquence) les plus fréquentes, avec leur coût."""
    known = incidents[~incidents["theme_cause"].isin(NOT_A_CAUSE)]
    pairs = (known.groupby(["theme_cause", "theme_consequence"])
             .agg(incidents=("incident_id", "count"), cout_moyen=("estimated_cost_eur", "mean"),
                  cout_total=("estimated_cost_eur", "sum"))
             .sort_values("incidents", ascending=False).head(TOP_PAIRS).reset_index())
    pairs["part_pct"] = pairs["incidents"].map(lambda n: share(n, len(incidents)))
    return pairs.rename(columns={"theme_cause": "cause", "theme_consequence": "conséquence"})


# ------------------------------------------------------------------------------------------------
# Heatmap causes x conséquences, chaînes prioritaires, criticité Pareto
# ------------------------------------------------------------------------------------------------

def chain_stats(incidents: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par case (thème de cause connue, thème de conséquence), arrêts de sécurité nominaux
    exclus : incidents, coût total et moyen, arrêt total, part du coût du périmètre, maintenance suspectée.
    `id` = « <id du thème de cause>><id du thème de conséquence> » (identifiant donné à l'agent)."""
    known = incidents[~incidents["theme_cause"].isin(NOT_A_CAUSE) & ~nominal_stops(incidents)
                      & (incidents["theme_consequence"] != UNCLASSIFIED)].copy()
    if known.empty:
        return pd.DataFrame(columns=["id", "cause", "consequence", "incidents", "cout_total", "cout_moyen",
                                     "arret_total_h", "part_cout_pct", "maintenance_suspectee_pct"])
    known["suspect"] = maintenance_suspected(known) if "maintenance_suspectee" in known else False
    stats = (known.groupby(["theme_cause_id", "theme_consequence_id", "theme_cause", "theme_consequence"])
             .agg(incidents=("incident_id", "count"), cout_total=("estimated_cost_eur", "sum"),
                  cout_moyen=("estimated_cost_eur", "mean"), arret_total_h=("downtime_hours", "sum"),
                  suspect=("suspect", "mean"))
             .reset_index())
    stats["id"] = stats["theme_cause_id"] + ">" + stats["theme_consequence_id"]
    stats["part_cout_pct"] = (100 * stats["cout_total"] / incidents["estimated_cost_eur"].sum()).round(1)
    stats["maintenance_suspectee_pct"] = (100 * stats.pop("suspect")).round(1)
    stats = stats.rename(columns={"theme_cause": "cause", "theme_consequence": "consequence"})
    stats = stats.round({"cout_total": 0, "cout_moyen": 0, "arret_total_h": 1})
    return stats.sort_values("cout_total", ascending=False).reset_index(drop=True)[
        ["id", "cause", "consequence", "incidents", "cout_total", "cout_moyen", "arret_total_h", "part_cout_pct",
         "maintenance_suspectee_pct"]]


def heatmap(incidents: pd.DataFrame, value: str = "cout_total") -> pd.DataFrame:
    """Matrice thème de cause (lignes) x thème de conséquence (colonnes) ; `value` : incidents,
    cout_total ou arret_total_h. Lignes et colonnes triées par total décroissant."""
    stats = chain_stats(incidents)
    if stats.empty:
        return pd.DataFrame()
    matrix = stats.pivot_table(index="cause", columns="consequence", values=value, aggfunc="sum", fill_value=0)
    return matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index,
                      matrix.sum(axis=0).sort_values(ascending=False).index]


def hotspots(incidents: pd.DataFrame, top_k: int) -> list[dict]:
    """Les top_k cases les plus coûteuses de la heatmap."""
    return chain_stats(incidents).head(top_k).to_dict(orient="records")


def main_consequence(incidents: pd.DataFrame, cause: str) -> str:
    """Thème de conséquence le plus coûteux d'une cause (hors arrêts nominaux), sur ce périmètre."""
    stats = chain_stats(incidents)
    stats = stats[stats["cause"] == cause]
    return stats["consequence"].iat[0] if len(stats) else ""


def cause_criticality(incidents: pd.DataFrame) -> pd.DataFrame:
    """Criticité Pareto des thèmes de causes connues d'un segment (arrêts nominaux exclus) :
    rang par coût total, part cumulée du coût, niveau (config.CRITICALITY_TIERS), composant et
    conséquence principaux."""
    known = incidents[~incidents["theme_cause"].isin(NOT_A_CAUSE) & ~nominal_stops(incidents)].copy()
    known["suspect"] = maintenance_suspected(known) if "maintenance_suspectee" in known else False
    stats = (known.groupby(["theme_cause_id", "theme_cause"])
             .agg(incidents=("incident_id", "count"), cout_total=("estimated_cost_eur", "sum"),
                  cout_moyen=("estimated_cost_eur", "mean"), arret_total_h=("downtime_hours", "sum"),
                  suspect=("suspect", "mean"))
             .reset_index().rename(columns={"theme_cause_id": "theme_id", "theme_cause": "cause"})
             .sort_values("cout_total", ascending=False).reset_index(drop=True))
    total = stats["cout_total"].sum()
    stats["part_cout_pct"] = (100 * stats["cout_total"] / total).round(1) if total else 0.0
    stats["cumul_cout_pct"] = (100 * stats["cout_total"].cumsum() / total).round(1) if total else 0.0
    stats["rang"] = range(1, len(stats) + 1)
    stats["criticite"] = [CRITICALITY_TIERS[0][1] if rank == 1 else
                          next(name for limit, name in CRITICALITY_TIERS if cum <= limit + 1e-9)
                          for rank, cum in zip(stats["rang"], stats["cumul_cout_pct"])]
    stats["maintenance_suspectee_pct"] = (100 * stats.pop("suspect")).round(1)
    stats["composant_principal"] = [main_component(known[known["theme_cause"] == c]) for c in stats["cause"]]
    stats["consequence_principale"] = [main_consequence(incidents, c) for c in stats["cause"]]
    return stats.round({"cout_total": 0, "cout_moyen": 0, "arret_total_h": 1})


# ------------------------------------------------------------------------------------------------
# Plan d'actions : couples (site, cause) et causes isolées
# ------------------------------------------------------------------------------------------------

PLAN_COLUMNS = ["id", "site", "asset_type", "theme_id", "cause", "incidents", "cout_total", "cout_par_an",
                "part_cout_pct", "cumul_pct", "portee", "composant_principal", "consequence_principale"]


def measure_pairs(incidents: pd.DataFrame, names: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Plan d'actions (D10-D12), tous segments confondus : couples (site, thème de cause connue), arrêts
    nominaux exclus, d'au moins PLAN_MIN_INCIDENTS incidents. Triés par coût par an ; gardés tant que la
    part cumulée du coût des causes connues n'atteint pas PLAN_COVERAGE_PCT (le couple qui franchit le
    seuil est gardé), au plus PLAN_MAX_MEASURES. Renvoie (table, {"pct", "plafond_atteint", "mesures"})."""
    known = incidents[~incidents["theme_cause"].isin(NOT_A_CAUSE) & ~nominal_stops(incidents)]
    total_cost = known["estimated_cost_eur"].sum()
    per_year = years(incidents)
    indicators = {at: site_indicators(incidents[incidents["asset_type"] == at], names)
                  for at in known["asset_type"].unique()}
    rows = []
    for (site, asset_type, theme_id, cause), group in known.groupby(["site", "asset_type", "theme_cause_id", "theme_cause"]):
        ind = indicators[asset_type]
        scope = SCOPE_SITE if cause in ind and position(ind, cause, site) == "élevé" else SCOPE_SEGMENT
        site_incidents = incidents[(incidents["site"] == site) & (incidents["asset_type"] == asset_type)]
        rows.append({"site": site, "asset_type": asset_type, "theme_id": theme_id, "cause": cause,
                     "incidents": len(group), "cout_total": group["estimated_cost_eur"].sum(),
                     "portee": scope, "composant_principal": main_component(group),
                     "consequence_principale": main_consequence(site_incidents, cause)})
    table = pd.DataFrame(rows, columns=["site", "asset_type", "theme_id", "cause", "incidents", "cout_total",
                                        "portee", "composant_principal", "consequence_principale"])
    table = table[table["incidents"] >= PLAN_MIN_INCIDENTS]
    table = table.assign(cout_par_an=table["cout_total"] / per_year,
                         part_cout_pct=100 * table["cout_total"] / total_cost if total_cost else 0.0)
    table = table.sort_values(["cout_par_an", "site", "theme_id"], ascending=[False, True, True]).reset_index(drop=True)
    table["cumul_pct"] = table["part_cout_pct"].cumsum()
    candidates = len(table)
    previous = table["cumul_pct"].shift(fill_value=0.0)
    table = table[previous < PLAN_COVERAGE_PCT].head(PLAN_MAX_MEASURES).reset_index(drop=True)
    final = float(table["cumul_pct"].iat[-1]) if len(table) else 0.0
    plafond = len(table) == PLAN_MAX_MEASURES and final < PLAN_COVERAGE_PCT and candidates > len(table)
    table.insert(0, "id", [f"M{i:02d}" for i in range(1, len(table) + 1)])
    table = table.round({"cout_total": 0, "cout_par_an": 0, "part_cout_pct": 1, "cumul_pct": 1})
    return table[PLAN_COLUMNS], {"pct": round(final, 1), "plafond_atteint": bool(plafond), "mesures": len(table)}


def isolated_causes(phrases: pd.DataFrame, asset_type: str) -> dict:
    """Causes isolées d'un segment : phrases de cause que l'affectation n'a rattachées à aucun thème
    (theme_id vide). Renvoie {"pct": part parmi les phrases de cause du segment, "phrases": nombre,
    "exemples": jusqu'à ISOLATED_EXAMPLES textes distincts}."""
    causes = phrases[(phrases["asset_type"] == asset_type) & (phrases["dimension"] == "cause")].sort_values("position")
    isolated = causes[causes["theme_id"].isna()]
    return {"pct": share(len(isolated), len(causes)), "phrases": len(isolated),
            "exemples": isolated["texte"].drop_duplicates().head(ISOLATED_EXAMPLES).tolist()}
