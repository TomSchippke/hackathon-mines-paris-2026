"""Faits calculés en Python : tout ce que le rapport affirme vient d'ici.

Le LLM ne fait que reformuler ces faits (writing.py) ; les tableaux du rapport (report.py) les
affichent tels quels. Une seule source par chiffre : tous les pourcentages d'un site sont exprimés en
part de l'ensemble de ses incidents (causes non précisées comprises).

Entrée commune : la table des incidents (AnalysisWorkspace.incidents_table), avec pour chaque
incident son thème de cause, son thème de conséquence et l'analyse du LLM.
"""

from __future__ import annotations

import itertools

import pandas as pd

from framing.cadrage import current
from framing.config import AGE_CRITERION, MAINTENANCE_CRITERION

from .config import (CRITICALITY_TIERS, DBCV_LABELS, DISTINCT_LARGE_GAP, DISTINCT_MIN_GAP, DISTINCT_RATIO,
                     FLEET_PRIORITIES, MAINTENANCE_INDICATOR, MAINTENANCE_SUSPECT_LEVELS, PRIORITY_THRESHOLDS,
                     STABILITY_LABELS, TOP_PAIRS, UNDETERMINED)

UNKNOWN = "Cause non précisée"      # thème des incidents dont la cause est inconnue (exclus du clustering)
UNCERTAIN = "Cause incertaine"      # cause inférée avec une confiance faible (exclue, comptée à part)
UNCLASSIFIED = "non classé"         # incidents que HDBSCAN n'a mis dans aucun cluster
NOT_A_CAUSE = {UNKNOWN, UNCERTAIN, UNCLASSIFIED}


def nominal_stops(incidents: pd.DataFrame) -> pd.Series:
    """Arrêts de sécurité nominaux : la protection a fonctionné, sans dommage (catégorie de conséquence
    marquée « fonctionnement normal » au cadrage, sévérité la plus faible). Ce n'est pas une cause à
    supprimer : exclus des plans et de la priorisation, présentés à part."""
    c = current()
    return incidents["categorie_consequence"].isin(c.nominal_consequences) & (incidents["severity"] == c.mildest)


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


# ------------------------------------------------------------------------------------------------
# Critères des profils
# ------------------------------------------------------------------------------------------------

def severe_criterion() -> str:
    return f"incidents « {current().most_severe} »"


def profile_indicators(incidents: pd.DataFrame) -> dict[str, dict]:
    """Critères mesurés sur chaque site d'un même segment : chaque facteur du cadrage (part des
    incidents où il est facteur principal ou secondaire), les deux indicateurs de l'extraction, puis
    des indicateurs secondaires (jamais critères d'un profil) :
    {critère: {"unite": "%" ou "€", "valeurs": {site: valeur}}}."""
    c = current()
    by_site = incidents.groupby("site")
    factors = (incidents["facteur_principal"].fillna("") + " | " + incidents["facteurs_secondaires"].fillna("")) \
        .map(lambda f: {x.strip() for x in f.split("|")})
    flag = lambda col: incidents[col].astype(str).str.lower().isin(["true", "1"])
    out = {}
    for criterion in c.factors:
        if criterion != UNDETERMINED:
            out[criterion] = 100 * factors.map(lambda f, n=criterion: n in f).groupby(incidents["site"]).mean()
    out[AGE_CRITERION] = 100 * flag("lie_vieillissement").groupby(incidents["site"]).mean()
    out[MAINTENANCE_CRITERION] = 100 * flag("lie_maintenance").groupby(incidents["site"]).mean()
    if "maintenance_suspectee" in incidents:  # indicateur secondaire : jamais critère d'un profil
        out[MAINTENANCE_INDICATOR] = 100 * maintenance_suspected(incidents).groupby(incidents["site"]).mean()
    out[severe_criterion()] = 100 * (incidents["severity"] == c.most_severe).groupby(incidents["site"]).mean()
    out["coût moyen par incident"] = by_site["estimated_cost_eur"].mean()
    return {c: {"unite": "€" if c.startswith("coût") else "%", "valeurs": {s: round(float(v), 1) for s, v in values.items()}}
            for c, values in out.items()}


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
    queue, None sinon (y compris valeurs à 0 ou égales partout)."""
    values = indicators[criterion]["valeurs"]
    mine, others = values[site], [v for s, v in values.items() if s != site]
    if mine > 0 and clearly_apart(mine, max(others)):
        return "élevé"
    if clearly_apart(min(others), mine):
        return "faible"
    return None


def distinctive_points(indicators: dict[str, dict], site: str) -> list[dict]:
    """Critères où `site` se distingue nettement des autres sites du même type."""
    points = []
    for criterion, d in indicators.items():
        where = position(indicators, criterion, site)
        if where:
            points.append({"critere": criterion, "position": "le plus élevé" if where == "élevé" else "le plus faible",
                           "valeur": d["valeurs"][site], "unite": d["unite"],
                           "autres_sites": sorted(v for s, v in d["valeurs"].items() if s != site)})
    return points


def expectations_met(indicators: dict[str, dict], site: str, asset_type: str, profile: str) -> list[str]:
    """Attentes du profil (tirées du contexte au cadrage) que le site vérifie nettement."""
    return [f"{criterion} {expected} ({reason})" for criterion, expected, reason in
            current().expectations(asset_type)[profile]
            if criterion in indicators and position(indicators, criterion, site) == expected]


def profile_compatibility(indicators: dict[str, dict], asset_type: str) -> dict[str, dict[str, list[str]]]:
    """Pour chaque site et chaque profil : les attentes du profil vérifiées nettement par le site."""
    sites = sorted(next(iter(indicators.values()))["valeurs"])
    return {site: {p: expectations_met(indicators, site, asset_type, p) for p in sorted(current().profiles(asset_type))}
            for site in sites}


def best_assignments(indicators: dict[str, dict], asset_type: str) -> list[dict[str, str]]:
    """Attributions site -> profil qui respectent le plus d'attentes (plusieurs en cas d'égalité)."""
    compat = profile_compatibility(indicators, asset_type)
    sites, profiles = sorted(compat), sorted(current().profiles(asset_type))
    scored = [(sum(len(compat[s][p]) for s, p in zip(sites, perm)), dict(zip(sites, perm)))
              for perm in itertools.permutations(profiles)]
    top = max(score for score, _ in scored)
    return [assignment for score, assignment in scored if score == top]


def assignment_score(indicators: dict[str, dict], asset_type: str, assignment: dict[str, str]) -> int:
    compat = profile_compatibility(indicators, asset_type)
    return sum(len(compat[site][profile]) for site, profile in assignment.items())


def point_text(p: dict) -> str:
    others = " et ".join(f"{v} {p['unite']}" for v in p["autres_sites"])
    return f"{p['critere']} : {p['valeur']} {p['unite']}, {p['position']} des sites (autres sites : {others})"


def profile_gaps(indicators: dict[str, dict], site: str, asset_type: str, profile: str) -> list[str]:
    """Ce que le profil laisse attendre et que les mesures ne confirment pas."""
    gaps = []
    for criterion, expected, reason in current().expectations(asset_type)[profile]:
        if criterion not in indicators:
            continue
        d = indicators[criterion]
        values, unit = d["valeurs"], d["unite"]
        mine = values[site]
        others = ", ".join(f"{v} {unit}" for s, v in values.items() if s != site)
        measured = f"{mine} {unit} sur ce site, contre {others} sur les autres"
        where = position(indicators, criterion, site)
        if where == expected:
            continue
        if expected == "élevé" and mine < 1.0:
            gaps.append(f"{criterion} attendu élevé ({reason}) mais quasi absent : {measured}")
        elif expected == "élevé":
            gaps.append(f"{criterion} attendu élevé ({reason}) mais pas nettement au-dessus des autres sites : {measured}")
        elif where == "élevé":
            gaps.append(f"{criterion} attendu faible ({reason}) mais nettement le plus élevé : {measured}")
    return gaps


# ------------------------------------------------------------------------------------------------
# Causes, coûts, priorités
# ------------------------------------------------------------------------------------------------

def share(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def cause_stats(site_incidents: pd.DataFrame, actionable_only: bool = False) -> pd.DataFrame:
    """Par thème de cause connue d'un site : incidents, part des incidents du site, coûts, arrêt,
    part du coût total du site et priorité. Trié par coût total décroissant.
    `actionable_only` : sans les arrêts de sécurité nominaux (base des plans de remédiation)."""
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
    stats["priorite"] = stats["part_cout_pct"].map(lambda p: label(p, PRIORITY_THRESHOLDS))
    return stats.sort_values("cout_total", ascending=False).reset_index(drop=True)


def dominant_causes(site_incidents: pd.DataFrame, n: int = 3) -> list[dict]:
    """Les n causes connues les plus fréquentes du site (hors « non précisée » et « non classé »)."""
    stats = cause_stats(site_incidents).sort_values("incidents", ascending=False).head(n)
    return stats[["cause", "incidents", "part_incidents_pct"]].to_dict(orient="records")


def priority_causes(site_incidents: pd.DataFrame, n: int = 3) -> list[dict]:
    """Les n causes connues au plus fort score (fréquence x coût moyen = coût total), hors arrêts de
    sécurité nominaux : base du plan de remédiation."""
    stats = cause_stats(site_incidents, actionable_only=True).head(n).round({"cout_moyen": 0, "arret_moyen": 1})
    return stats.to_dict(orient="records")


def dominant_factor(incidents: pd.DataFrame, cause: str) -> str:
    """Facteur principal le plus fréquent parmi les incidents d'un thème de cause."""
    factors = incidents.loc[incidents["theme_cause"] == cause, "facteur_principal"].dropna()
    factors = factors[factors != UNDETERMINED]
    return factors.mode().iat[0] if len(factors) else ""


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


def explicit_only_profiles(incidents: pd.DataFrame, asset_type: str) -> tuple[list[dict], int]:
    """Test de robustesse : meilleures attributions site -> profil en ne gardant que les causes
    EXPLICITES (sans aucune inférence). Renvoie (attributions, nombre d'incidents utilisés)."""
    sub = incidents[(incidents["asset_type"] == asset_type) & (incidents["cause_type"] == "explicite")]
    return best_assignments(profile_indicators(sub), asset_type), len(sub)


def typical_actions(incidents: pd.DataFrame, cause: str, n: int = 3) -> list[str]:
    """Actions correctives les plus fréquentes proposées par l'extraction pour une cause."""
    return incidents.loc[incidents["theme_cause"] == cause, "action_corrective"].dropna().value_counts().head(n).index.tolist()


def unknown_share(incidents: pd.DataFrame) -> float:
    return share(int((incidents["theme_cause"] == UNKNOWN).sum()), len(incidents))


def years(incidents: pd.DataFrame) -> float:
    dates = pd.to_datetime(incidents["date"], errors="coerce")
    return max((dates.max() - dates.min()).days / 365.25, 1.0)


def fleet_priorities(incidents: pd.DataFrame) -> pd.DataFrame:
    """Interventions sur l'ensemble des sites : couples (site, cause connue) au plus fort coût observé. Le
    coût observé par an est la borne haute de ce qu'une action supprimant la cause économiserait."""
    known = incidents[~incidents["theme_cause"].isin(NOT_A_CAUSE) & ~nominal_stops(incidents)]
    stats = (known.groupby(["site", "theme_cause"])
             .agg(incidents=("incident_id", "count"), cout_total=("estimated_cost_eur", "sum"),
                  arret_total_h=("downtime_hours", "sum"))
             .reset_index().rename(columns={"theme_cause": "cause"}))
    stats["cout_par_an"] = (stats["cout_total"] / years(incidents)).round(0)
    return stats.sort_values("cout_total", ascending=False).head(FLEET_PRIORITIES).reset_index(drop=True)


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
    rang par coût total, part cumulée du coût, niveau (config.CRITICALITY_TIERS)."""
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
    stats["facteur"] = [dominant_factor(incidents, c) for c in stats["cause"]]
    stats["consequence_principale"] = [main_consequence(incidents, c) for c in stats["cause"]]
    return stats.round({"cout_total": 0, "cout_moyen": 0, "arret_total_h": 1})
