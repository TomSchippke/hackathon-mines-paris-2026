"""Diagnostic final des clusters fusionnés (voies de remédiation), avec validation stricte."""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from .llm_client import call_llm_json

MAX_TEXTS_PER_FINE = 2
MAX_TEXTS_TOTAL = 8
MAX_TEXT_CHARS = 600
VALID_SCOPES = ("fleet", "site", "equipment")

SYSTEM_DIAGNOSIS_PROMPT_V2 = """Tu es un expert en fiabilité industrielle (maintenance, retour d'expérience).
Le cluster analysé représente UNE voie de remédiation : un ensemble de défaillances qu'une même action corrective permet d'éviter.

Ta mission : produire le diagnostic final de ce cluster à partir des données fournies (clusters fins membres, remédiation partagée, concentrations statistiquement significatives, règles précurseurs, impact).

Règles strictes :
- Énumère chaque défaillance évitée (prevented_outcomes), une phrase courte par résultat.
- Une cible "equipment" doit être un équipement listé, une cible "site" un site listé ; les périodes ne sont jamais des cibles (citables dans la justification).
- Les recommandations doivent être spécifiques et actionnables. Utilise scope "equipment" ou "site" UNIQUEMENT pour des cibles listées dans la section « Concentrations » ; sinon scope "fleet" avec targets vide.
- Les règles d'alerte précoce (early_warning_rules) ne peuvent utiliser QUE des rule_id fournis dans la section « Règles précurseurs ».
- S'il n'y a aucune concentration ou aucun précurseur, dis-le explicitement (liste vide) au lieu d'inventer.

Format de réponse : un objet JSON STRICT avec les clés :
- "problem_title": titre court (string)
- "problem_description": description du problème (string)
- "common_cause": cause racine commune (string)
- "system_impact": impact opérationnel et financier (string)
- "prevented_outcomes": liste de strings
- "recommendations": liste d'objets {"action", "scope": "fleet"|"site"|"equipment", "targets": [string], "rationale"}
- "early_warning_rules": liste d'objets {"rule_id", "rule", "action"}"""


def make_rule_id(antecedent: Any, consequent: Any, window_days: Any) -> str:
    """Identifiant de règle (identique à precursors.extract_evidence)."""
    return f"{antecedent}->{consequent}@{int(window_days)}d"


def _truthy(series: pd.Series) -> pd.Series:
    """Convertit une colonne booléenne éventuellement lue en texte / NaN."""
    return series.map(lambda v: v is True or str(v).strip().lower() in ("true", "1", "1.0"))


def _clean(v: Any) -> Any:
    if isinstance(v, float) and pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def build_diagnosis_context(
    merged_id: str,
    member_fine_ids: list[str],
    fine_titles: dict[str, dict],
    fine_cards: dict[str, dict],
    merge_log_entry: dict | None,
    concentration: pd.DataFrame,
    precursor_pairs: pd.DataFrame,
    evidence: pd.DataFrame,
    impact: dict,
) -> dict:
    """Assemble le contexte (filtré) fourni au LLM pour un cluster fusionné."""
    fines = []
    texts_total = 0
    for fid in member_fine_ids:
        card = fine_cards.get(fid, {}) or {}
        title = fine_titles.get(fid, {}) or {}
        texts = []
        for t in (card.get("representative_texts") or [])[:MAX_TEXTS_PER_FINE]:
            if texts_total >= MAX_TEXTS_TOTAL:
                break
            texts.append(str(t).strip()[:MAX_TEXT_CHARS])
            texts_total += 1
        fines.append({
            "fine_cluster_id": fid,
            "fine_title": title.get("fine_title", ""),
            "failure_mode": title.get("failure_mode", ""),
            "size": card.get("size"),
            "keywords": list(card.get("keywords") or [])[:10],
            "representative_texts": texts,
        })

    conc_rows: list[dict] = []
    if concentration is not None and len(concentration) and "merged_cluster_id" in concentration:
        c = concentration[concentration["merged_cluster_id"] == merged_id]
        c = c[_truthy(c["significant"])] if "significant" in c else c.iloc[0:0]
        for r in c.to_dict("records"):
            conc_rows.append({k: _clean(r.get(k)) for k in
                              ("dimension", "value", "observed", "expected", "lift", "q_value")})

    rules: list[dict] = []
    if precursor_pairs is not None and len(precursor_pairs):
        p = precursor_pairs
        mask = _truthy(p["significant"]) & _truthy(p["plausible"])
        scope = p["scope_cluster"] if "scope_cluster" in p else pd.Series("", index=p.index)
        into = (p["consequent"] == merged_id) | ((p["kind"] == "escalation") & (scope == merged_id))
        # une escalade d'un autre cluster ne doit pas passer via consequent == merged_id
        into &= ~((p["kind"] == "escalation") & (scope != merged_id))
        counts = {}
        if evidence is not None and len(evidence) and "rule_id" in evidence:
            counts = evidence["rule_id"].value_counts().to_dict()
        for r in p[mask & into].to_dict("records"):
            rid = make_rule_id(r["antecedent"], r["consequent"], r["window_days"])
            rules.append({
                "rule_id": rid,
                "kind": r.get("kind"),
                "antecedent": r["antecedent"],
                "consequent": r["consequent"],
                "window_days": int(r["window_days"]),
                "n_hits": _clean(r.get("n_hits")),
                "n_equipments": _clean(r.get("n_equipments")),
                "lift": _clean(r.get("lift")),
                "q_value": _clean(r.get("q_value")),
                "mechanism": _clean(r.get("mechanism")),
                "monitoring_rule": _clean(r.get("monitoring_rule")),
                "n_evidence_incidents": int(counts.get(rid, 0)),
            })

    merge = merge_log_entry or {}
    return {
        "merged_cluster_id": merged_id,
        "fine_clusters": fines,
        "shared_remediation": merge.get("shared_remediation", ""),
        "merge_prevented_outcomes": merge.get("prevented_outcomes", []) or [],
        "merge_justification": merge.get("justification", ""),
        "concentration": conc_rows,
        "precursor_rules": rules,
        "impact": impact or {},
    }


def _build_user_prompt(ctx: dict) -> str:
    lines = [f"Cluster fusionné : {ctx['merged_cluster_id']}", ""]
    lines.append(f"Remédiation partagée (agent de fusion) : {ctx.get('shared_remediation') or 'N/A'}")
    if ctx.get("merge_justification"):
        lines.append(f"Justification de la fusion : {ctx['merge_justification']}")
    if ctx.get("merge_prevented_outcomes"):
        lines.append("Défaillances évitées selon l'agent de fusion : "
                     + json.dumps(ctx["merge_prevented_outcomes"], ensure_ascii=False, default=str))
    lines += ["", "## Clusters fins membres"]
    for f in ctx["fine_clusters"]:
        lines.append(f"- {f['fine_cluster_id']} | {f['fine_title']} | mode de défaillance : "
                     f"{f['failure_mode']} | taille : {f['size']} | mots-clés : {', '.join(f['keywords'])}")
        for t in f["representative_texts"]:
            lines.append(f"    * Exemple : {t}")
    lines += ["", "## Impact", json.dumps(ctx.get("impact", {}), ensure_ascii=False, default=str)]
    lines += ["", "## Concentrations (significatives)"]
    if ctx["concentration"]:
        for dim, label in (("equipment_id", "Équipements (scope \"equipment\")"),
                           ("site", "Sites (scope \"site\")"),
                           ("period", "Périodes (contexte uniquement, JAMAIS une cible)")):
            rows = [c for c in ctx["concentration"] if c["dimension"] == dim]
            if rows:
                lines.append(f"{label} :")
                for c in rows:
                    lines.append(f"- {c['value']} (observé {c['observed']}, "
                                 f"attendu {c['expected']}, lift {c['lift']})")
    else:
        lines.append("Aucune concentration significative : utiliser uniquement scope \"fleet\".")
    lines += ["", "## Règles précurseurs (plausibles, significatives)"]
    if ctx["precursor_rules"]:
        for r in ctx["precursor_rules"]:
            lines.append(f"- rule_id : {r['rule_id']} | type : {r['kind']} | lift {r['lift']} | "
                         f"{r['n_hits']} cas sur {r['n_equipments']} équipements | "
                         f"preuves : {r['n_evidence_incidents']} incidents | mécanisme : {r['mechanism']} | "
                         f"surveillance : {r['monitoring_rule']}")
    else:
        lines.append("Aucun précurseur validé : early_warning_rules doit être vide.")
    lines += ["", "Génère le diagnostic complet au format JSON conformément aux instructions."]
    return "\n".join(lines)


def _str(v: Any, default: str, limit: int = 2000) -> str:
    return v.strip()[:limit] if isinstance(v, str) and v.strip() else default


def _defaults(ctx: dict) -> dict:
    titles = [f.get("fine_title") for f in ctx.get("fine_clusters", []) if f.get("fine_title")]
    title = ctx.get("shared_remediation") or (titles[0] if titles else f"Anomalie groupe {ctx.get('merged_cluster_id')}")
    return {
        "problem_title": title,
        "problem_description": "Description non disponible.",
        "common_cause": "Cause indéterminée.",
        "system_impact": "Impact opérationnel non précisé.",
        "prevented_outcomes": [],
        "recommendations": [],
        "early_warning_rules": [],
    }


def _validate(result: Any, ctx: dict) -> dict:
    out = _defaults(ctx)
    if not isinstance(result, dict):
        return out
    out["problem_title"] = _str(result.get("problem_title"), out["problem_title"], 200)
    for k in ("problem_description", "common_cause", "system_impact"):
        out[k] = _str(result.get(k), out[k])

    po = result.get("prevented_outcomes")
    if isinstance(po, list):
        out["prevented_outcomes"] = [s.strip()[:500] for s in po if isinstance(s, str) and s.strip()]

    valid_by_scope = {"equipment": set(), "site": set()}
    for c in ctx.get("concentration", []):
        dim = {"equipment_id": "equipment", "site": "site"}.get(c.get("dimension"))
        if dim and c.get("value") is not None:
            valid_by_scope[dim].add(str(c["value"]))
    recs = result.get("recommendations")
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict) or not _str(r.get("action"), ""):
                continue
            scope = r.get("scope") if isinstance(r.get("scope"), str) and r["scope"] in VALID_SCOPES else "fleet"
            raw = r.get("targets")
            raw = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) else [])
            allowed = valid_by_scope.get(scope, set())
            targets = [t for t in raw if isinstance(t, str) and t in allowed]
            if scope == "fleet":
                targets = []
            elif not targets:
                scope = "fleet"
            out["recommendations"].append({
                "action": r["action"].strip()[:2000],
                "scope": scope,
                "targets": targets,
                "rationale": _str(r.get("rationale"), ""),
            })

    valid_rules = {r["rule_id"] for r in ctx.get("precursor_rules", [])}
    ewr = result.get("early_warning_rules")
    if isinstance(ewr, list):
        for r in ewr:
            if (isinstance(r, dict) and isinstance(r.get("rule_id"), str)
                    and r["rule_id"] in valid_rules):
                out["early_warning_rules"].append({
                    "rule_id": r["rule_id"],
                    "rule": _str(r.get("rule"), ""),
                    "action": _str(r.get("action"), ""),
                })
    return out


def diagnose_merged_cluster(context: dict, llm: Callable = call_llm_json) -> dict:
    """Diagnostic final d'un cluster fusionné ; ne lève jamais d'exception."""
    try:
        result = llm(
            system_prompt=SYSTEM_DIAGNOSIS_PROMPT_V2,
            user_prompt=_build_user_prompt(context),
            cache_tag="diagnosis_v2",
        )
    except Exception:
        result = None
    try:
        return _validate(result, context)
    except Exception:
        return _defaults(context)
