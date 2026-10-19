"""Diagnostic final des clusters fusionnés (voies de remédiation), avec validation stricte."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import Any, Callable

import pandas as pd

from .concentration import TO_VERIFY
from .config import NOMINAL_MIN_MILD_SHARE, QUOTE_MIN_MATCH
from .llm_client import call_llm_json

MAX_TEXTS_PER_FINE = 2
MAX_TEXTS_TOTAL = 8
MAX_TEXT_CHARS = 600
MAX_CAUSE_QUOTES = 3
VALID_SCOPES = ("fleet", "site", "equipment")

SYSTEM_DIAGNOSIS_PROMPT_V2 = """Tu es un expert en fiabilité industrielle (maintenance, retour d'expérience).
Le cluster analysé représente UNE voie de remédiation : un ensemble de défaillances qu'une même action corrective permet d'éviter.

Ta mission : produire le diagnostic final de ce cluster à partir des données fournies (clusters fins membres avec des exemples de rapports, remédiation partagée, concentrations statistiquement significatives, règles précurseurs, impact).

Règles strictes :
- Énumère chaque défaillance évitée (prevented_outcomes), une phrase courte par résultat.
- Une cible "equipment" doit être un équipement listé, une cible "site" un site listé ; les périodes ne sont jamais des cibles (citables dans la justification).
- Les recommandations doivent être spécifiques et actionnables. Utilise scope "equipment" ou "site" UNIQUEMENT pour des cibles listées dans la section « Concentrations » ; sinon scope "fleet" avec targets vide.
- Les règles d'alerte précoce (early_warning_rules) ne peuvent utiliser QUE des rule_id fournis dans la section « Règles précurseurs ».
- S'il n'y a aucune concentration ou aucun précurseur, dis-le explicitement (liste vide) au lieu d'inventer.
- La cause commune doit être celle que disent les rapports : justifie-la par 1 à 3 citations RECOPIÉES MOT POUR MOT dans les exemples de rapports fournis. Elles seront vérifiées dans les rapports du cluster ; une cause commune sans citation vérifiée est signalée comme non justifiée.

Format de réponse : un objet JSON STRICT avec les clés :
- "problem_title": titre court (string)
- "problem_description": description du problème (string)
- "common_cause": cause racine commune (string)
- "common_cause_quotes": liste de 1 à 3 citations (strings) recopiées mot pour mot
- "system_impact": impact opérationnel et financier (string)
- "prevented_outcomes": liste de strings
- "recommendations": liste d'objets {"action", "scope": "fleet"|"site"|"equipment", "targets": [string], "rationale"}
- "early_warning_rules": liste d'objets {"rule_id", "rule", "action"}"""


# Fonctionnement nominal : deux questions séparées, plus fiables qu'un jugement global
SYSTEM_NOMINAL_PROMPT = """Tu es ingénieur fiabilité. On te donne des extraits de rapports d'incidents d'un \
même groupe. Réponds en JSON :
- "defect_found" : true si, dans la plupart de ces incidents, un défaut, une dégradation ou une anomalie \
a été constaté sur l'équipement, même mineur, sans dommage visible ou découvert lors d'un contrôle ; false sinon.
- "protection_only" : true si, dans la plupart de ces incidents, un dispositif de protection ou de mise en \
sécurité s'est déclenché (arrêt, coupure) en réaction à une condition extérieure identifiée, et que ce \
déclenchement est la seule conséquence avant une reprise normale ; false sinon (simple alerte ou \
surveillance, déclenchement sans cause extérieure, dégât, panne).
- "reason" : une phrase qui justifie tes deux réponses."""


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
    member_texts: list[str] | None = None,
    mild_share: float | None = None,
) -> dict:
    """Assemble le contexte (filtré) fourni au LLM pour un cluster fusionné.

    `member_texts` : rapports des incidents du cluster (jamais envoyés au LLM, servent à vérifier
    les citations) ; `mild_share` : part des incidents au niveau de sévérité le plus bas.
    """
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
                              ("dimension", "value", "observed", "expected", "lift", "q_value", "evidence_level")})

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
        "member_texts": list(member_texts or []),
        "mild_share": mild_share,
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
    if ctx.get("mild_share") is not None:
        lines.append(f"Part des incidents au niveau de sévérité le plus bas : {ctx['mild_share']:.0%}")
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
        "common_cause_quotes": [],
        "common_cause_source": "",
        "system_impact": "Impact opérationnel non précisé.",
        "prevented_outcomes": [],
        "recommendations": [],
        "early_warning_rules": [],
        "nominal": False,
        "nominal_proposed": False,
        "nominal_reason": "",
    }


def simplify(text: Any) -> str:
    """Minuscules, sans accents ni ponctuation : tolère les petites variations de recopie."""
    text = unicodedata.normalize("NFKD", str(text or "").lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def quote_found(quote: Any, text: Any) -> bool:
    """L'extrait cité figure-t-il dans le texte (à QUOTE_MIN_MATCH près) ? Trois mots au moins."""
    q, t = simplify(quote), simplify(text)
    if len(q.split()) < 3 or not t:
        return False
    if q in t:
        return True
    m = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size >= QUOTE_MIN_MATCH * len(q)


def _check_cause(result: dict, ctx: dict, out: dict) -> None:
    """Citations gardées si elles figurent dans un rapport du cluster ; sans citation vérifiée, la
    cause commune est marquée non justifiée (`common_cause_source` = « non_citee »)."""
    raw = result.get("common_cause_quotes")
    raw = raw if isinstance(raw, list) else []
    texts = ctx.get("member_texts") or []
    quotes = [q.strip()[:500] for q in raw if isinstance(q, str) and q.strip()]
    quotes = [q for q in quotes if any(quote_found(q, t) for t in texts)][:MAX_CAUSE_QUOTES]
    out.update(common_cause_quotes=quotes, common_cause_source="citee" if quotes else "non_citee")


def _nominal_prompt(ctx: dict) -> str:
    texts = [t for f in ctx.get("fine_clusters", []) for t in f.get("representative_texts") or []]
    return "Rapports du groupe :\n" + "\n".join(f"- {t}" for t in texts)


def _check_nominal(ctx: dict, out: dict, llm: Callable) -> None:
    """Fonctionnement nominal : le LLM dit qu'aucun défaut n'a été constaté et que la conséquence est
    seulement une protection déclenchée ; le code exige en plus que le cluster soit surtout au plus
    bas niveau de sévérité. Un cluster nominal n'a pas de recommandation."""
    try:
        res = llm(system_prompt=SYSTEM_NOMINAL_PROMPT, user_prompt=_nominal_prompt(ctx), cache_tag="nominal_check")
    except Exception:
        res = None
    res = res if isinstance(res, dict) else {}
    out["nominal_proposed"] = res.get("protection_only") is True and res.get("defect_found") is False
    share = ctx.get("mild_share")
    out["nominal"] = out["nominal_proposed"] and share is not None and share >= NOMINAL_MIN_MILD_SHARE
    if out["nominal"]:
        out["nominal_reason"] = _str(res.get("reason"), "", 500)
        out["recommendations"] = []


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
    weak_by_scope = {"equipment": set(), "site": set()}
    for c in ctx.get("concentration", []):
        dim = {"equipment_id": "equipment", "site": "site"}.get(c.get("dimension"))
        if dim and c.get("value") is not None:
            valid_by_scope[dim].add(str(c["value"]))
            if c.get("evidence_level") == TO_VERIFY:
                weak_by_scope[dim].add(str(c["value"]))
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
                # cibles dont la concentration n'a qu'une preuve faible : à confirmer sur place
                "targets_to_verify": [t for t in targets if t in weak_by_scope.get(scope, set())],
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
    _check_cause(result, ctx, out)
    return out


def diagnose_merged_cluster(context: dict, llm: Callable = call_llm_json) -> dict:
    """Diagnostic final d'un cluster fusionné ; ne lève jamais d'exception."""
    try:
        result = llm(
            system_prompt=SYSTEM_DIAGNOSIS_PROMPT_V2,
            user_prompt=_build_user_prompt(context),
            cache_tag="diagnosis_v3",
        )
    except Exception:
        result = None
    try:
        out = _validate(result, context)
    except Exception:
        out = _defaults(context)
    _check_nominal(context, out, llm)
    return out
