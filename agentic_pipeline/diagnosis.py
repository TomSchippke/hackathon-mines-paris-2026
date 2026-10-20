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
MAX_RETRY_TEXTS = 30
MAX_CRITIC_TEXTS = 6
VALID_SCOPES = ("fleet", "site", "equipment")
FALLBACK_ACTION = "Aucune action spécifique identifiée : inspecter pour établir la cause."

SYSTEM_DIAGNOSIS_PROMPT_V2 = """Tu es un expert en fiabilité industrielle (maintenance, retour d'expérience).
Le cluster analysé représente UNE voie de remédiation : un ensemble de défaillances qu'une même action corrective permet d'éviter.

Ta mission : produire le diagnostic final de ce cluster à partir des données fournies (clusters fins membres avec des exemples de rapports, remédiation partagée, concentrations statistiquement significatives, séries en escalade, règles précurseurs, impact).

Règles strictes :
- Énumère chaque défaillance évitée (prevented_outcomes), une phrase courte par résultat.
- Une cible "equipment" doit être un équipement listé dans la section « Concentrations » ou dans la section « Séries en escalade », une cible "site" un site listé dans « Concentrations » ; les périodes ne sont jamais des cibles (citables dans la justification).
- Chaque recommandation décrit UNE action physique concrète sur UN composant ou sous-système précis, tirée de ce que montrent les rapports. Utilise scope "equipment" ou "site" UNIQUEMENT pour des cibles listées ; sinon scope "fleet" avec targets vide.
- Une série en escalade est la même famille de défaillance qui s'aggrave sur la même machine : un équipement en escalade appelle une action ciblée sur cet équipement.
- Les règles d'alerte précoce (early_warning_rules) ne peuvent utiliser QUE des rule_id fournis dans la section « Règles précurseurs ».
- S'il n'y a aucune concentration ou aucun précurseur, dis-le explicitement (liste vide) au lieu d'inventer.
- La cause commune doit être celle que disent les rapports : justifie-la par 1 à 3 citations RECOPIÉES MOT POUR MOT dans les exemples de rapports fournis. Elles seront vérifiées dans les rapports du cluster ; une cause commune sans citation vérifiée est signalée comme déduite, non établie.

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
- "external_condition" : la condition extérieure identifiée qui a provoqué le déclenchement, en quelques \
mots ; chaîne vide si les rapports n'en mentionnent aucune.
- "external_condition_quote" : le passage d'un rapport qui mentionne cette condition, RECOPIÉ MOT POUR \
MOT ; chaîne vide si aucun passage ne la mentionne. Il sera vérifié dans les rapports.
- "reason" : une phrase qui justifie tes réponses."""


# Relance de la cause commune : citations sur un échantillon plus large de rapports
SYSTEM_CAUSE_RETRY_PROMPT = """Tu es ingénieur fiabilité. On te donne les modes de défaillance et des rapports \
d'incidents d'un même groupe, ainsi qu'une cause commune proposée qui n'a pas pu être justifiée par une \
citation. Réponds en JSON :
- "common_cause" : la cause commune telle que la disent les rapports ; corrige la cause proposée si les \
rapports disent autre chose. Si aucun passage ne l'établit, donne la cause la plus probable déduite des \
modes de défaillance.
- "quotes" : 1 à 3 passages des rapports, RECOPIÉS MOT POUR MOT, qui établissent cette cause ; liste vide \
si aucun passage ne l'établit (cause déduite). Les passages seront vérifiés dans les rapports."""


# Relecture des recommandations : écarte les actions génériques ou non étayées
SYSTEM_RECOMMENDATION_CRITIC_PROMPT = (
    "Tu es un relecteur critique en fiabilité industrielle. On te donne les recommandations "
    "proposées pour un groupe d'incidents, numérotées, avec les modes de défaillance et des extraits "
    "de rapports du groupe. Une recommandation n'est utile que si elle décrit UNE action physique "
    "concrète sur UN composant ou sous-système précis. Par défaut, sois sévère.\n"
    "Pour chaque recommandation, rends un verdict :\n"
    "- \"keep\" : elle est déjà spécifique (une action concrète sur un composant ou sous-système "
    "nommé) ;\n"
    "- \"rewrite\" : elle est générique, mais les rapports présentés permettent d'en tirer une action "
    "concrète sur un composant ou sous-système précis : réécris-la en restant strictement dans ce que "
    "montrent les rapports ;\n"
    "- \"drop\" : elle n'est pas récupérable ou n'est pas étayée par les rapports présentés.\n"
    "Est générique notamment : programme de maintenance, inspection régulière, surveillance "
    "renforcée, formation du personnel, contrôle systématique, revue des procédures...\n"
    "Une action réécrite n'enchaîne pas des actions distinctes (« et », « ou », « / ») et ne repose "
    "sur aucune hypothèse non visible dans les rapports. Tu ne juges que l'action : la portée et les "
    "cibles ne changent pas.\n"
    "Pour keep et rewrite, nomme le composant et l'action ; si tu ne peux pas nommer un seul "
    "composant, choisis drop.\n"
    'Réponds en JSON : {"recommendations": [{"index": 1, "verdict": "keep"|"rewrite"|"drop", '
    '"component": "...", "action": "...", "reason": "..."}]}'
)


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
    escalations: pd.DataFrame | None = None,
) -> dict:
    """Assemble le contexte (filtré) fourni au LLM pour un cluster fusionné.

    `member_texts` : rapports des incidents du cluster (jamais envoyés au LLM, servent à vérifier
    les citations) ; `mild_share` : part des incidents au niveau de sévérité le plus bas ;
    `escalations` : séries en escalade (colonnes `escalation.ESCALATION_COLUMNS`).
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
        mask = _truthy(p["significant"]) & _truthy(p["plausible"]) & (p["consequent"] == merged_id)
        if "kind" in p:
            mask &= p["kind"] != "recurrence"
        counts = {}
        if evidence is not None and len(evidence) and "rule_id" in evidence:
            counts = evidence["rule_id"].value_counts().to_dict()
        for r in p[mask].to_dict("records"):
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

    esc_rows: list[dict] = []
    if escalations is not None and len(escalations) and "merged_cluster_id" in escalations:
        for r in escalations[escalations["merged_cluster_id"] == merged_id].to_dict("records"):
            esc_rows.append({
                "equipment_id": str(r.get("equipment_id")),
                "n_incidents": _clean(r.get("n_incidents")),
                "severity_sequence": _clean(r.get("severity_sequence")),
                "dates": _clean(r.get("dates")),
                "total_cost_eur": _clean(r.get("total_cost_eur")),
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
        "escalations": esc_rows,
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
        lines.append("Aucune concentration significative.")
    lines += ["", "## Séries en escalade (même famille de défaillance qui s'aggrave sur la même machine ; "
                  "scope \"equipment\")"]
    if ctx.get("escalations"):
        for e in ctx["escalations"]:
            lines.append(f"- {e['equipment_id']} : {e['n_incidents']} incidents, sévérités {e['severity_sequence']} "
                         f"(dates {e['dates']}), coût total {e['total_cost_eur']} EUR")
    else:
        lines.append("Aucune série en escalade.")
    if not ctx["concentration"] and not ctx.get("escalations"):
        lines.append("Aucune cible listée : utiliser uniquement scope \"fleet\".")
    lines += ["", "## Règles précurseurs (plausibles, significatives)"]
    if ctx["precursor_rules"]:
        for r in ctx["precursor_rules"]:
            lines.append(f"- rule_id : {r['rule_id']} | lift {r['lift']} | "
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
        "nominal_external_condition": "",
        "nominal_external_quote": "",
    }


def _fallback_recommendation() -> dict:
    return {"action": FALLBACK_ACTION, "component": "", "scope": "fleet", "targets": [], "target_reason": {},
            "targets_to_verify": [], "rationale": "", "confirm_cause_first": False, "critic_verdict": "fallback"}


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


def _verified_quotes(raw: Any, texts: list[str]) -> list[str]:
    """Citations (strings) retrouvées dans au moins un rapport du cluster, MAX_CAUSE_QUOTES au plus."""
    raw = raw if isinstance(raw, list) else []
    quotes = [q.strip()[:500] for q in raw if isinstance(q, str) and q.strip()]
    return [q for q in quotes if any(quote_found(q, t) for t in texts)][:MAX_CAUSE_QUOTES]


def _failure_modes(ctx: dict) -> list[str]:
    return [f"{f.get('fine_title', '')} : {f.get('failure_mode', '')}" for f in ctx.get("fine_clusters", [])]


def _cause_retry_prompt(ctx: dict, cause: str) -> str:
    texts = [str(t).strip()[:MAX_TEXT_CHARS] for t in (ctx.get("member_texts") or [])[:MAX_RETRY_TEXTS]]
    lines = [f"Cause commune proposée : {cause}", "", "Modes de défaillance :"]
    lines += [f"- {m}" for m in _failure_modes(ctx)]
    lines += ["", "Rapports du groupe :"] + [f"- {t}" for t in texts]
    return "\n".join(lines)


def _check_cause(result: Any, ctx: dict, out: dict, llm: Callable) -> None:
    """Cause commune justifiée par des citations retrouvées dans les rapports du cluster
    (`common_cause_source` = « citee »). Sinon, une relance sur MAX_RETRY_TEXTS rapports ; si elle
    n'apporte pas de citation vérifiée, la cause est marquée déduite (« inferee »)."""
    texts = ctx.get("member_texts") or []
    quotes = _verified_quotes(result.get("common_cause_quotes") if isinstance(result, dict) else None, texts)
    if quotes:
        out.update(common_cause_quotes=quotes, common_cause_source="citee")
        return
    try:
        res = llm(system_prompt=SYSTEM_CAUSE_RETRY_PROMPT, user_prompt=_cause_retry_prompt(ctx, out["common_cause"]),
                  cache_tag="cause_retry")
    except Exception:
        res = None
    res = res if isinstance(res, dict) else {}
    quotes = _verified_quotes(res.get("quotes"), texts)
    out["common_cause"] = _str(res.get("common_cause"), out["common_cause"])
    out.update(common_cause_quotes=quotes, common_cause_source="citee" if quotes else "inferee")


def _nominal_prompt(ctx: dict) -> str:
    texts = [t for f in ctx.get("fine_clusters", []) for t in f.get("representative_texts") or []]
    return "Rapports du groupe :\n" + "\n".join(f"- {t}" for t in texts)


def _check_nominal(ctx: dict, out: dict, llm: Callable) -> None:
    """Fonctionnement nominal : le LLM dit qu'aucun défaut n'a été constaté, que la conséquence est
    seulement une protection déclenchée et cite la condition extérieure ; le code exige que cette
    citation figure dans un rapport du cluster, puis que le cluster soit surtout au plus bas niveau de
    sévérité. Un cluster nominal n'a pas de recommandation."""
    try:
        res = llm(system_prompt=SYSTEM_NOMINAL_PROMPT, user_prompt=_nominal_prompt(ctx), cache_tag="nominal_check")
    except Exception:
        res = None
    res = res if isinstance(res, dict) else {}
    condition = _str(res.get("external_condition"), "", 300)
    quote = _str(res.get("external_condition_quote"), "", 500)
    texts = ctx.get("member_texts") or []
    quoted = bool(quote) and any(quote_found(quote, t) for t in texts)
    out["nominal_external_condition"] = condition
    out["nominal_external_quote"] = quote if quoted else ""
    out["nominal_proposed"] = (res.get("protection_only") is True and res.get("defect_found") is False
                               and bool(condition) and quoted)
    share = ctx.get("mild_share")
    out["nominal"] = out["nominal_proposed"] and share is not None and share >= NOMINAL_MIN_MILD_SHARE
    if out["nominal"]:
        out["nominal_reason"] = _str(res.get("reason"), "", 500)
        out["recommendations"] = []


def _critic_prompt(ctx: dict, recs: list[dict]) -> str:
    texts = [t for f in ctx.get("fine_clusters", []) for t in f.get("representative_texts") or []]
    lines = ["Modes de défaillance du groupe :"] + [f"- {m}" for m in _failure_modes(ctx)]
    lines += ["", "Extraits de rapports :"] + [f"- {t}" for t in texts[:MAX_CRITIC_TEXTS]]
    lines += ["", "Recommandations :"]
    for i, r in enumerate(recs, 1):
        lines.append(f"{i}. {r['action']}" + (f" (justification : {r['rationale']})" if r["rationale"] else ""))
    return "\n".join(lines)


def _criticize(ctx: dict, out: dict, llm: Callable) -> None:
    """Relecture des recommandations validées : keep / rewrite / drop. Le critique ne peut changer que
    l'action et le composant ; portée et cibles restent celles validées par le code. Recommandation
    non citée ou verdict invalide : écartée. Appel en échec : recommandations gardées, « non_verifie »."""
    recs = out["recommendations"]
    if not recs:
        out["recommendations"] = [_fallback_recommendation()]
        return
    try:
        res = llm(system_prompt=SYSTEM_RECOMMENDATION_CRITIC_PROMPT, user_prompt=_critic_prompt(ctx, recs),
                  cache_tag="recommendation_critic")
        verdicts = res.get("recommendations") if isinstance(res, dict) else None
    except Exception:
        verdicts = None
    if not isinstance(verdicts, list):
        for r in recs:
            r.update(critic_verdict="non_verifie", component="")
        return
    seen: dict[int, dict] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("index")
        if isinstance(idx, str) and idx.strip().isdigit():
            idx = int(idx.strip())
        elif isinstance(idx, float) and idx.is_integer():
            idx = int(idx)
        if isinstance(idx, bool) or not isinstance(idx, int) or not 1 <= idx <= len(recs) or idx in seen:
            continue
        seen[idx] = v
    if not seen:  # liste sans aucun index exploitable (vide, base 0...) : réponse mal formée
        for r in recs:
            r.update(critic_verdict="non_verifie", component="")
        return
    kept = []
    for i, r in enumerate(recs, 1):
        v = seen.get(i, {})
        verdict, component, action = v.get("verdict"), _str(v.get("component"), "", 300), _str(v.get("action"), "")
        if verdict not in ("keep", "rewrite") or not component or not action:
            continue
        r.update(component=component, critic_verdict=verdict)
        if verdict == "rewrite":
            r["action"] = action
        kept.append(r)
    out["recommendations"] = kept or [_fallback_recommendation()]


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

    conc_by_scope = {"equipment": set(), "site": set()}
    weak_by_scope = {"equipment": set(), "site": set()}
    for c in ctx.get("concentration", []):
        dim = {"equipment_id": "equipment", "site": "site"}.get(c.get("dimension"))
        if dim and c.get("value") is not None:
            conc_by_scope[dim].add(str(c["value"]))
            if c.get("evidence_level") == TO_VERIFY:
                weak_by_scope[dim].add(str(c["value"]))
    escalating = {str(e["equipment_id"]) for e in ctx.get("escalations", [])}
    valid_by_scope = {"equipment": conc_by_scope["equipment"] | escalating, "site": conc_by_scope["site"]}
    recs = result.get("recommendations")
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict) or not _str(r.get("action"), ""):
                continue
            scope = r.get("scope") if isinstance(r.get("scope"), str) and r["scope"] in VALID_SCOPES else "fleet"
            raw = r.get("targets")
            raw = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) else [])
            allowed = valid_by_scope.get(scope, set())
            targets = list(dict.fromkeys(t for t in raw if isinstance(t, str) and t in allowed))
            if scope == "fleet":
                targets = []
            elif not targets:
                scope = "fleet"
            reason = {}
            for t in targets:
                in_conc = t in conc_by_scope.get(scope, set())
                in_esc = scope == "equipment" and t in escalating
                reason[t] = "both" if in_conc and in_esc else ("concentration" if in_conc else "escalation")
            out["recommendations"].append({
                "action": r["action"].strip()[:2000],
                "component": "",
                "scope": scope,
                "targets": targets,
                "target_reason": reason,
                # cibles soutenues seulement par une concentration à preuve faible (pas par une escalade) : à confirmer
                "targets_to_verify": [t for t in targets if t in weak_by_scope.get(scope, set()) and reason[t] == "concentration"],
                "rationale": _str(r.get("rationale"), ""),
                "confirm_cause_first": False,
                "critic_verdict": "",
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


def _rec_rank(r: dict) -> int:
    """Escalade d'abord, puis les autres actions ciblées (site / équipement), puis la flotte."""
    if any(v in ("escalation", "both") for v in r.get("target_reason", {}).values()):
        return 0
    return 1 if r.get("scope") != "fleet" else 2


def diagnose_merged_cluster(context: dict, llm: Callable = call_llm_json) -> dict:
    """Diagnostic final d'un cluster fusionné ; ne lève jamais d'exception.

    Étapes : appel principal -> validation -> cause commune (citée, sinon relance, sinon déduite)
    -> fonctionnement nominal -> critique des recommandations (clusters non nominaux) -> ordre."""
    try:
        result = llm(
            system_prompt=SYSTEM_DIAGNOSIS_PROMPT_V2,
            user_prompt=_build_user_prompt(context),
            cache_tag="diagnosis_v4",
        )
    except Exception:
        result = None
    try:
        out = _validate(result, context)
    except Exception:
        out = _defaults(context)
    try:
        _check_cause(result, context, out, llm)
    except Exception:
        out.update(common_cause_quotes=[], common_cause_source="inferee")
    try:
        _check_nominal(context, out, llm)
    except Exception:
        out.update(nominal=False, nominal_proposed=False)
    if not out["nominal"]:
        try:
            _criticize(context, out, llm)
        except Exception:
            for r in out["recommendations"]:
                r.update(critic_verdict="non_verifie", component="")
        # cause seulement déduite : la confirmer sur place avant toute action ciblée
        inferred = out["common_cause_source"] == "inferee"
        for r in out["recommendations"]:
            r["confirm_cause_first"] = inferred and r["scope"] != "fleet"
        out["recommendations"].sort(key=_rec_rank)
    return out
