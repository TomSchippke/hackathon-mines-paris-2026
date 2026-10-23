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
VALID_SCOPES = ("fleet", "site", "equipment")
FALLBACK_ACTION = "Aucune action spécifique identifiée : inspecter pour établir la cause."
MAX_DETAIL_WORDS = 12
# Refus par le code, jamais montrés au LLM : verbes qui ne disent pas quoi faire, composants qui ne
# désignent rien de précis (formes simplifiées, sans accents)
GENERIC_VERBS = {"surveiller", "intervenir", "planifier", "effectuer", "realiser", "gerer", "assurer",
                 "ameliorer", "optimiser", "suivre", "traiter", "prevoir", "envisager", "mettre", "faire",
                 "proceder", "poursuivre", "maintenir"}
GENERIC_COMPONENTS = {"anomalie", "anomalies", "equipement", "equipements", "systeme", "systemes",
                      "installation", "installations", "materiel", "element", "elements", "composant",
                      "composants", "defaut", "defauts", "probleme", "problemes", "maintenance", "intervention",
                      "zone", "partie", "piece", "pieces", "site", "parc", "ensemble"}
STOPWORDS = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "a", "au", "aux", "sur", "pour"}

SYSTEM_DIAGNOSIS_PROMPT_V2 = """Tu es un expert en fiabilité industrielle (maintenance, retour d'expérience).
Le cluster analysé représente UNE voie de remédiation : un ensemble de défaillances qu'une même action corrective permet d'éviter.

Ta mission : produire le diagnostic final de ce cluster à partir des données fournies (clusters fins membres avec des exemples de rapports, remédiation partagée, concentrations statistiquement significatives, séries en escalade, impact).

Règles strictes :
- Une cible "equipment" doit être un équipement listé dans la section « Concentrations » ou dans la section « Séries en escalade », une cible "site" un site listé dans « Concentrations » ; les périodes ne sont jamais des cibles (citables dans la justification).
- Chaque recommandation décrit UNE action physique concrète sur UN composant ou sous-système précis, tirée de ce que montrent les rapports. Le composant est nommé avec les mots des rapports : il sera recherché dans les rapports du cluster.
- Au plus une recommandation par composant : les recommandations d'un cluster sont distinctes.
- Utilise scope "equipment" ou "site" UNIQUEMENT pour des cibles listées : c'est une intervention sur ces cibles. Sinon scope "fleet" avec targets vide : c'est un programme appliqué à tous les équipements du même type, pas la réparation d'une pièce.
- Une série en escalade est la même famille de défaillance qui s'aggrave sur la même machine : un équipement en escalade appelle une action ciblée sur cet équipement.
- S'il n'y a aucune concentration, dis-le explicitement au lieu d'inventer.
- La cause commune doit être celle que disent les rapports : justifie-la par 1 à 3 citations RECOPIÉES MOT POUR MOT dans les exemples de rapports fournis. Elles seront vérifiées dans les rapports du cluster ; une cause commune sans citation vérifiée est signalée comme déduite, non établie.

Format de réponse : un objet JSON STRICT avec les clés :
- "problem_title": titre court (string)
- "problem_description": description du problème (string)
- "common_cause": cause racine commune (string)
- "common_cause_quotes": liste de 1 à 3 citations (strings) recopiées mot pour mot
- "recommendations": liste d'objets {"verbe": verbe d'action à l'infinitif, "composant": composant ou sous-système visé, "detail": précision sur ce qu'il faut contrôler, remplacer ou régler, qui ne répète ni le verbe ni le composant (12 mots au plus, chaîne vide si rien à ajouter), "scope": "fleet"|"site"|"equipment", "targets": [string], "rationale"}"""


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


# Relance des recommandations refusées par le code (verbe générique, composant absent des rapports)
SYSTEM_ACTION_RETRY_PROMPT = """Tu es ingénieur fiabilité. On te donne les modes de défaillance et des \
rapports d'incidents d'un même groupe, et des recommandations refusées, chacune avec la raison du refus. \
Corrige chaque recommandation pour qu'elle décrive UNE action physique concrète sur UN composant ou \
sous-système précis, en restant strictement dans ce que montrent les rapports. Le composant est nommé avec \
les mots des rapports : il y sera recherché. La portée et les cibles ne changent pas.
Réponds en JSON : {"recommendations": [{"index": 1, "verbe": "verbe d'action à l'infinitif", \
"composant": "...", "detail": "précision qui ne répète ni le verbe ni le composant, 12 mots au plus, \
chaîne vide si rien à ajouter"}]}"""


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
        "recommendations": [],
        "nominal": False,
        "nominal_proposed": False,
        "nominal_reason": "",
        "nominal_external_condition": "",
        "nominal_external_quote": "",
    }


def _fallback_recommendation() -> dict:
    return {"action": FALLBACK_ACTION, "verb": "", "component": "", "detail": "", "scope": "fleet", "targets": [],
            "target_reason": {}, "targets_to_verify": [], "rationale": "", "confirm_cause_first": False,
            "critic_verdict": "fallback"}


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


def _fields(r: dict) -> tuple[str, str, str]:
    """(verbe, composant, détail) d'une réponse LLM ; détail tronqué à MAX_DETAIL_WORDS mots (« … »)."""
    words = _str(r.get("detail"), "", 300).split()
    detail = " ".join(words[:MAX_DETAIL_WORDS]) + ("…" if len(words) > MAX_DETAIL_WORDS else "")
    return _str(r.get("verbe"), "", 60), _str(r.get("composant"), "", 200), detail


def _action_text(verb: str, component: str, detail: str) -> str:
    text = f"{verb[:1].upper()}{verb[1:]} {component}".strip()
    return f"{text} : {detail}" if detail else text


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word[-1] in "sx" else word


def _grounded(component: str, texts: list[str]) -> bool:
    """Le composant figure-t-il dans un rapport ? Tous ses mots pleins (au singulier) dans un même rapport."""
    words = [_stem(w) for w in simplify(component).split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return False
    return any(all(w in t for w in words) for t in (simplify(x) for x in texts))


def _refusal(verb: str, component: str, texts: list[str]) -> str:
    """Raison du refus d'une action par le code, chaîne vide si elle est acceptée."""
    v = simplify(verb).split()
    if not v or not v[0].endswith(("er", "ir", "re")):
        return "le verbe doit être un verbe d'action à l'infinitif"
    if v[0] in GENERIC_VERBS:
        return f"le verbe « {verb} » ne dit pas quoi faire physiquement"
    words = [w for w in simplify(component).split() if w not in STOPWORDS]
    if not words or all(w in GENERIC_COMPONENTS for w in words):
        return f"le composant « {component} » ne désigne pas un composant ou sous-système précis"
    if not _grounded(component, texts):
        return f"le composant « {component} » ne figure dans aucun rapport du groupe"
    return ""


def _set_fields(r: dict, verb: str, component: str, detail: str, verdict: str) -> None:
    r.update(verb=verb, component=component, detail=detail, action=_action_text(verb, component, detail),
             critic_verdict=verdict)


def _retry_prompt(ctx: dict, failed: list[tuple[dict, str]]) -> str:
    texts = [str(t).strip()[:MAX_TEXT_CHARS] for t in (ctx.get("member_texts") or [])[:MAX_RETRY_TEXTS]]
    lines = ["Modes de défaillance du groupe :"] + [f"- {m}" for m in _failure_modes(ctx)]
    lines += ["", "Rapports du groupe :"] + [f"- {t}" for t in texts]
    lines += ["", "Recommandations refusées :"]
    for i, (r, reason) in enumerate(failed, 1):
        lines.append(f"{i}. verbe : {r['verb']} | composant : {r['component']} | détail : {r['detail']} "
                     f"| portée : {r['scope']} | refus : {reason}")
    return "\n".join(lines)


def _criticize(ctx: dict, out: dict, llm: Callable) -> None:
    """Contrôle des actions par le code : verbe d'action non générique, composant précis et retrouvé
    dans les rapports du cluster. Les actions refusées sont renvoyées une fois au LLM avec la raison
    du refus (seuls verbe, composant et détail peuvent changer) ; refusées à nouveau, elles sont
    écartées. Sans action restante : action par défaut. Doublons (verbe, composant, portée) fusionnés."""
    texts = ctx.get("member_texts") or []
    failed: list[tuple[dict, str]] = []
    for r in out["recommendations"]:
        reason = _refusal(r["verb"], r["component"], texts)
        if reason:
            failed.append((r, reason))
        else:
            r["critic_verdict"] = "valide"
    if failed:
        try:
            res = llm(system_prompt=SYSTEM_ACTION_RETRY_PROMPT, user_prompt=_retry_prompt(ctx, failed),
                      cache_tag="action_retry")
            fixes = res.get("recommendations") if isinstance(res, dict) else None
        except Exception:
            fixes = None
        by_index: dict[int, dict] = {}
        for f in fixes if isinstance(fixes, list) else []:
            idx = f.get("index") if isinstance(f, dict) else None
            if isinstance(idx, str) and idx.strip().isdigit():
                idx = int(idx.strip())
            elif isinstance(idx, float) and idx.is_integer():
                idx = int(idx)
            if isinstance(idx, int) and not isinstance(idx, bool) and idx not in by_index:
                by_index[idx] = f
        for i, (r, _) in enumerate(failed, 1):
            verb, component, detail = _fields(by_index.get(i, {}))
            if not _refusal(verb, component, texts):
                _set_fields(r, verb, component, detail, "corrigee")
    kept: dict[tuple, dict] = {}
    for r in out["recommendations"]:
        if r["critic_verdict"] not in ("valide", "corrigee"):
            continue
        key = (simplify(r["verb"]), simplify(r["component"]), r["scope"])
        if key in kept:
            k = kept[key]
            k["targets"] = list(dict.fromkeys(k["targets"] + r["targets"]))
            k["target_reason"] = {**r["target_reason"], **k["target_reason"]}
            k["targets_to_verify"] = [t for t in k["targets"] if t in set(k["targets_to_verify"]) | set(r["targets_to_verify"])]
        else:
            kept[key] = r
    out["recommendations"] = list(kept.values()) or [_fallback_recommendation()]


def _validate(result: Any, ctx: dict) -> dict:
    out = _defaults(ctx)
    if not isinstance(result, dict):
        return out
    out["problem_title"] = _str(result.get("problem_title"), out["problem_title"], 200)
    for k in ("problem_description", "common_cause"):
        out[k] = _str(result.get(k), out[k])

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
            if not isinstance(r, dict):
                continue
            verb, component, detail = _fields(r)
            if not verb and not component:
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
                "action": _action_text(verb, component, detail),
                "verb": verb,
                "component": component,
                "detail": detail,
                "scope": scope,
                "targets": targets,
                "target_reason": reason,
                # cibles soutenues seulement par une concentration à preuve faible (pas par une escalade) : à confirmer
                "targets_to_verify": [t for t in targets if t in weak_by_scope.get(scope, set()) and reason[t] == "concentration"],
                "rationale": _str(r.get("rationale"), ""),
                "confirm_cause_first": False,
                "critic_verdict": "",
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
    -> fonctionnement nominal -> contrôle des actions (clusters non nominaux) -> ordre."""
    try:
        result = llm(
            system_prompt=SYSTEM_DIAGNOSIS_PROMPT_V2,
            user_prompt=_build_user_prompt(context),
            cache_tag="diagnosis_v5",
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
            out["recommendations"] = [_fallback_recommendation()]
        # cause seulement déduite : la confirmer sur place avant toute action ciblée
        inferred = out["common_cause_source"] == "inferee"
        for r in out["recommendations"]:
            r["confirm_cause_first"] = inferred and r["scope"] != "fleet"
        out["recommendations"].sort(key=_rec_rank)
    return out
