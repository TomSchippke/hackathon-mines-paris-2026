"""Agents LLM pour la caractérisation des clusters et le reclassement des outliers."""

from __future__ import annotations

import json
from typing import Any

from .llm_client import call_llm_json

SYSTEM_DIAGNOSIS_PROMPT = """Tu es un expert senior en ingénierie de fiabilité industrielle, maintenance prédictive et analyse causale d'incidents (énergie, industrie, infrastructures).

Ton rôle est d'analyser un groupe (cluster) d'incidents récurrents pour produire un diagnostic précis, rigoureux et exploitable par les équipes opérationnelles.

Pour le cluster qui t'est présenté, tu dois obligatoirement générer un objet JSON avec les clés suivantes (tous les textes en français) :
- "problem_title" : Intitulé court, technique et précis du mode de défaillance ou du problème principal (ex: "Dégradation d'isolation et défaut d'arc électrique", "Vibrations anormales et micro-fissuration des pales").
- "problem_description" : Synthèse factuelle et claire de ce qui se produit concrètement lors de ces incidents.
- "common_cause" : Analyse de la cause racine commune ou des facteurs déclencheurs récurrents (usure prématurée, surchauffe, intempéries, défaut de composant, etc.).
- "system_impact" : Conséquences opérationnelles sur le système (perte de rendement, déclenchement de protections, arrêt d'urgence, risque incendie...).
- "actionable_recommendations" : 2 à 3 actions préventives ou correctives concrètes et actionnables pour réduire ou éradiquer cette défaillance.

Réponds STRICTEMENT avec un objet JSON valide, sans texte d'introduction ni conclusion."""


def diagnose_cluster(
    cluster_id: str,
    keywords: list[str],
    representative_texts: list[str],
    metadata_profile: dict[str, Any],
) -> dict[str, str]:
    """Appelle le LLM pour caractériser le problème, la cause et les impacts d'un cluster."""
    user_prompt = f"""Analyse le cluster d'incidents suivant :

Identifiant du cluster : {cluster_id}
Mots-clés c-TF-IDF dominants : {', '.join(keywords) if keywords else 'N/A'}
Profil statistique : {json.dumps(metadata_profile, ensure_ascii=False)}

Rapports d'incidents représentatifs de ce cluster :
"""
    for i, doc in enumerate(representative_texts, 1):
        user_prompt += f"\n--- Incident exemple {i} ---\n{doc.strip()}\n"

    user_prompt += "\nGénère le diagnostic complet au format JSON conformément aux instructions."

    result = call_llm_json(
        system_prompt=SYSTEM_DIAGNOSIS_PROMPT,
        user_prompt=user_prompt,
        cache_tag="diagnosis",
    )

    return {
        "problem_title": result.get("problem_title", f"Anomalie groupe {cluster_id}"),
        "problem_description": result.get("problem_description", "Description non disponible."),
        "common_cause": result.get("common_cause", "Cause indéterminée."),
        "system_impact": result.get("system_impact", "Impact opérationnel non précisé."),
        "actionable_recommendations": result.get("actionable_recommendations", "Planifier une inspection ciblée."),
    }


SYSTEM_OUTLIER_PROMPT = """Tu es un expert en classification d'incidents industriels.
Nous avons identifié une taxonomie de clusters d'incidents établis. Plusieurs incidents ont été initialement isolés (bruit / outliers).

Ta mission : pour chaque incident de la liste fournie, détermine s'il correspond clairement à l'un des clusters existants, ou s'il s'agit d'un cas isolé ou d'une nouvelle sous-catégorie émergente.

Format de réponse attendu : un objet JSON contenant la clé "classifications", qui est une liste d'objets avec :
- "incident_id": (l'identifiant de l'incident)
- "assigned_cluster_id": l'identifiant du cluster sélectionné parmi les clusters existants, ou "isole_atypique" si l'incident est un vrai cas unique sans rapport avec la taxonomie.
- "reason": brève justification en français (1 phrase).

Réponds STRICTEMENT avec un objet JSON valide."""


def reassign_outliers_batch(
    outliers_data: list[dict[str, Any]],
    clusters_taxonomy: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Reclasse un lot d'incidents isolés vers les clusters existants ou les catégorise."""
    taxonomy_str = "\n".join(
        f"- ID: {c['cluster_id']} | Titre: {c['problem_title']} | Cause: {c['common_cause']}"
        for c in clusters_taxonomy
    )

    incidents_str = "\n".join(
        f"Incident ID {item['incident_id']} : {item['text'].strip()}"
        for item in outliers_data
    )

    user_prompt = f"""Clusters existants dans la taxonomie :
{taxonomy_str}

Incidents à évaluer :
{incidents_str}

Pour chaque incident, indique l'affectation optimale dans le JSON de réponse."""

    result = call_llm_json(
        system_prompt=SYSTEM_OUTLIER_PROMPT,
        user_prompt=user_prompt,
        cache_tag="outlier_reassignment",
    )

    classifications = result.get("classifications", [])
    mapping = {str(c.get("incident_id")): c for c in classifications if "incident_id" in c}

    final_results = []
    for item in outliers_data:
        str_id = str(item["incident_id"])
        c_info = mapping.get(str_id, {})
        assigned = c_info.get("assigned_cluster_id", "isole_atypique")
        valid_cluster_ids = {c["cluster_id"] for c in clusters_taxonomy}
        if assigned not in valid_cluster_ids and assigned != "isole_atypique":
            assigned = "isole_atypique"
        final_results.append({
            "incident_id": item["incident_id"],
            "assigned_cluster_id": assigned,
            "reason": c_info.get("reason", "Analyse causale LLM."),
        })

    return final_results
