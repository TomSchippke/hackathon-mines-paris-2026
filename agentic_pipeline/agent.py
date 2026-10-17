"""Agent LLM de reclassement des incidents isolés vers les clusters fins."""

from __future__ import annotations

from typing import Any, Callable

from .config import OUTLIER_LABEL
from .llm_client import call_llm_json

SYSTEM_OUTLIER_PROMPT = """Tu es un expert en classification d'incidents industriels.
Nous avons identifié une taxonomie de clusters fins d'incidents (un cluster = un mode de défaillance), établie sur un même type d'actif. Plusieurs incidents ont été initialement isolés (bruit / outliers).

Ta mission : pour chaque incident de la liste fournie, détermine s'il correspond clairement à l'un des clusters existants (même mode de défaillance), ou s'il s'agit d'un cas isolé ou d'une nouvelle sous-catégorie émergente.

Format de réponse attendu : un objet JSON contenant la clé "classifications", qui est une liste d'objets avec :
- "incident_id": (l'identifiant de l'incident)
- "assigned_cluster_id": l'identifiant du cluster sélectionné parmi les clusters existants, ou "isole_atypique" si l'incident est un vrai cas unique sans rapport avec la taxonomie.
- "reason": brève justification en français (1 phrase).

Réponds STRICTEMENT avec un objet JSON valide."""


def reassign_outliers_batch(
    outliers_data: list[dict[str, Any]],
    clusters_taxonomy: list[dict[str, str]],
    llm: Callable[..., dict[str, Any]] = call_llm_json,
) -> list[dict[str, Any]]:
    """Reclasse un lot d'incidents isolés vers les clusters fins existants (même partition).

    `clusters_taxonomy` : liste de {"cluster_id": fine_id, "problem_title": fine_title,
    "common_cause": mode de défaillance}. Robuste aux réponses LLM mal formées.
    """
    taxonomy_str = "\n".join(
        f"- ID: {c['cluster_id']} | Titre: {c.get('problem_title', '')} | "
        f"Mode de défaillance: {c.get('common_cause', '')}"
        for c in clusters_taxonomy
    )

    incidents_str = "\n".join(
        f"Incident ID {item['incident_id']} : {str(item['text']).strip()}"
        for item in outliers_data
    )

    user_prompt = f"""Clusters existants dans la taxonomie :
{taxonomy_str}

Incidents à évaluer :
{incidents_str}

Pour chaque incident, indique l'affectation optimale dans le JSON de réponse."""

    try:
        result = llm(
            system_prompt=SYSTEM_OUTLIER_PROMPT,
            user_prompt=user_prompt,
            cache_tag="outlier_reassignment",
        )
    except Exception:
        result = None

    classifications = result.get("classifications", []) if isinstance(result, dict) else []
    if not isinstance(classifications, list):
        classifications = []
    mapping = {
        str(c.get("incident_id")): c
        for c in classifications
        if isinstance(c, dict) and "incident_id" in c
    }

    valid_cluster_ids = {str(c["cluster_id"]) for c in clusters_taxonomy}
    final_results = []
    for item in outliers_data:
        c_info = mapping.get(str(item["incident_id"]), {})
        assigned = str(c_info.get("assigned_cluster_id", OUTLIER_LABEL))
        if assigned not in valid_cluster_ids:
            assigned = OUTLIER_LABEL
        final_results.append({
            "incident_id": item["incident_id"],
            "assigned_cluster_id": assigned,
            "reason": c_info.get("reason") or "Analyse causale LLM.",
        })

    return final_results
