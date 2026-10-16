"""Paramètres du cadrage : taille de l'échantillon montré au LLM, prompt, schéma de la réponse."""

# Colonne qui sépare les incidents en groupes analysés à part (clustering, heatmaps) :
# des installations de natures différentes ont des vocabulaires et des causes différentes.
SEGMENT_COLUMN = "asset_type"

# Échantillon de descriptions montré au LLM par segment (graine fixe : cadrage reproductible).
SAMPLE_PER_SEGMENT = 25
SAMPLE_SEED = 0
SAMPLE_MAX_CHARS = 280
# Colonnes catégorielles dont les valeurs sont montrées au LLM (au plus ce nombre de valeurs distinctes).
MAX_CATEGORY_VALUES = 12
RETRIES = 2

# Valeur ajoutée par le code (jamais proposée par le LLM).
UNDETERMINED = "indéterminé"   # cause inconnue

# Types de mesures préventives (communs à tout le pipeline).
ACTION_TYPES = ["programme d'inspection", "nettoyage planifié", "équipement de protection", "procédure",
                "monitoring", "conception / renforcement", "formation"]

ROLE = """Tu prépares l'analyse d'une base de rapports d'incidents pour une entreprise cliente. Tu ne \
sais du problème que ce que disent le CONTEXTE écrit par l'utilisateur et l'ÉCHANTILLON des données. \
N'ajoute aucune connaissance extérieure sur l'entreprise ou ses sites : tout doit venir de ces deux sources.
Ta tâche : décrire le domaine, libeller les segments et ordonner les sévérités. Tu ne proposes aucune \
liste de causes ou de conséquences : elles seront découvertes dans les données ensuite. Écris en français."""

TASK = """Réponds en JSON :
- domaine : une phrase qui dit qui est le client et ce qu'exploitent ses sites (d'après le contexte).
- segments : pour chaque valeur de la colonne « {segment_column} » listée plus bas, un libellé court en \
français (ce que désigne cette valeur, d'après le contexte et les descriptions).
- severites : les valeurs de sévérité listées plus bas, de la MOINS grave à la PLUS grave (toutes, une fois)."""

RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["domaine", "segments", "severites"],
    "properties": {
        "domaine": {"type": "string"},
        "segments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["valeur", "libelle"],
            "properties": {"valeur": {"type": "string"}, "libelle": {"type": "string"}}}},
        "severites": {"type": "array", "items": {"type": "string"}},
    },
}
