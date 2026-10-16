"""Paramètres de l'extraction des causes : prompt, schéma de réponse, parallélisme."""

import re

# Appels LLM en parallèle. Peu, pour ménager le quota de débit Azure (requêtes et tokens par minute) :
# au-delà, erreurs 429 et attentes, gérées automatiquement mais plus lentes.
WORKERS = 2

# Colonnes jamais montrées au LLM : le site et les identifiants qui le contiennent (voir pipeline.py),
# et les colonnes techniques de l'extraction des PDF.
PROMPT_EXCLUDED_COLUMNS = {"incident_id", "site", "text", "equipment_id", "technician_id", "document_path",
                           "document_format", "extraction_method", "extraction_warnings"}

# Pas de liste fermée de causes, de composants ni de conséquences : ils sont découverts dans les
# données (étapes suivantes), le LLM les écrit avec les mots du rapport.

# Comment la cause a été obtenue : écrite dans le rapport, déduite d'indices du rapport, ou inconnue.
CAUSE_TYPES = ["explicite", "inférée", "inconnue"]

# Comment une cause contributive a été obtenue : écrite dans le rapport, ou déduite d'un indice du rapport.
CONTRIBUTIVE_TYPES = ["explicite", "inférée"]

# Nombre maximal de causes contributives par incident (indiqué au LLM ; pas imposé par le schéma).
MAX_CONTRIBUTIVES = 4

CONFIANCE = ["faible", "moyenne", "haute"]

# Hypothèse « manque de maintenance », déduite de la CONSÉQUENCE (encrassement accumulé, corrosion
# avancée, usure non détectée...). Champ SÉPARÉ : il ne modifie jamais cause_racine ni cause_type, qui
# restent stricts (indice vérifié). Sert d'indicateur secondaire, jamais de cause.
MAINTENANCE_SUSPECTEE = ["non", "plausible", "probable"]

# Valeur unique pour une cause inconnue. Toutes les variantes (« non précisée », « non déterminée »...)
# y sont ramenées, et ces incidents sont comptés à part (exclus du clustering des causes).
CAUSE_INCONNUE = "cause non précisée dans le rapport"

# Un indice (extrait cité par le LLM) est accepté s'il figure dans la description à QUOTE_MIN_MATCH près
# (la plus longue portion commune doit couvrir cette part de l'indice) ; sinon la cause est annulée.
QUOTE_MIN_MATCH = 0.8
UNKNOWN_CAUSE_PATTERN = re.compile(r"non (précis|determin|détermin|identifi|connu|renseign|indiqu)|inconnu|"
                                   r"pas (précis|indiqu|d'information)|aucune (cause|information|indication)",
                                   re.IGNORECASE)

SYSTEM_PROMPT = """Tu es ingénieur fiabilité. Contexte : {domaine} On te donne un rapport \
d'incident. Analyse-le et réponds en JSON.

- cause_racine : POURQUOI c'est arrivé, en une phrase courte, avec les mots du rapport. Remonte à la \
cause première, le phénomène qui a déclenché l'enchaînement, plutôt qu'au symptôme final (pas \
« court-circuit » si le rapport dit ce qui l'a provoqué). Ne décris pas les dégâts ici.
- cause_type :
  « explicite » si la DESCRIPTION écrit la cause ;
  « inférée » seulement si la description ne l'écrit pas mais contient un indice concret qui la \
désigne (par exemple « usure visible », « lors de travaux », un événement survenu juste avant). \
Les conditions du jour seules (champs du rapport hors description) ne suffisent pas : l'indice doit \
être dans la description et lié à l'incident ;
  « inconnue » dans tous les autres cas. Écris alors exactement « {cause_inconnue} » dans \
cause_racine et laisse indice vide. En cas de doute, choisis « inconnue » : une cause inventée est \
pire qu'une cause inconnue.
- indice : pour une cause racine explicite ou inférée, l'extrait de la description qui la justifie, \
RECOPIÉ MOT POUR MOT (10 à 20 mots). Il sera vérifié : un extrait absent du rapport annule la cause.
- causes_contributives : les CONDITIONS qui ont permis l'incident ou l'ont aggravé (exposition ou \
environnement décrit dans le rapport, état dégradé constaté, équipement sans protection, consigne absente, \
défaut signalé et non traité...), \
sans être le phénomène qui a déclenché l'incident. Liste vide si aucune, quatre au maximum. Chaque \
élément a ses propres champs :
  « texte » : la condition, en une phrase, avec les mots du rapport ;
  « type » : « explicite » si la description l'écrit, « inférée » si elle se déduit d'un indice concret ;
  « indice » : l'extrait de la description qui justifie CETTE condition, RECOPIÉ MOT POUR MOT (10 à 20 mots).
  Chaque indice est vérifié : un extrait absent du rapport annule la condition. Ne reprends pas la \
cause racine, et ne mets jamais une conséquence (dégât, arrêt, perte d'activité) ici.
- consequence : CE QUE L'INCIDENT A PROVOQUÉ, en une phrase courte : dégât, arrêt, perte \
d'activité, risque. Ne répète pas la cause ici, et n'indique ni durée ni montant.
- composant_brut : le composant touché, avec les mots du rapport (par exemple « variateur de la pompe 2 »), \
vide si le rapport ne le dit pas.
- maintenance_suspectee : même si le rapport ne le dit pas, un défaut de maintenance (inspection, \
nettoyage, graissage, remplacement préventif manquant ou en retard) a-t-il pu laisser l'incident se \
produire ou s'aggraver ? Raisonne à partir de la CONSÉQUENCE et de l'état constaté : dégradation \
progressive qu'une inspection aurait repérée (accumulation, corrosion avancée, usure poussée, \
desserrage, fuite ancienne) -> « plausible » ou « probable » ; événement soudain et externe sur un \
équipement sain, défaut de fabrication ou erreur de travaux -> « non ». « probable » seulement si \
l'état décrit montre une dégradation installée depuis longtemps. Ce champ est une hypothèse : il ne \
change PAS cause_racine ni cause_type.
- justification_maintenance : une phrase qui justifie maintenance_suspectee (vide si « non »).
- action_corrective : l'action concrète qui éviterait que l'incident se reproduise, en une phrase.
- confiance : ta confiance dans la cause racine identifiée (« faible » si l'indice est ténu).

Ne t'appuie que sur le rapport : n'invente pas de cause sans indice. Une cause inventée est pire \
qu'une cause inconnue."""

FIELDS = ["cause_racine", "cause_type", "indice", "causes_contributives", "consequence", "composant_brut",
          "maintenance_suspectee", "justification_maintenance", "action_corrective", "confiance"]


def system_prompt() -> str:
    from framing.cadrage import current

    return SYSTEM_PROMPT.format(domaine=current().domain, cause_inconnue=CAUSE_INCONNUE)


def response_format() -> dict:
    contributive = {
        "type": "object",
        "additionalProperties": False,
        "required": ["texte", "type", "indice"],
        "properties": {
            "texte": {"type": "string"},
            "type": {"type": "string", "enum": CONTRIBUTIVE_TYPES},
            "indice": {"type": "string"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "analyse_incident",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": FIELDS,
                "properties": {
                    "cause_racine": {"type": "string"},
                    "cause_type": {"type": "string", "enum": CAUSE_TYPES},
                    "indice": {"type": "string"},
                    "causes_contributives": {"type": "array", "items": contributive},
                    "consequence": {"type": "string"},
                    "composant_brut": {"type": "string"},
                    "maintenance_suspectee": {"type": "string", "enum": MAINTENANCE_SUSPECTEE},
                    "justification_maintenance": {"type": "string"},
                    "action_corrective": {"type": "string"},
                    "confiance": {"type": "string", "enum": CONFIANCE},
                },
            },
        },
    }
