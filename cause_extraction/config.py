"""Paramètres de l'extraction des causes : catégories, prompt, parallélisme."""

# Appels LLM en parallèle. Peu, pour ménager le quota de débit Azure (requêtes et tokens par minute) :
# au-delà, erreurs 429 et attentes, gérées automatiquement mais plus lentes.
WORKERS = 2

# Facteur à l'origine de l'incident. Liste fermée : on peut compter et comparer les sites.
# Elle reprend les facteurs qui distinguent les profils de site du sujet (environnement, climat,
# âge, maintenance) et les causes techniques courantes.
FACTEURS = [
    "corrosion saline / embruns",
    "foudre / orage",
    "vent fort / tempête",
    "grêle",
    "chaleur / canicule",
    "poussière / encrassement",
    "nuisibles / animaux",
    "sol / fondations",
    "usure / vieillissement",
    "défaut de fabrication",
    "défaut ou retard de maintenance",
    "travaux / erreur humaine",
    "défaut électrique interne",
    "indéterminé",
]

COMPOSANTS = [
    # PV
    "module photovoltaïque", "onduleur", "câblage / connectique", "boîtier de jonction",
    "structure / toiture / bâtiment",
    # Éolien
    "pale", "rotor / moyeu", "système de freinage", "transmission / multiplicateur", "nacelle",
    "tour / fondations",
    # Commun
    "armoire / protection électrique", "système de contrôle / capteurs", "autre",
]

CONFIANCE = ["faible", "moyenne", "haute"]

# Cause écrite par le LLM quand le rapport ne la donne pas (formulation imposée : ces incidents
# se regroupent alors dans un même cluster de causes, facile à repérer).
CAUSE_INCONNUE = "cause non précisée dans le rapport"

SYSTEM_PROMPT = f"""Tu es ingénieur fiabilité pour un exploitant de centrales photovoltaïques et de \
parcs éoliens. On te donne un rapport d'incident. Analyse-le et réponds en JSON.

- composant : le composant touché, parmi la liste imposée.
- cause_racine : POURQUOI c'est arrivé, en une phrase courte, avec les mots du rapport. La cause \
est le plus souvent écrite ou clairement suggérée dans la description : météo (tempête, foudre, \
grêle, chaleur), animaux (rongement), salissure, usure, travaux, défaut d'un composant... Écris-la \
même si elle n'est pas formulée comme « la cause est ». Remonte à la cause première quand le \
rapport la donne (« corrosion des connecteurs due aux embruns » plutôt que « court-circuit »). Ne \
décris pas les dégâts ici.
  Exemples : « une usure anormale du câblage en raison de rongement » -> « rongement du câblage \
par des animaux » ; « salissure constatée sur plusieurs rangées de modules » -> « encrassement des \
modules » ; « surtension pendant un épisode de tempête intense » -> « surtension provoquée par la \
tempête » ; « départ de feu lors d'une opération de travaux » -> « travaux sur le site ».
  Seulement si la description ne donne AUCUNE indication sur l'origine de l'incident (elle ne \
décrit que ce qui a été constaté et fait), écris exactement « {CAUSE_INCONNUE} » et mets \
facteur_principal à « indéterminé ».
- consequence : CE QUE L'INCIDENT A PROVOQUÉ, en une phrase courte : dégât, arrêt, perte de \
production, risque (par exemple « incendie de la toiture et arrêt de la centrale »). Ne répète \
pas la cause ici.
- facteur_principal : le facteur à l'origine de l'incident, parmi la liste imposée.
- facteurs_secondaires : autres facteurs ayant contribué (liste vide si aucun).
- lie_vieillissement : true si l'usure ou l'âge de l'installation a joué un rôle.
- lie_maintenance : true si un défaut, un retard ou une absence de maintenance a joué un rôle.
- action_corrective : l'action qui éviterait que l'incident se reproduise, en une phrase.
- confiance : ta confiance dans la cause identifiée.

Ne t'appuie que sur le rapport : n'invente pas de cause qu'il ne suggère pas.

Facteurs possibles : {", ".join(FACTEURS)}.
Composants possibles : {", ".join(COMPOSANTS)}."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyse_incident",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["composant", "cause_racine", "consequence", "facteur_principal",
                         "facteurs_secondaires", "lie_vieillissement", "lie_maintenance",
                         "action_corrective", "confiance"],
            "properties": {
                "composant": {"type": "string", "enum": COMPOSANTS},
                "cause_racine": {"type": "string"},
                "consequence": {"type": "string"},
                "facteur_principal": {"type": "string", "enum": FACTEURS},
                "facteurs_secondaires": {"type": "array", "items": {"type": "string", "enum": FACTEURS}},
                "lie_vieillissement": {"type": "boolean"},
                "lie_maintenance": {"type": "boolean"},
                "action_corrective": {"type": "string"},
                "confiance": {"type": "string", "enum": CONFIANCE},
            },
        },
    },
}
