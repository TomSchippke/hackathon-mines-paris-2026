"""Paramètres de l'extraction des causes : catégories, prompt, parallélisme."""

import re

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

# Ce que l'incident a provoqué, en catégories fermées (normalisées) : comparables d'un site à l'autre.
CATEGORIES_CONSEQUENCE = [
    "perte de production",
    "arrêt de sécurité automatique",
    "déclenchement de protection / surtension",
    "dégât matériel mineur",
    "dégât matériel majeur / destruction",
    "incendie",
    "aucun impact notable",
    "autre",
]

# Comment la cause a été obtenue : écrite dans le rapport, déduite d'indices du rapport, ou inconnue.
CAUSE_TYPES = ["explicite", "inférée", "inconnue"]

CONFIANCE = ["faible", "moyenne", "haute"]

# Valeur unique pour une cause inconnue. Toutes les variantes (« non précisée », « non déterminée »...)
# y sont ramenées, et ces incidents sont comptés à part (exclus du clustering des causes).
CAUSE_INCONNUE = "cause non précisée dans le rapport"

# Un indice (extrait cité par le LLM) est accepté s'il figure dans la description à QUOTE_MIN_MATCH près
# (la plus longue portion commune doit couvrir cette part de l'indice) ; sinon la cause est annulée.
QUOTE_MIN_MATCH = 0.8
UNKNOWN_CAUSE_PATTERN = re.compile(r"non (précis|determin|détermin|identifi|connu|renseign|indiqu)|inconnu|"
                                   r"pas (précis|indiqu|d'information)|aucune (cause|information|indication)",
                                   re.IGNORECASE)

SYSTEM_PROMPT = f"""Tu es ingénieur fiabilité pour un exploitant de centrales photovoltaïques et de \
parcs éoliens. On te donne un rapport d'incident. Analyse-le et réponds en JSON.

- composant : le composant touché, parmi la liste imposée.
- cause_racine : POURQUOI c'est arrivé, en une phrase courte, avec les mots du rapport. Remonte à la \
cause première (« corrosion des connecteurs due aux embruns » plutôt que « court-circuit »). Ne \
décris pas les dégâts ici.
  Exemples : « une usure anormale du câblage en raison de rongement » -> « rongement du câblage \
par des animaux » ; « salissure constatée sur plusieurs rangées de modules » -> « encrassement des \
modules » ; « surtension pendant un épisode de tempête intense » -> « surtension provoquée par la \
tempête » ; « départ de feu lors d'une opération de travaux » -> « travaux sur le site ».
- cause_type :
  « explicite » si la DESCRIPTION écrit la cause ;
  « inférée » seulement si la description ne l'écrit pas mais contient un indice concret qui la \
désigne (par exemple « après un épisode de vent extrême », « usure visible », « lors de travaux »). \
La météo du jour seule ne suffit pas : l'indice doit être dans la description et lié à l'incident ;
  « inconnue » dans tous les autres cas. Écris alors exactement « {CAUSE_INCONNUE} » dans \
cause_racine, mets facteur_principal à « indéterminé » et laisse indice vide. En cas de doute, \
choisis « inconnue » : une cause inventée est pire qu'une cause inconnue.
- indice : pour une cause explicite ou inférée, l'extrait de la description qui la justifie, \
RECOPIÉ MOT POUR MOT (10 à 20 mots). Il sera vérifié : un extrait absent du rapport annule la cause.
- consequence : CE QUE L'INCIDENT A PROVOQUÉ, en une phrase courte : dégât, arrêt, perte de \
production, risque (par exemple « incendie de la toiture et arrêt de la centrale »). Ne répète \
pas la cause ici, et n'indique ni durée ni montant.
- categorie_consequence : la catégorie de cette conséquence, parmi la liste imposée.
- facteur_principal : le facteur à l'origine de l'incident, parmi la liste imposée.
- facteurs_secondaires : autres facteurs ayant contribué (liste vide si aucun).
- lie_vieillissement : true si l'usure ou l'âge de l'installation a joué un rôle.
- lie_maintenance : true seulement si le rapport indique un défaut, un retard ou une absence de \
maintenance (une simple intervention de réparation ne suffit pas).
- action_corrective : l'action concrète qui éviterait que l'incident se reproduise, en une phrase.
- confiance : ta confiance dans la cause identifiée (« faible » si l'indice est ténu).

Ne t'appuie que sur le rapport : n'invente pas de cause sans indice.

Facteurs possibles : {", ".join(FACTEURS)}.
Composants possibles : {", ".join(COMPOSANTS)}.
Catégories de conséquence : {", ".join(CATEGORIES_CONSEQUENCE)}."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyse_incident",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["composant", "cause_racine", "cause_type", "indice", "consequence", "categorie_consequence",
                         "facteur_principal", "facteurs_secondaires", "lie_vieillissement", "lie_maintenance",
                         "action_corrective", "confiance"],
            "properties": {
                "composant": {"type": "string", "enum": COMPOSANTS},
                "cause_racine": {"type": "string"},
                "cause_type": {"type": "string", "enum": CAUSE_TYPES},
                "indice": {"type": "string"},
                "consequence": {"type": "string"},
                "categorie_consequence": {"type": "string", "enum": CATEGORIES_CONSEQUENCE},
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
