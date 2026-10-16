"""Paramètres du cadrage : bornes des listes, taille de l'échantillon montré au LLM, prompt."""

# Colonne qui sépare les incidents en groupes analysés à part (clustering, profils, heatmaps) :
# des installations de natures différentes ont des vocabulaires et des causes différentes.
SEGMENT_COLUMN = "asset_type"

# Échantillon de descriptions montré au LLM par segment (graine fixe : cadrage reproductible).
SAMPLE_PER_SEGMENT = 25
SAMPLE_SEED = 0
SAMPLE_MAX_CHARS = 280
# Colonnes catégorielles dont les valeurs sont montrées au LLM (au plus ce nombre de valeurs distinctes).
MAX_CATEGORY_VALUES = 12

# Bornes des listes fermées (en dehors : le cadrage est refusé et redemandé au LLM).
FACTORS_RANGE = (6, 16)
COMPONENTS_RANGE = (4, 16)
CONSEQUENCES_RANGE = (4, 9)
MAX_KEYWORDS = 6
RETRIES = 2

# Valeurs ajoutées par le code (jamais proposées par le LLM).
UNDETERMINED = "indéterminé"   # facteur d'une cause inconnue
OTHER = "autre"
# Indicateurs calculés à partir des drapeaux de l'extraction (lie_vieillissement, lie_maintenance) :
# critères possibles d'un profil, au même titre que les facteurs.
AGE_CRITERION = "lié à l'usure / l'âge"
MAINTENANCE_CRITERION = "lié à la maintenance"
RESERVED = {UNDETERMINED, OTHER, AGE_CRITERION, MAINTENANCE_CRITERION}

# Types de mesures préventives (communs à tout le pipeline).
ACTION_TYPES = ["programme d'inspection", "nettoyage planifié", "équipement de protection", "procédure",
                "monitoring", "conception / renforcement", "formation"]
DIRECTIONS = ["élevé", "faible"]

ROLE = """Tu prépares l'analyse d'une base de rapports d'incidents pour une entreprise cliente. Tu ne \
sais du problème que ce que disent le CONTEXTE écrit par l'utilisateur et l'ÉCHANTILLON des données. \
N'ajoute aucune connaissance extérieure sur l'entreprise ou ses sites : tout doit venir de ces deux sources.
Ta tâche : définir le cadre d'analyse, c'est-à-dire les listes fermées qui serviront à classer chaque \
incident, de façon comparable d'un site à l'autre. Écris en français."""

TASK = f"""Réponds en JSON :
- domaine : une phrase qui dit qui est le client et ce qu'exploitent ses sites (d'après le contexte).
- segments : pour chaque valeur de la colonne « {{segment_column}} » listée plus bas, un libellé court en \
français (ce que désigne cette valeur, d'après le contexte et les descriptions).
- facteurs : de {FACTORS_RANGE[0]} à {FACTORS_RANGE[1]} facteurs à l'ORIGINE des incidents (POURQUOI ils \
arrivent : conditions extérieures, usure, défauts techniques, erreurs humaines, organisation...), qui \
couvrent les descriptions de l'échantillon. Noms courts (« a / b » permis), sans recouvrement, jamais une \
conséquence (pas « arrêt », « perte », « dégât »). Pour chacun :
    mots_cles : 2 à {MAX_KEYWORDS} mots (ou débuts de mots) des descriptions qui le reconnaissent ;
    type, mesure, mecanisme : une mesure PRÉVENTIVE type contre ce facteur (quoi, sur quels équipements, à \
quelle fréquence) et pourquoi elle le réduit. Pas une réparation ponctuelle.
- composants : de {COMPONENTS_RANGE[0]} à {COMPONENTS_RANGE[1]} équipements ou parties touchés, d'après \
les descriptions.
- categories_consequence : de {CONSEQUENCES_RANGE[0]} à {CONSEQUENCES_RANGE[1]} catégories de ce que les \
incidents PROVOQUENT (perte d'activité, dégât mineur, dégât majeur, risque pour les personnes...). \
fonctionnement_normal = true seulement pour une protection qui a fonctionné comme prévu sans dommage \
(par exemple un arrêt de sécurité automatique) : ce n'est pas une panne.
- severites : les valeurs de sévérité listées plus bas, de la MOINS grave à la PLUS grave (toutes, une fois).
- profils : SEULEMENT si le contexte décrit des profils ou des types de sites à reconnaître ; sinon une liste \
vide. N'invente jamais de profil. Pour un segment, autant de profils que de sites. Pour chaque profil : un \
identifiant court, sa description (mots du contexte) et ses attentes : des critères (un de tes facteurs, ou \
« {AGE_CRITERION} », ou « {MAINTENANCE_CRITERION} ») avec le sens « élevé » (le site devrait se distinguer \
en tête des sites du segment) ou « faible » (en queue), et la raison tirée du contexte."""

RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["domaine", "segments", "facteurs", "composants", "categories_consequence", "severites", "profils"],
    "properties": {
        "domaine": {"type": "string"},
        "segments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["valeur", "libelle"],
            "properties": {"valeur": {"type": "string"}, "libelle": {"type": "string"}}}},
        "facteurs": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["nom", "mots_cles", "type", "mesure", "mecanisme"],
            "properties": {"nom": {"type": "string"}, "mots_cles": {"type": "array", "items": {"type": "string"}},
                           "type": {"type": "string", "enum": ACTION_TYPES},
                           "mesure": {"type": "string"}, "mecanisme": {"type": "string"}}}},
        "composants": {"type": "array", "items": {"type": "string"}},
        "categories_consequence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["nom", "fonctionnement_normal"],
            "properties": {"nom": {"type": "string"}, "fonctionnement_normal": {"type": "boolean"}}}},
        "severites": {"type": "array", "items": {"type": "string"}},
        "profils": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["segment", "id", "description", "attentes"],
            "properties": {"segment": {"type": "string"}, "id": {"type": "string"}, "description": {"type": "string"},
                           "attentes": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False,
                               "required": ["critere", "sens", "raison"],
                               "properties": {"critere": {"type": "string"},
                                              "sens": {"type": "string", "enum": DIRECTIONS},
                                              "raison": {"type": "string"}}}}}}},
    },
}
