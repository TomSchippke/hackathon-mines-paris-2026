"""Paramètres de l'agent d'analyse : bornes des outils, seuils de qualité, profils de site, prompt."""

ASSET_TYPES = ["solar", "wind"]
ASSET_LABELS = {"solar": "photovoltaïque", "wind": "éolien"}

# Deux clusterings par type d'installation, sur les phrases extraites par le LLM (extract_causes) :
#   cause       -> pourquoi l'incident est arrivé (« corrosion des connecteurs due aux embruns »)
#   consequence -> ce qu'il a provoqué (« incendie de la toiture et arrêt de la centrale »)
# Un cluster « feu » ne mélange donc plus le feu comme cause et le feu comme conséquence.
DIMENSIONS = {"cause": "cause_racine", "consequence": "consequence"}  # dimension -> colonne extraite
DIMENSION_LABELS = {"cause": "causes", "consequence": "conséquences"}

# --- Bornes des outils (un appel hors bornes est refusé, avec la raison) ---------------------
MIN_CLUSTER_SIZE_RANGE = (5, 50)     # taille minimale d'un cluster acceptée par run_clustering
MAX_CLUSTERING_RUNS = 5              # essais de clustering par type d'installation et par dimension
MAX_TOPICS_PER_CALL = 10             # clusters détaillés par appel de topic_details
MAX_THEMES_PER_CALL = 20             # thèmes créés par appel de define_themes
MAX_THEME_NAME_CHARS = 60
ACTIONS_PER_SITE = (2, 8)            # nombre d'actions du plan de remédiation par site
N_EXAMPLES = 5                       # phrases extraites montrées par cluster (courtes)
N_DESCRIPTIONS = 1                   # descriptions complètes montrées par cluster (longues)
EXAMPLE_MAX_CHARS = 300
TOP_PAIRS = 10                       # paires cause -> conséquence renvoyées par liens_cause_consequence

# Critères des profils de site (sujet, p. 40), mesurés sur chaque site à partir des facteurs extraits
# (facteur principal ou secondaire). Ce sont eux qui distinguent les profils A, B, C : ils ne doivent
# pas être fusionnés dans un même thème de causes.
PROFILE_CRITERIA = {
    "chaleur / canicule": ["chaleur / canicule"],
    "vent fort / tempête": ["vent fort / tempête"],
    "foudre / orage": ["foudre / orage"],
    "grêle": ["grêle"],
    "embruns / corrosion saline": ["corrosion saline / embruns"],
    "poussière / encrassement": ["poussière / encrassement"],
    "nuisibles / animaux": ["nuisibles / animaux"],
    "sol / fondations": ["sol / fondations"],
    "défaut de fabrication": ["défaut de fabrication"],
}
# Facteurs considérés comme une même cause pour la règle de fusion (indéterminé : compatible avec tout).
NEUTRAL_FACTORS = {"indéterminé"}

# --- Seuils de qualité d'un clustering (vérifiés par le code, pas par l'agent) -----------------
# Stabilité : accord (ARI) entre les clusters obtenus avec 3 graines aléatoires d'UMAP différentes.
# 1 = identiques. En dessous de 0.8, les clusters dépendent du hasard : on ne peut pas s'y fier.
MIN_STABILITY = 0.8
MAX_UNCLASSIFIED_PCT = 15            # % d'incidents non classés (bruit HDBSCAN)
N_TOPICS_RANGE = (3, 40)
STABILITY_SEEDS = (42, 1, 2)

# Fusion de clusters dans un thème : refusée si les incidents réunis sont nettement moins semblables
# entre eux que dans les clusters d'origine (cohésion = similarité cosinus moyenne au centre).
MERGE_MIN_COHESION_RATIO = 0.9

# --- Profils de site donnés par le sujet (Kick-off, p. 40) ------------------------------------
SITE_PROFILES = {
    "solar": {
        "A": "récent (~3 ans) ; zone continentale sèche, peu de nuisibles ; étés chauds ; bonne maintenance",
        "B": "âge moyen (~6 ans) ; zone agricole mixte, nuisibles modérés ; tempéré, orages d'été modérés ; "
             "maintenance moyenne",
        "C": "ancien (~9 ans) ; rural/côtier : embruns salés et poussière agricole accumulée, nuisibles "
             "fréquents ; océanique, tempêtes fréquentes ; maintenance faible",
    },
    "wind": {
        "A": "récent (~4 ans) ; façade côtière, embruns, sol sableux/meuble ; très forte exposition aux "
             "tempêtes (vent océanique dominant) ; bonne maintenance",
        "B": "ancien (~15 ans, un des premiers parcs français) ; plaine agricole intérieure, sol stable ; "
             "vents modérés ; maintenance moyenne",
        "C": "intermédiaire (~7 ans) ; terrain en altitude, accès difficile, orages fréquents ; forte "
             "exposition à la foudre et aux rafales ; maintenance faible",
    },
}

# --- Phases du workflow : le code fixe l'ordre, le LLM décide dans chaque phase ---------------
# Une phase = une petite session d'agent, avec 2 ou 3 outils et un seul objectif. Un petit modèle
# suit bien une consigne courte ; il se perd dans un plan de 40 étapes.
TURNS = {"clustering": 12, "themes": 50, "report": 20}  # tours maximum par phase

ROLE = """Tu es analyste fiabilité pour un exploitant de centrales photovoltaïques et de parcs \
éoliens. Tu travailles avec des outils : ils font les calculs, toi tu prends les décisions et tu \
les justifies.

Règles :
- N'utilise que des chiffres renvoyés par les outils. N'invente rien, ni chiffre ni identifiant.
- Si un outil refuse ton appel, lis la raison et change d'arguments : ne répète jamais un appel refusé.
- Appelle les outils (plusieurs par tour si utile) jusqu'à atteindre l'objectif de la phase.
- Écris en français, de façon claire pour une équipe de maintenance."""

PHASE_CLUSTERING = f"""PHASE : choisir le clustering des {{label}} des incidents {{asset}}.
Objectif : un clustering choisi avec select_clustering.
1. run_clustering avec plusieurs valeurs de min_cluster_size (entier entre {MIN_CLUSTER_SIZE_RANGE[0]} \
et {MIN_CLUSTER_SIZE_RANGE[1]}, {MAX_CLUSTERING_RUNS} essais maximum). Commence par 10, puis ajuste : \
trop de clusters ou stabilité faible -> augmenter ; trop de non classés ou trop peu de clusters -> baisser.
2. Compare : stabilité (>= {MIN_STABILITY} exigé), non classés (<= {MAX_UNCLASSIFIED_PCT} % exigé), \
nombre de clusters ({N_TOPICS_RANGE[0]} à {N_TOPICS_RANGE[1]} exigé, idéalement 8 à 25 pour rester lisible), DBCV (plus haut = mieux séparés).
3. select_clustering avec le run_id d'un essai conforme (« conforme_aux_seuils »: true), et justifie par les scores."""

PHASE_THEMES = f"""PHASE : nommer les clusters de {{label}} des incidents {{asset}}.
Objectif : chaque cluster rangé dans un thème.
Clusters à ranger : {{clusters}}
1. topic_details (jusqu'à {MAX_TOPICS_PER_CALL} clusters par appel) pour lire leurs phrases typiques.
2. define_themes (jusqu'à {MAX_THEMES_PER_CALL} thèmes par appel) : un nom court et clair par thème. \
Un thème peut réunir plusieurs clusters s'ils décrivent la même {{singular}} ; sinon un cluster = un thème.
Ne réunis JAMAIS des facteurs différents dans un même thème, même s'ils se ressemblent : foudre, \
grêle, tempête, chaleur, embruns, poussière, nuisibles, sol, usure, défaut de fabrication sont des \
thèmes distincts (ce sont eux qui distinguent les sites). Un nom général comme « conditions \
météorologiques » est interdit : nomme le phénomène précis (« foudre », « grêle »...).
Continue jusqu'à ce que « restants_sans_theme » soit vide."""

PHASE_REPORT = f"""PHASE : rédiger le rapport d'analyse.
Objectif : le rapport enregistré avec write_report.
1. liens_cause_consequence pour solar et pour wind : quelles causes mènent à quelles conséquences.
2. site_evidence pour solar et pour wind : statistiques par site.
3. write_report :
   - associe chaque site à un profil (A, B ou C, chaque profil une seule fois par type d'installation) \
avec le tableau « indicateurs_profils » de site_evidence : pour chaque critère des profils (chaleur, \
tempête, foudre, embruns, nuisibles, poussière, sol, usure, maintenance, coût), il donne la valeur de \
chaque site et son rang parmi les 3 sites (1 = le plus élevé). Attribue à chaque profil le site dont \
les rangs correspondent le mieux à sa description, et cite ces rangs dans justification_profil ;
   - pour chaque site, {ACTIONS_PER_SITE[0]} à {ACTIONS_PER_SITE[1]} actions qui agissent sur ses causes dominantes ;
   - dans « limites », chiffre la part des rapports qui ne donnent pas la cause.
Profils de site :
{{profiles}}"""
