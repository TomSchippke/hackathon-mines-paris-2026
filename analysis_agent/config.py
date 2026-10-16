"""Paramètres de l'agent d'analyse : bornes des outils, seuils de qualité, prompts.

Rien ici ne dépend du problème traité : domaine, libellés des segments et ordre des sévérités viennent
du cadrage (framing/, à partir de context/contexte.md). Les causes et conséquences sont découvertes
dans les données, sans liste fermée.
"""

from framing.config import ACTION_TYPES, UNDETERMINED  # noqa: F401 (réexportés pour le reste du paquet)

# Deux clusterings par segment, sur les phrases extraites par le LLM (extract_causes) :
#   cause       -> pourquoi l'incident est arrivé
#   consequence -> ce qu'il a provoqué
# Un même phénomène vu comme cause et comme conséquence n'est donc pas mélangé dans un cluster.
DIMENSIONS = {"cause": "cause_racine", "consequence": "consequence"}  # dimension -> colonne extraite
DIMENSION_LABELS = {"cause": "causes", "consequence": "conséquences"}

# --- Bornes des outils (un appel hors bornes est refusé, avec la raison) ---------------------
MIN_CLUSTER_SIZE_RANGE = (5, 50)     # taille minimale d'un cluster acceptée par run_clustering
MAX_CLUSTERING_RUNS = 5              # essais de clustering par segment et par dimension
MAX_TOPICS_PER_CALL = 40             # clusters détaillés par appel de topic_details
MAX_THEMES_PER_CALL = 40             # thèmes créés par appel de define_themes
MAX_THEME_NAME_CHARS = 60
N_EXAMPLES = 5                       # phrases extraites montrées par cluster (courtes)
N_DESCRIPTIONS = 1                   # descriptions complètes montrées par cluster (longues)
EXAMPLE_MAX_CHARS = 300
TOP_PAIRS = 8                        # paires cause -> conséquence montrées dans le rapport

# Mots ignorés dans les mots-clés et retirés des phrases avant clustering : mots vides, et mots
# génériques présents partout dans un segment (nom de l'installation...), qui sinon forment de faux
# clusters. Ces derniers sont trouvés dans les données : un mot présent dans au moins GENERIC_MIN_DF
# des descriptions d'un segment ne distingue aucune cause.
EXTRA_STOPWORDS = ["pendant", "pas", "tout", "toute", "tous", "lors", "sans", "aucun", "aucune", "dû", "due",
                   "dus", "liée", "lié", "suite", "cause", "causé", "causée", "provoqué", "provoquée", "entraînant",
                   "site", "installation", "machine", "incident", "heure", "heures", "temporaire"]
GENERIC_MIN_DF = 0.4

# Découverte et affectation (D3, D14)
ASSIGN_BATCH = 20                  # phrases par appel LLM d'affectation
NONE_LABEL = "aucun"               # réponse « aucun thème ne convient »
NONE_MAX_PCT = 5.0                 # part maximale de phrases « aucun » par segment (D14)
MAX_RECLUSTER_ROUNDS = 2
RECLUSTER_MIN_CLUSTER_SIZE = 5     # taille minimale d'un cluster lors d'une reprise
# Composants (D8)
MAX_COMPONENT_GROUPS = 15
# Arrêts nominaux (D5) : part minimale des incidents d'un thème dans la sévérité la plus faible
NOMINAL_MIN_MILDEST_PCT = 80.0
# Plan d'actions (D10-D12)
PLAN_COVERAGE_PCT = 80.0
PLAN_MAX_MEASURES = 12
PLAN_MIN_INCIDENTS = 5
SCOPE_SITE, SCOPE_SEGMENT = "spécifique au site", "commun au segment"
TO_DEFINE = "À définir"

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

# --- Phases du workflow : le code fixe l'ordre, le LLM décide dans chaque phase ---------------
# Une phase = une petite session d'agent, avec 2 ou 3 outils et un seul objectif. Un petit modèle
# suit bien une consigne courte ; il se perd dans un plan de 40 étapes.
TURNS = {"clustering": 12, "themes": 50, "priorities": 10, "nominal": 8}  # tours maximum par phase

ROLE = """Tu es analyste fiabilité. Contexte : {domaine} Tu travailles avec des outils : ils font \
les calculs, toi tu prends les décisions et tu les justifies.

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
nombre de clusters ({N_TOPICS_RANGE[0]} à {N_TOPICS_RANGE[1]} exigé), DBCV (plus haut = mieux séparés).
3. select_clustering avec le run_id d'un essai conforme (« conforme_aux_seuils »: true). Préfère le \
plus détaillé (le plus de clusters, idéalement 8 à 20) parmi les essais conformes dont la qualité \
est au moins « séparation moyenne » : un clustering trop grossier mélange des causes différentes."""

PHASE_THEMES = f"""PHASE : nommer les clusters de {{label}} des incidents {{asset}}.
Objectif : chaque cluster rangé dans un thème.
Voici les clusters, avec leurs mots-clés et des phrases typiques :
{{clusters}}

Fusions suggérées par le code (centres très proches) :
{{suggestions}}

Appelle define_theme une fois par thème (plusieurs appels dans le même tour si tu veux), avec un nom \
court et précis et la liste des clusters du thème. Un thème peut réunir plusieurs clusters s'ils \
décrivent la même {{singular}} (même phénomène dans les phrases typiques) ; sinon un cluster = un thème.
Nomme le phénomène précis, jamais une catégorie vague comme « conditions extérieures » ou « problème \
technique ». Un thème de CAUSE ne nomme que la cause, jamais sa conséquence (pas « arrêt », \
« déclenchement », « perte »...).
En cas de doute sur deux clusters, compare_clusters donne leur similarité et un avis. Si deux thèmes \
créés font doublon, merge_themes les fusionne.
La phase se termine d'elle-même quand tous les clusters ont un thème et qu'il ne reste aucun doublon."""

# --- Heatmap causes x conséquences et chaînes prioritaires (phase priorités) --------------------
HOTSPOTS_MAX = 10                    # cases renvoyées au plus par heatmap_hotspots (top_k)
PRIORITY_CHAINS_RANGE = (3, 5)       # chaînes prioritaires choisies par segment

# --- Criticité Pareto des thèmes de causes (repris de la branche octave) ---------------------------
# Part cumulée du coût (thèmes triés par coût décroissant) -> niveau. Le premier thème est toujours critique.
CRITICALITY_TIERS = [(70.0, "Critique"), (90.0, "Élevée"), (98.0, "Moyenne"), (100.0, "Faible")]

# Hypothèse « manque de maintenance » (extraction, champ maintenance_suspectee) : niveaux comptés comme
# maintenance suspectée. Indicateur SECONDAIRE : montré dans le rapport, jamais une cause.
MAINTENANCE_SUSPECT_LEVELS = ["probable"]
MAINTENANCE_INDICATOR = "maintenance suspectée (hypothèse)"

PHASE_PRIORITIES = f"""PHASE : choisir les chaînes cause -> conséquence prioritaires des incidents {{asset}}.
Objectif : des chaînes choisies avec select_priority_chains.
La heatmap croise les thèmes de causes connues (lignes) et les thèmes de conséquences (colonnes) ; \
chaque case compte les incidents et leur coût total (arrêts de sécurité nominaux exclus).
1. heatmap_hotspots sans site : les cases les plus coûteuses des incidents {{asset}}. Tu peux aussi regarder un \
site précis (site = l'un de {{sites}}) pour repérer une chaîne propre à un site.
2. select_priority_chains avec {PRIORITY_CHAINS_RANGE[0]} à {PRIORITY_CHAINS_RANGE[1]} identifiants de chaînes \
(« id » renvoyé par heatmap_hotspots sans site, parmi les {HOTSPOTS_MAX} premières), et une justification \
courte : le coût compte, mais une conséquence grave (destruction, risque pour les personnes) peut passer devant une \
conséquence fréquente et bénigne. Ne cite que des chiffres renvoyés par l'outil."""

PHASE_NOMINAL = f"""PHASE : choisir les thèmes de conséquences qui sont des arrêts nominaux des incidents {{asset}}.
Objectif : la liste des thèmes nominaux, choisie avec flag_nominal (une liste vide est permise).
Un arrêt nominal est le fonctionnement normal d'une protection : un arrêt de sécurité automatique, sans \
dommage ni risque. Ce n'est pas une panne.
1. nominal_evidence sur les incidents {{asset}} : pour chaque thème de conséquence, son nombre d'incidents, \
la part à la sévérité la plus faible, le coût moyen et des exemples.
2. flag_nominal avec les identifiants des thèmes nominaux et une justification courte. Un thème n'est accepté \
que si au moins {NOMINAL_MIN_MILDEST_PCT:g} % de ses incidents sont à la sévérité la plus faible.
3. Si aucun thème n'est un fonctionnement normal de protection, appelle flag_nominal avec une liste vide.
Ne choisis que des thèmes de conséquences. Ne cite que des chiffres renvoyés par l'outil."""

# --- Juge (judge.py) : relit chaque texte avec une grille et propose une correction ---------------
JUDGE_ROLE = """Tu es un relecteur exigeant d'un rapport de maintenance. On te donne UN texte du rapport, les \
FAITS sur lesquels il doit reposer et une GRILLE de questions. Réponds en JSON :
- problemes : la liste des problèmes trouvés (vide si aucun) ;
- correction : le texte corrigé (le texte d'origine s'il n'y a rien à corriger).
Règles pour la correction : n'utilise que les faits fournis (aucun chiffre, aucune cause, aucun lieu, \
aucun âge qui n'y figure pas) ; ne parle jamais de « rang » ; garde le même rôle et la même longueur."""

# --- Rédaction contrainte (writing.py) : le LLM reformule des faits calculés en Python ----------
WRITE_RETRIES = 2  # nouvelles tentatives si le validateur trouve une erreur ; ensuite texte Python

WRITER_ROLE = """Tu rédiges un rapport d'analyse d'incidents pour une équipe de maintenance. On te \
donne des FAITS calculés à partir des données. Règles strictes :
- N'utilise QUE ces faits. Aucun chiffre, aucune cause, aucun critère qui n'y figure pas.
- Ne parle jamais de « rang ». Reprends les chiffres tels quels (tu peux arrondir à l'unité).
- N'évoque pas un phénomène (une cause, une condition, un lieu...) absent des faits, même s'il figure dans \
le contexte.
- Phrases courtes, claires, en français."""

# Actions trop vagues, refusées par le validateur dans les plans de remédiation.
GENERIC_ACTIONS = [r"renforcer (la |le |les )?(maintenance|surveillance|suivi)", r"améliorer (la |le )?(maintenance|suivi)",
                   r"^\s*maintenance préventive\s*\.?$", r"surveillance renforcée", r"mettre en place un suivi"]

# --- Plans de remédiation : mesures PRÉVENTIVES ou organisationnelles à l'échelle du site --------
# (types de mesures : framing.config.ACTION_TYPES ; le contenu de chaque mesure vient des données)
# Une mesure qui commence par une réparation ponctuelle est refusée : on veut éviter que la cause se reproduise.
CURATIVE_PATTERN = r"^\s*(remplacer|réparer|changer|redémarrer|réinitialiser|resserrer|nettoyer le|éteindre)\b"

# --- Arrêts de sécurité nominaux --------------------------------------------------------------------
# Un arrêt de sécurité automatique sans gravité (thème de conséquence choisi comme nominal par l'agent,
# sévérité la plus faible) est le FONCTIONNEMENT NORMAL de la protection, pas une panne : ces incidents
# sont présentés à part (perte structurelle liée à l'exposition du site) et exclus des plans de
# remédiation et de la priorisation.
NOMINAL_LEVERS = ("Leviers d'optimisation possibles (pas des réparations) : revoir les seuils de déclenchement et la "
                  "stratégie de redémarrage (temporisation), anticiper les épisodes à risque par la prévision des "
                  "conditions, planifier la maintenance pendant les arrêts.")

# Noms de thèmes de CAUSES : ils ne doivent contenir que la cause, pas la conséquence (arrêt, incendie...)
# ni la façon dont l'anomalie a été détectée, ni « indéterminé » (les causes inconnues sont à part).
CONSEQUENCE_WORDS = (r"\b(arrêts?|déclenchements?|pertes?|dégâts?|dommages?|production|sécurité|incendies?|feux?|"
                     r"indéterminée?s?|détections?|détectée?s?)\b")

# Doublons de thèmes : deux clusters (ou thèmes) dont les centres sont très proches décrivent la même chose.
MERGE_SIMILARITY = 0.8

# Qualité minimale imposée au clustering choisi, s'il existe au moins un essai conforme qui l'atteint.
MIN_CHOSEN_DBCV = 0.5

# Validation manuelle des causes inférées (validation_inferences.py).
VALIDATION_SAMPLE = 20       # incidents à cause inférée tirés par segment
VALIDATION_SEED = 42

# --- Faits calculés (facts.py) -------------------------------------------------------------------
# Un site « se distingue » sur un critère s'il est seul en tête (ou seul en queue) avec un écart net :
# au moins DISTINCT_MIN_GAP points, et soit DISTINCT_RATIO fois la valeur du suivant, soit au moins
# DISTINCT_LARGE_GAP points d'écart (pour les grandes valeurs : 60 % contre 47 % est un écart net).
DISTINCT_MIN_GAP = 3.0
DISTINCT_RATIO = 1.5
DISTINCT_LARGE_GAP = 10.0

# Qualité d'un clustering en mots, d'après des seuils fixes (pas d'adjectif libre du LLM).
DBCV_LABELS = [(0.7, "bonne séparation"), (0.5, "séparation moyenne"), (0.0, "séparation faible")]
STABILITY_LABELS = [(0.9, "très stable"), (0.8, "stable"), (0.0, "instable")]
