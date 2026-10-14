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
MAX_TOPICS_PER_CALL = 40             # clusters détaillés par appel de topic_details
MAX_THEMES_PER_CALL = 40             # thèmes créés par appel de define_themes
MAX_THEME_NAME_CHARS = 60
N_EXAMPLES = 5                       # phrases extraites montrées par cluster (courtes)
N_DESCRIPTIONS = 1                   # descriptions complètes montrées par cluster (longues)
EXAMPLE_MAX_CHARS = 300
TOP_PAIRS = 8                        # paires cause -> conséquence montrées dans le rapport

# Mots ignorés dans les mots-clés et retirés des phrases avant clustering : mots vides et mots
# génériques présents partout (« éolienne », « centrale »), qui sinon forment de faux clusters.
EXTRA_STOPWORDS = ["pendant", "pas", "tout", "toute", "tous", "lors", "sans", "aucun", "aucune", "dû", "due",
                   "dus", "liée", "lié", "suite", "cause", "causé", "causée", "provoqué", "provoquée", "entraînant",
                   "éolienne", "éoliennes", "centrale", "centrales", "photovoltaïque", "photovoltaïques", "parc",
                   "site", "installation", "turbine", "machine", "incident", "heure", "heures", "temporaire"]

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
TURNS = {"clustering": 12, "themes": 50, "priorities": 10, "profiles": 12}  # tours maximum par phase

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
nombre de clusters ({N_TOPICS_RANGE[0]} à {N_TOPICS_RANGE[1]} exigé), DBCV (plus haut = mieux séparés).
3. select_clustering avec le run_id d'un essai conforme (« conforme_aux_seuils »: true). Préfère le \
plus détaillé (le plus de clusters, idéalement 8 à 20) parmi les essais conformes dont la qualité \
est au moins « séparation moyenne » : un clustering trop grossier mélange des causes différentes."""

PHASE_THEMES = f"""PHASE : nommer les clusters de {{label}} des incidents {{asset}}.
Objectif : chaque cluster rangé dans un thème.
Voici les clusters, avec leur {{column}} dominant(e) et des phrases typiques :
{{clusters}}

Fusions suggérées par le code (centres très proches ET même {{column}} dominant(e)) :
{{suggestions}}

Appelle define_theme une fois par thème (plusieurs appels dans le même tour si tu veux), avec un nom \
court et précis et la liste des clusters du thème. Un thème peut réunir plusieurs clusters s'ils \
décrivent la même {{singular}} et ont le même {{column}} dominant(e) ; sinon un cluster = un thème.
Nomme le phénomène précis (« foudre », « grêle », « arrêt de sécurité »...), jamais une catégorie \
vague comme « conditions météorologiques ». Un thème de CAUSE ne nomme que la cause (« Vent fort / \
tempête »), jamais sa conséquence (pas « arrêt », « déclenchement », « perte »...).
En cas de doute sur deux clusters, compare_clusters donne leur similarité et un avis. Si deux thèmes \
créés font doublon, merge_themes les fusionne.
La phase se termine d'elle-même quand tous les clusters ont un thème et qu'il ne reste aucun doublon."""

PHASE_PROFILES = """PHASE : associer chaque site à un profil du sujet.
Objectif : les profils attribués avec assign_profiles.
1. site_evidence pour solar, puis pour wind. Pour chaque site, « points_distinctifs » liste les critères \
où il se distingue nettement des 2 autres sites du même type (calculés à partir des données).
2. « compatibilite_profils » indique, pour chaque site et chaque profil, les attentes du profil que \
le site vérifie nettement, et « meilleures_attributions » les attributions qui en respectent le plus. \
Choisis l'une de ces meilleures attributions (si plusieurs, départage avec les points distinctifs et \
les descriptions des profils). Une attribution qui respecte moins d'attentes sera refusée.
3. assign_profiles avec les 6 sites, en expliquant ton raisonnement.
Profils :
{profiles}"""

# --- Heatmap causes x conséquences et chaînes prioritaires (phase priorités) --------------------
HOTSPOTS_MAX = 10                    # cases renvoyées au plus par heatmap_hotspots (top_k)
PRIORITY_CHAINS_RANGE = (3, 5)       # chaînes prioritaires choisies par type d'installation

# --- Criticité Pareto des thèmes de causes (repris de la branche octave) ---------------------------
# Part cumulée du coût (thèmes triés par coût décroissant) -> niveau. Le premier thème est toujours critique.
CRITICALITY_TIERS = [(70.0, "Critique"), (90.0, "Élevée"), (98.0, "Moyenne"), (100.0, "Faible")]

# Hypothèse « manque de maintenance » (extraction, champ maintenance_suspectee) : niveaux comptés comme
# maintenance suspectée. Indicateur SECONDAIRE : montré dans le rapport, absent de PROFILE_EXPECTATIONS.
MAINTENANCE_SUSPECT_LEVELS = ["probable"]
MAINTENANCE_INDICATOR = "maintenance suspectée (hypothèse)"

PHASE_PRIORITIES = f"""PHASE : choisir les chaînes cause -> conséquence prioritaires des incidents {{asset}}.
Objectif : des chaînes choisies avec select_priority_chains.
La heatmap croise les thèmes de causes connues (lignes) et les thèmes de conséquences (colonnes) ; \
chaque case compte les incidents et leur coût total (arrêts de sécurité nominaux exclus).
1. heatmap_hotspots sans site : les cases les plus coûteuses du parc {{asset}}. Tu peux aussi regarder un \
site précis (site = l'un de {{sites}}) pour repérer une chaîne propre à un site.
2. select_priority_chains avec {PRIORITY_CHAINS_RANGE[0]} à {PRIORITY_CHAINS_RANGE[1]} identifiants de chaînes \
(« id » renvoyé par heatmap_hotspots sans site, parmi les {HOTSPOTS_MAX} premières), et une justification \
courte : le coût compte, mais une conséquence grave (incendie, destruction) peut passer devant une \
conséquence fréquente et bénigne. Ne cite que des chiffres renvoyés par l'outil."""

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
- N'évoque pas un phénomène (corrosion, foudre, chaleur...) absent des faits, même s'il figure dans \
la description d'un profil.
- Phrases courtes, claires, en français."""

# Actions trop vagues, refusées par le validateur dans les plans de remédiation.
GENERIC_ACTIONS = [r"renforcer (la |le |les )?(maintenance|surveillance|suivi)", r"améliorer (la |le )?(maintenance|suivi)",
                   r"^\s*maintenance préventive\s*\.?$", r"surveillance renforcée", r"mettre en place un suivi"]

# --- Plans de remédiation : mesures PRÉVENTIVES ou organisationnelles à l'échelle du site --------
ACTION_TYPES = ["programme d'inspection", "nettoyage planifié", "équipement de protection", "procédure",
                "monitoring", "conception / renforcement", "formation"]
# Une mesure qui commence par une réparation ponctuelle est refusée : on veut éviter que la cause se reproduise.
CURATIVE_PATTERN = r"^\s*(remplacer|réparer|changer|redémarrer|réinitialiser|resserrer|nettoyer le|éteindre)\b"
# Mesure de secours (si le LLM échoue) : une mesure préventive type par facteur principal.
PREVENTIVE_CATALOGUE = {
    "corrosion saline / embruns": ("programme d'inspection", "Inspecter tous les 6 mois connecteurs et boîtiers exposés aux embruns et appliquer un traitement anticorrosion", "les embruns corrodent les contacts : un contrôle et une protection réguliers arrêtent la corrosion avant la panne"),
    "foudre / orage": ("équipement de protection", "Auditer la protection foudre (parafoudres, mise à la terre) et contrôler les parafoudres après chaque saison orageuse", "des parafoudres en état dérivent la surtension de foudre avant qu'elle n'atteigne les équipements"),
    "vent fort / tempête": ("programme d'inspection", "Inspecter pales, fixations et structures après chaque tempête et programmer les travaux lourds hors saison des tempêtes", "les dommages de tempête sont repérés et réparés avant qu'ils ne s'aggravent à la tempête suivante"),
    "grêle": ("équipement de protection", "Vérifier la tenue à la grêle des modules exposés et inspecter le champ après chaque épisode de grêle", "des modules adaptés et un contrôle rapide limitent les dégâts de grêle et leurs suites électriques"),
    "chaleur / canicule": ("monitoring", "Suivre en continu la température des onduleurs et modules et ajuster la ventilation avant les vagues de chaleur", "la surchauffe due à la chaleur est détectée et évitée avant qu'elle ne provoque bridage ou défaillance"),
    "poussière / encrassement": ("nettoyage planifié", "Fixer une fréquence de nettoyage des modules indexée sur la perte de production mesurée", "l'encrassement est retiré avant qu'il ne fasse chuter la production ou ne crée des points chauds"),
    "nuisibles / animaux": ("équipement de protection", "Protéger les gaines et entrées de câbles (grillage, gaines anti-rongeurs) et inspecter les zones à risque chaque trimestre", "les animaux n'ont plus accès aux câbles qu'ils rongent"),
    "sol / fondations": ("programme d'inspection", "Réaliser chaque année un relevé géotechnique des fondations (tassements, fissures, affouillements)", "les mouvements du sol sous les fondations sont détectés avant qu'ils n'endommagent la structure"),
    "usure / vieillissement": ("programme d'inspection", "Planifier le remplacement préventif des pièces d'usure selon leur âge et leurs heures de fonctionnement", "l'usure est traitée avant la panne : les pièces vieillissantes sont changées avant de casser"),
    "défaut de fabrication": ("procédure", "Recenser les pièces de la série défectueuse et les remplacer de façon planifiée avec recours auprès du fabricant", "toutes les pièces de la série au défaut de fabrication sont traitées avant de tomber en panne"),
    "défaut ou retard de maintenance": ("procédure", "Mettre à jour le plan de maintenance préventive et suivre le respect des échéances par un indicateur mensuel", "les interventions oubliées ou en retard, à l'origine des incidents, sont rattrapées"),
    "travaux / erreur humaine": ("procédure", "Imposer un permis de travail (dont permis de feu) et une analyse de risques avant toute intervention", "les travaux sont encadrés et les gestes à risque évités"),
    "défaut électrique interne": ("programme d'inspection", "Réaliser une thermographie annuelle des armoires, boîtiers et connexions électriques", "les échauffements sont détectés avant qu'ils ne deviennent des défauts ou des départs de feu"),
}

# --- Arrêts de sécurité nominaux --------------------------------------------------------------------
# Un arrêt de sécurité automatique sans gravité (ex. éolienne arrêtée au-delà de la vitesse de vent
# de coupure) est le FONCTIONNEMENT NORMAL de la protection, pas une panne : ces incidents sont
# présentés à part (perte structurelle liée à l'exposition du site) et exclus des plans de remédiation
# et de la priorisation.
NOMINAL_CONSEQUENCES = ["arrêt de sécurité automatique"]
NOMINAL_SEVERITIES = ["mineure"]
NOMINAL_LEVERS = ("Leviers d'optimisation possibles (pas des réparations) : revoir la stratégie de coupure et de "
                  "redémarrage (seuils, temporisation de redémarrage), anticiper les épisodes de vent par la "
                  "prévision météo, planifier la maintenance pendant les arrêts.")

# Noms de thèmes de CAUSES : ils ne doivent contenir que la cause, pas la conséquence (arrêt, incendie...)
# ni la façon dont l'anomalie a été détectée, ni « indéterminé » (les causes inconnues sont à part).
CONSEQUENCE_WORDS = (r"\b(arrêts?|déclenchements?|pertes?|dégâts?|dommages?|production|sécurité|incendies?|feux?|"
                     r"indéterminée?s?|détections?|détectée?s?)\b")

# Doublons de thèmes : deux clusters (ou thèmes) dont les centres sont très proches ET qui ont le même
# facteur dominant (causes) ou la même catégorie dominante (conséquences) décrivent la même chose.
# La similarité seule ne suffit pas : « maintenance » et « usure » sont à 0.85 sans être la même cause.
# Seuil calibré sur nos données : les deux vrais doublons observés étaient à 0.80 et 0.84.
MERGE_SIMILARITY = 0.8

# Qualité minimale imposée au clustering choisi, s'il existe au moins un essai conforme qui l'atteint.
MIN_CHOSEN_DBCV = 0.5

# Validation manuelle des causes inférées (validation_inferences.py).
VALIDATION_SAMPLE = 20       # incidents à cause inférée tirés par type d'installation
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

# Priorité d'une cause sur un site : sa part du coût total des incidents du site.
PRIORITY_THRESHOLDS = [(25.0, "haute"), (10.0, "moyenne"), (0.0, "basse")]
FLEET_PRIORITIES = 8  # interventions listées dans la priorisation à l'échelle du parc

# Ce que chaque profil laisse attendre (notre lecture des descriptions du sujet). « élevé » : le site
# devrait se distinguer nettement en tête des 3 sites sur ce critère ; « faible » : nettement en queue.
# Sert à 3 choses : compatibilité site/profil (l'agent ne peut pas choisir une attribution qui respecte
# moins d'attentes qu'une autre), écarts au profil dans le rapport, attribution de secours.
PROFILE_EXPECTATIONS = {
    "solar": {
        "A": [("chaleur / canicule", "élevé", "étés chauds"), ("usure / vieillissement", "faible", "site récent"),
              ("lié à la maintenance", "faible", "bonne maintenance"), ("nuisibles / animaux", "faible", "peu de nuisibles")],
        "B": [("nuisibles / animaux", "élevé", "zone agricole, nuisibles modérés")],
        "C": [("embruns / corrosion saline", "élevé", "zone côtière, embruns salés"),
              ("vent fort / tempête", "élevé", "tempêtes fréquentes"), ("poussière / encrassement", "élevé", "poussière agricole"),
              ("usure / vieillissement", "élevé", "site ancien"), ("lié à la maintenance", "élevé", "maintenance faible")],
    },
    "wind": {
        "A": [("vent fort / tempête", "élevé", "très forte exposition aux tempêtes"),
              ("embruns / corrosion saline", "élevé", "façade côtière, embruns"), ("sol / fondations", "élevé", "sol sableux/meuble"),
              ("usure / vieillissement", "faible", "parc récent"), ("lié à la maintenance", "faible", "bonne maintenance")],
        "B": [("usure / vieillissement", "élevé", "parc ancien (~15 ans)"), ("vent fort / tempête", "faible", "vents modérés")],
        "C": [("foudre / orage", "élevé", "forte exposition à la foudre"), ("lié à la maintenance", "élevé", "maintenance faible")],
    },
}
