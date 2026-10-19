"""Configuration pour le pipeline agentique de segmentation des incidents."""

from __future__ import annotations

from pathlib import Path
# Noms de colonnes du schéma d'entrée (contrat fixe : voir REQUIRED_COLUMNS dans pipeline.py)
# Chemins d'entrée et de sortie par défaut
DEFAULT_INPUT = Path("outputs/incidents_full.csv")
FALLBACK_INPUT = Path("data/incidents_full.csv")
DEFAULT_OUT_DIR = Path("outputs/segmentation")
CACHE_DIR = Path("outputs/cache/agent")
EMBEDDINGS_CACHE_DIR = Path("outputs/cache")

# Noms de colonnes standards attendus (avec auto-détection intelligente en secours)
TEXT_COL = "text"
COST_COL = "estimated_cost_eur"
DOWNTIME_COL = "downtime_hours"
DATE_COL = "date"
ID_COL = "incident_id"

# Paramètres de clustering (UMAP & HDBSCAN)
MIN_CLUSTER_SIZE = 10
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
NGRAM_RANGE = (1, 2)

# Nombre d'incidents représentatifs transmis à l'agent par cluster
NUM_REPRESENTATIVE_DOCS = 4

# Taille des lots pour l'agent de reclassement des outliers
OUTLIER_BATCH_SIZE = 15

# Stopwords français
FRENCH_STOPWORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "sur", "au", "aux", "pour",
    "par", "est", "sont", "a", "été", "suite", "lors", "dans", "que", "qui", "ce", "cette", "ces",
    "afin", "plus", "après", "avant", "ou", "d", "l", "à", "avec", "sans", "son", "sa", "ses",
    "il", "elle", "ont", "être", "fait", "ainsi", "puis", "selon", "entre", "sous",
]

# --- Clustering v2 (voir docs/implementation_plan_clustering_v2.md) ---

# Littéraux propres au domaine : seul endroit où ils sont définis
LANGUAGE = "fr"
SEVERITY_ORDER = ["mineure", "modérée", "sévère"]  # rang 0 < 1 < 2
PARTITION_COL = "asset_type"
EQUIPMENT_COL = "equipment_id"
SEVERITY_COL = "severity"
OUTLIER_LABEL = "isole_atypique"
# Incidents dont le rapport ne dit pas la cause : groupe à part, jamais clusterisé ni diagnostiqué
UNKNOWN_CAUSE_LABEL = "cause_non_precisee"
NOISE_LABELS = (OUTLIER_LABEL, UNKNOWN_CAUSE_LABEL)
RANDOM_STATE = 42

# Profil de concentration (où agir) : lift vs part attendue + test binomial
CONCENTRATION_COLS = ["site", "equipment_id"]
CONCENTRATION_PERIOD = "Q"  # rafale temporelle par trimestre
CONCENTRATION_MIN_LIFT = 1.5
CONCENTRATION_MIN_OBS = 3
CONCENTRATION_FDR = 0.10
# Preuve forte si q ≤ ce seuil ; entre ce seuil et CONCENTRATION_FDR, la concentration est « à vérifier »
CONCENTRATION_STRONG_Q = 0.05

# Balayage HDBSCAN par partition, sélection par DBCV
MCS_SWEEP = [5, 7, 10, 15, 20, 25]
MIN_SAMPLES = None  # défaut hdbscan (= min_cluster_size)
SWEEP_MAX_OUTLIER_RATE = 0.35  # réglages au-delà exclus de la sélection
SWEEP_MIN_CLUSTERS = 2

# Agent de fusion par remédiation
MERGE_NEIGHBOURS = 4  # k plus proches centroïdes par cluster fin
MERGE_MAX_GROUP = 8  # clusters max par prompt
MERGE_MAX_ROUNDS = 3
MERGE_DBCV_MAX_DROP = 0.15  # chute relative tolérée sans appel au critique
MERGE_DBCV_HARD_DROP = 0.35  # au-delà : veto ferme, le critique ne peut pas l'outrepasser

# Fouille de précurseurs (même équipement, fenêtre glissante, modèle nul par permutation)
PRECURSOR_WINDOWS = [7, 30, 60]
PRECURSOR_N_PERMUTATIONS = 10000  # plancher p = 1e-4 : à 1000, le BH sur ~200 tests rend q ≤ 0.05 inatteignable
PRECURSOR_FDR = 0.05
PRECURSOR_MIN_HITS = 5
PRECURSOR_MIN_EQUIPMENTS = 3

# Extraction de la cause par incident (le clustering porte sur la cause, pas sur le rapport entier)
CAUSE_COL = "cause_racine"
CAUSE_QUOTE_COL = "indice"
CONSEQUENCE_COL = "consequence"
CAUSE_UNKNOWN_TEXT = "cause non précisée dans le rapport"
# Un extrait cité est accepté si sa plus longue portion commune avec le rapport en couvre cette part
QUOTE_MIN_MATCH = 0.8
EXTRACTION_WORKERS = 4

# Fonctionnement nominal : proposé par le diagnostic, accepté seulement si au moins cette part des
# incidents du cluster est au niveau de sévérité le plus bas
NOMINAL_MIN_MILD_SHARE = 0.8
