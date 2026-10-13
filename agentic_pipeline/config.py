"""Configuration pour le pipeline agentique de segmentation des incidents."""

from __future__ import annotations

from pathlib import Path

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
DEFAULT_SPLIT_COL = "asset_type"

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
