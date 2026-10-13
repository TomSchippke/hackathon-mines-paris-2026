"""Paramètres du topic modeling."""

from pathlib import Path

DEFAULT_INPUT = Path("outputs/incidents_full.csv")  # produit par pdf_pipeline
DEFAULT_OUT_DIR = Path("outputs")

# Embeddings : appels Azure OpenAI par lots.
EMBEDDING_BATCH_SIZE = 16

# Le vocabulaire PV et éolien diffère fortement : clusteriser ensemble ferait surtout ressortir
# cette séparation. Un modèle BERTopic par type d'installation.
SPLIT_COLUMN = "asset_type"

# UMAP : réduction de dimension avant clustering (random_state fixé : résultats reproductibles).
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)

# HDBSCAN : taille minimale d'un topic. Plus petit -> topics plus nombreux et plus fins.
MIN_CLUSTER_SIZE = 10

# Mots-clés des topics : unigrammes et bigrammes ("point chaud", "boîte de jonction"...).
NGRAM_RANGE = (1, 2)

FRENCH_STOPWORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "sur", "au", "aux", "pour",
    "par", "est", "sont", "a", "été", "suite", "lors", "dans", "que", "qui", "ce", "cette", "ces",
    "afin", "plus", "après", "avant", "ou", "d", "l", "à", "avec", "sans", "son", "sa", "ses",
    "il", "elle", "ont", "être", "fait", "ainsi", "puis", "selon", "entre", "sous",
]
