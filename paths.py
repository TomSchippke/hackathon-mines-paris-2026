"""Organisation du dossier outputs/ (voir outputs/README.md).

    outputs/
    ├── 1_extraction/      étape 1 (pdf)          : incidents_full.csv          ← à utiliser
    ├── 2_analyse/         étape 2 (analyse)      : rapport, thèmes, causes     ← à utiliser
    ├── 3_verification/    étape 3 (verification) : avis de l'agent vérificateur ← à utiliser
    ├── exploration/       lancements manuels des modules (topic_modeling, cause_extraction)
    └── cache/             interne aux étapes, ne pas lire ni modifier à la main
"""

from pathlib import Path

DATA_CSV = Path("data/incidents_students.csv")
OUTPUTS = Path("outputs")

# Étape 1 : extraction des PDF
EXTRACTION_DIR = OUTPUTS / "1_extraction"
INCIDENTS_FULL = EXTRACTION_DIR / "incidents_full.csv"

# Étape 2 : agent d'analyse
ANALYSIS_DIR = OUTPUTS / "2_analyse"
ANALYSIS_REPORT = ANALYSIS_DIR / "rapport_analyse.md"
ANALYSIS_JOURNAL = ANALYSIS_DIR / "journal.json"

# Étape 3 : agent vérificateur
REVIEW_DIR = OUTPUTS / "3_verification"

# Lancements manuels, hors pipeline principale (python -m topic_modeling, python -m cause_extraction)
EXPLORATION_DIR = OUTPUTS / "exploration"
TOPICS_DIR = EXPLORATION_DIR / "topics"
INCIDENTS_TOPICS = TOPICS_DIR / "incidents_topics.csv"
TOPIC_INFO = TOPICS_DIR / "topic_info.csv"
CAUSES_DIR = EXPLORATION_DIR / "causes"
INCIDENTS_CAUSES = CAUSES_DIR / "incidents_causes.csv"

# Interne : lectures OCR brutes (ocr_doctr/), embeddings (embeddings/) et réponses du LLM (llm/)
CACHE_DIR = OUTPUTS / "cache"
