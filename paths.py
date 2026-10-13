"""Organisation du dossier outputs/ (voir outputs/README.md).

    outputs/
    ├── 1_extraction/      résultat de l'étape 1, à utiliser : incidents_full.csv
    ├── 2_topics/          résultats de l'étape 2, à utiliser
    ├── cache/             interne aux étapes, ne pas lire ni modifier à la main
    └── comparaison_ocr/   évaluation des moteurs OCR (documentation des choix)
"""

from pathlib import Path

DATA_CSV = Path("data/incidents_students.csv")
OUTPUTS = Path("outputs")

# Résultats à utiliser
EXTRACTION_DIR = OUTPUTS / "1_extraction"
INCIDENTS_FULL = EXTRACTION_DIR / "incidents_full.csv"
TOPICS_DIR = OUTPUTS / "2_topics"

# Interne : lectures OCR brutes (ocr_doctr/, ocr_tesseract/) et embeddings (embeddings/)
CACHE_DIR = OUTPUTS / "cache"

# Évaluation de l'OCR : extractions moteur par moteur et vérité terrain relue à la main
OCR_COMPARISON_DIR = OUTPUTS / "comparaison_ocr"
OCR_GROUND_TRUTH = OCR_COMPARISON_DIR / "verite_terrain_scans.csv"
