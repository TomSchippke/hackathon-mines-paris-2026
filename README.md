# Hackathon Mines Paris 2026 – Capgemini

Pipeline agentique d'analyse des rapports d'incidents PV et éolien.

## Données

Les données ne sont **pas versionnées** (volumineuses, fournies par l'organisateur – `data/` est dans le `.gitignore`).
Récupérez-les et placez-les à la racine du repo avec cette arborescence exacte :

```
hackathon-mines-paris-2026/
├── data/
│   ├── incidents_students.csv   # table des incidents (colonne document_path → PDF)
│   ├── pdf_digital/             # 600 PDF avec texte extractible
│   │   └── incident_XXXX.pdf
│   └── pdf_scan/                # 600 PDF scannés (images, OCR nécessaire)
│       └── incident_XXXX.pdf
├── notebooks/
│   └── starterkit_students.ipynb
├── outputs/                     # résultats générés (non versionnés), ex. incidents_full.csv
└── ...
```

Le notebook (dans `notebooks/`, il se place lui-même à la racine du repo) lit `data/incidents_students.csv` et les chemins `document_path` pointent vers
`data/pdf_digital/` et `data/pdf_scan/` : ne renommez pas ces dossiers.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer la pipeline

```bash
python main.py                    # toutes les étapes, l'une après l'autre
python main.py --steps topics     # une seule étape
```

| Étape | Module | Sortie |
|---|---|---|
| 1. `pdf` : extraction des PDF (digitaux + scans) | [`pdf_pipeline/`](pdf_pipeline/README.md) | `outputs/incidents_full.csv` |
| 2. `topics` : topic modeling BERTopic sur les 1800 incidents | [`topic_modeling/`](topic_modeling/README.md) | `outputs/topics/` |
| 3. `segmentation` : segmentation causale & criticité LLM | [`agentic_pipeline/`](agentic_pipeline/) | `outputs/segmentation/` |
| 4. `equipment` : chronologies pannes & audit techniciens | [`equipment_analysis/`](equipment_analysis/) | `outputs/equipment/` |
| 5. `webapp` : tableau de bord interactif Streamlit | [`webapp/`](webapp/) | `streamlit run webapp/app.py` |

Chaque étape garde un cache dans `outputs/cache/` : relancer ne refait que ce qui a changé.

## Extraction des PDF (digitaux + scans)

Les rapports `pdf_digital` et `pdf_scan` n'ont que `site` renseigné dans le CSV. La pipeline
`pdf_pipeline/` lit les 1200 PDF (OCR docTR pour les scans, Tesseract en secours) et produit
`outputs/incidents_full.csv` : les 1800 incidents avec toutes les colonnes remplies.

```bash
brew install tesseract tesseract-lang   # Linux : apt install tesseract-ocr tesseract-ocr-fra
python -m pdf_pipeline                  # ~30 min la première fois, puis quelques secondes (cache)
```

Détails (étapes, choix des OCR, comparaison, ajout de nouveaux fichiers) : [`pdf_pipeline/README.md`](pdf_pipeline/README.md).

## Configuration (Azure OpenAI)

Copiez `.env.example` en `.env` et renseignez vos propres valeurs :

```bash
cp .env.example .env
```

Le fichier `.env` est ignoré par git : **ne committez jamais vos clés** (le repo est public).
