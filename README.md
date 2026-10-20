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
python main.py                              # étapes par défaut : pdf, segmentation, equipment, report
python main.py --steps segmentation         # une seule étape
python main.py --steps topics               # topic modeling exploratoire (optionnel)
```

| Étape | Module | Sortie | Défaut |
|---|---|---|---|
| 1. `pdf` : extraction des PDF (digitaux + scans) | [`pdf_pipeline/`](pdf_pipeline/README.md) | `outputs/incidents_full.csv` | oui |
| 2. `segmentation` : clustering agentique (modes de défaillance, remédiations, concentration, veille d'escalade, précurseurs, diagnostic) | [`agentic_pipeline/`](agentic_pipeline/README.md) | `outputs/segmentation/` | oui |
| 3. `equipment` : chronologies de pannes et score « bad actor » par équipement | [`equipment_analysis/`](equipment_analysis/README.md) | `outputs/equipment/` | oui |
| 4. `report` : rapport opérationnel HTML statique et autonome (vue d'ensemble de la flotte, plan d'action par voie de remédiation avec ses preuves : citations vérifiées, chiffres de concentration, chronologies d'escalade, liens vers les PDF sources ; section « sans action requise » ; méthode et limites) ; option `--summary template\|llm` | `report/` | `outputs/rapport_operationnel.html` | oui |
| `topics` : topic modeling BERTopic sur les 1800 incidents (exploratoire, non utilisé par la segmentation) | [`topic_modeling/`](topic_modeling/README.md) | `outputs/topics/` | non |

Sorties de l'étape `segmentation` (`outputs/segmentation/`) :

| Fichier | Contenu |
|---|---|
| `incidents_segmented.csv` | un incident par ligne, avec `fine_cluster_id`, `merged_cluster_id`, `cluster_id` (= `merged_cluster_id`), criticité et `assignment_source` (`dense_cluster`, `merged_singleton`, `unclassified_outlier`) |
| `clusters_fine.csv` | modes de défaillance (clusters fins, singletons d'isolés compris) |
| `clusters_merged.csv` | voies de remédiation (clusters fusionnés), triées par coût total décroissant, avec palier de criticité Pareto, diagnostic et recommandations ; remplace l'ancien `clusters_summary.csv` |
| `merge_log.json` | journal auditable de l'agent de fusion |
| `concentration.csv` | concentration par site, équipement et trimestre pour chaque cluster fusionné |
| `escalations.csv` | veille d'escalade : séries (équipement × cluster fusionné, ≥ 3 incidents) dont la sévérité ne fait que monter, classées par coût total ; liste descriptive, sans test |
| `precursor_pairs.csv` | paires précurseur / récurrence, avec lift, q-value et règle de surveillance |
| `precursor_evidence.csv` | incidents concernés par chaque règle précurseur |
| `fine_sweep.csv` | balayage HDBSCAN par partition (DBCV, taux d'isolés) |

Sorties de l'étape `equipment` (`outputs/equipment/`) : `incidents_with_timeline.csv` et `equipment_summary.csv` (voir [`equipment_analysis/`](equipment_analysis/README.md)).

Diagnostic de stabilité UMAP (hors `main.py`, sans appel à l'API) : `.venv/bin/python -m scripts.umap_stability` écrit `outputs/diagnostics/umap_stability.csv`.

Chaque étape garde un cache dans `outputs/cache/` : relancer ne refait que ce qui a changé.

Tests unitaires et de non-régression (sans appel réseau ni LLM) :

```bash
.venv/bin/python -m pytest tests -q
```

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
