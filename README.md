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
├── outputs/                     # résultats générés (non versionnés), voir outputs/README.md
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
python main.py                                # toutes les étapes, l'une après l'autre
python main.py --steps analyse verification   # seulement certaines étapes
```

```
 1. pdf            PDF + CSV ──> incidents_full.csv (1800 incidents complets)
                                     │
 2. analyse        AGENT D'ANALYSE (LLM Azure) qui appelle des outils Python bornés :
                   overview ─> run_clustering ×N ─> select_clustering ─> extract_causes
                   ─> topic_details / define_theme ─> site_evidence ─> write_report
                                     │  rapport + journal de toutes ses décisions
 3. verification   AGENT VÉRIFICATEUR : contrôles automatiques + relecture du journal et du
                   rapport, comparaison aux chiffres recalculés ─> verdict et constats
```

| Étape | Module | Sortie |
|---|---|---|
| 1. `pdf` : extraction des PDF (digitaux + scans) | [`pdf_pipeline/`](pdf_pipeline/README.md) | `outputs/1_extraction/incidents_full.csv` |
| 2. `analyse` : agent d'analyse (clustering, causes, thèmes, profils, remédiation) | [`analysis_agent/`](analysis_agent/README.md) | `outputs/2_analyse/rapport_analyse.md` |
| 3. `verification` : agent vérificateur | [`review_agent/`](review_agent/README.md) | `outputs/3_verification/verification.md` |

Briques utilisées par l'agent d'analyse, lançables aussi seules pour explorer
(sorties dans `outputs/exploration/`) : [`topic_modeling/`](topic_modeling/README.md) (BERTopic,
embeddings) et [`cause_extraction/`](cause_extraction/README.md) (cause racine par le LLM).
`llm/` contient le client Azure, le cache des réponses et la boucle d'agent commune aux deux agents.

Les résultats à utiliser sont dans les dossiers numérotés ; `outputs/cache/` est interne aux
étapes (relancer ne refait que ce qui a changé). Détail : [`outputs/README.md`](outputs/README.md).

## Extraction des PDF (digitaux + scans)

Les rapports `pdf_digital` et `pdf_scan` n'ont que `site` renseigné dans le CSV. La pipeline
`pdf_pipeline/` lit les 1200 PDF (OCR docTR pour les scans) et produit
`outputs/1_extraction/incidents_full.csv` : les 1800 incidents avec toutes les colonnes remplies.

```bash
python -m pdf_pipeline                  # ~30 min la première fois, puis quelques secondes (cache)
```

Détails (étapes, choix de l'OCR, cache, ajout de nouveaux fichiers) : [`pdf_pipeline/README.md`](pdf_pipeline/README.md).

## Configuration (Azure OpenAI)

Copiez `.env.example` en `.env` et renseignez vos propres valeurs :

```bash
cp .env.example .env
```

Le fichier `.env` est ignoré par git : **ne committez jamais vos clés** (le repo est public).
