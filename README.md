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
├── starterkit_students.ipynb
└── ...
```

Le notebook lit `data/incidents_students.csv` et les chemins `document_path` pointent vers
`data/pdf_digital/` et `data/pdf_scan/` : ne renommez pas ces dossiers.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (Azure OpenAI)

Copiez `.env.example` en `.env` et renseignez vos propres valeurs :

```bash
cp .env.example .env
```

Le fichier `.env` est ignoré par git : **ne committez jamais vos clés** (le repo est public).
