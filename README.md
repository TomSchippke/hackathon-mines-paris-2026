# Hackathon Mines Paris 2026 – Capgemini

Pipeline agentique d'analyse de rapports d'incidents, qui produit un rapport d'actions pour le client.

Les agents ne savent **rien du problème a priori** : ils n'en connaissent que quelques lignes écrites
par l'utilisateur dans [`context/contexte.md`](context/contexte.md). Tout le reste (facteurs à
l'origine des incidents, composants, catégories de conséquences, profils de site éventuels, mesures
préventives types) est déduit de ce texte et des données à l'étape de **cadrage** ([`framing/`](framing/README.md)).

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

## Décrire le problème

Écrire en quelques lignes, dans `context/contexte.md`, ce qu'il faut savoir du problème : qui est le
client, quels sites, ce qu'on sait d'eux, l'objectif. Exemple fourni :

```
Le client exploite 6 sites de production d'énergie renouvelable : 3 centrales solaires
photovoltaïques et 3 parcs éoliens. Les 3 sites d'un même type sont exposés à des conditions
semblables. [...]
```

Pour que chaque site soit associé à un profil (plutôt que seulement caractérisé), décrire les profils
dans ce même fichier (voir [`framing/README.md`](framing/README.md)). Après le cadrage, relire
`outputs/1b_cadrage/cadrage.md` ; pour le corriger, copier `cadrage.json` en `context/cadrage.json`.

## Lancer la pipeline

```bash
python main.py                                # toutes les étapes, l'une après l'autre
python main.py --steps analyse verification   # seulement certaines étapes
python main.py --steps cadrage                # refaire le cadrage après avoir modifié context/contexte.md
python main.py --steps webapp                 # tableau de bord Streamlit (à la demande)
python main.py --steps export                 # rapport interactif en un fichier HTML, à envoyer
```

**Partager le rapport** : `outputs/2_analyse/rapport_interactif.html` est un fichier unique (~6 Mo, plotly.js
intégré) qui s'ouvre d'un double-clic dans un navigateur, sans Python ni connexion. Mêmes graphiques
interactifs que la webapp ; les filtres sont remplacés par des sections dépliables (une heatmap par site,
toutes les fiches de causes, tous les plans par site).

```
 1. pdf            PDF + CSV ──> incidents_full.csv (1800 incidents complets)
                                     │
 1b. cadrage       context/contexte.md + échantillon des données ──> LLM ──> listes fermées, segments,
                   profils éventuels, mesures types ; vérifiés par le code ──> cadrage.json (à relire)
                                     │
 2. analyse        AGENT D'ANALYSE (LLM Azure) qui appelle des outils Python bornés :
                   extraction cause/conséquence (+ hypothèse « manque de maintenance »)
                   ─> clustering + nommage/fusion des thèmes (par segment × causes/conséquences)
                   ─> heatmap causes × conséquences ─> chaînes prioritaires ─> profils ou caractérisation
                   ─> rédaction : faits Python, reformulés par le LLM, vérifiés par un validateur
                   ─> fiches des causes (criticité Pareto) et plans par site
                   ─> juge : relit chaque texte, corrections acceptées seulement si validées
                                     │  rapport + journal de toutes ses décisions
 3. verification   contrôles automatiques + AGENT VÉRIFICATEUR (avis sur la cohérence) ─> verdict
```

| Étape | Module | Sortie |
|---|---|---|
| 1. `pdf` : extraction des PDF (digitaux + scans) | [`pdf_pipeline/`](pdf_pipeline/README.md) | `outputs/1_extraction/incidents_full.csv` |
| 1b. `cadrage` : cadre d'analyse déduit du contexte et des données | [`framing/`](framing/README.md) | `outputs/1b_cadrage/cadrage.md` |
| 2. `analyse` : workflow d'analyse (causes, conséquences, thèmes, profils, priorisation, rapport) | [`analysis_agent/`](analysis_agent/README.md) | `outputs/2_analyse/rapport_analyse.md` |
| 3. `verification` : agent vérificateur | [`review_agent/`](review_agent/README.md) | `outputs/3_verification/verification.md` |
| `export` : le même rapport en un fichier HTML autonome, à envoyer | [`webapp/export_html.py`](webapp/export_html.py) | `outputs/2_analyse/rapport_interactif.html` |
| 4. `webapp` : **le rapport en version interactive** (synthèse exécutive, heatmap, Pareto et fiches diagnostic, sites et plans d'actions, explorateur + vérification) | [`webapp/app.py`](webapp/app.py) | Streamlit, lecture seule des sorties |

Cette branche (`branch-octave-tom-merged`) réunit le workflow de Tom (extraction cause/conséquence,
4 clusterings, agent aux outils bornés, rédaction validée, juge, vérificateur) et les idées de la
branche d'Octave (fiche diagnostic par cluster, criticité Pareto, webapp Streamlit).

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
