# outputs/

Tout ce que produit `python main.py`. Le contenu n'est pas versionné (seul ce README l'est) :
il se régénère avec `python main.py`, ou se partage directement entre membres de l'équipe.

```
outputs/
├── 1_extraction/            ← À UTILISER
│   └── incidents_full.csv       les 1800 incidents, toutes colonnes remplies (PDF inclus)
├── 2_analyse/               ← À UTILISER
│   ├── rapport_analyse.md       LE LIVRABLE : causes, profils de site, plan de remédiation
│   ├── rapport_interactif.html  le même rapport avec graphiques, en un fichier à envoyer (python main.py --steps export)
│   ├── journal.md               toutes les décisions de l'agent (journal.json : version machine)
│   ├── incidents_analyse.csv    chaque incident : cause, conséquence, leurs thèmes, action corrective
│   ├── themes.csv               thèmes de causes et de conséquences nommés par l'agent
│   ├── heatmap_<type>.png/.csv  heatmap thèmes de causes × thèmes de conséquences (coût, incidents)
│   ├── fiches_causes.csv        fiche et criticité Pareto de chaque thème de cause
│   ├── clustering_essais.csv    scores de chaque essai de clustering
│   └── decisions.json           clusterings choisis, chaînes prioritaires et contenu du rapport
├── 3_verification/          ← À UTILISER
│   ├── verification.md          verdict, contrôles automatiques, constats du vérificateur
│   └── journal.md               déroulé de la vérification
├── validation/              ← À REMPLIR À LA MAIN (jamais écrasé)
│   └── inferences_a_valider.csv 40 causes inférées à juger (colonne verdict_humain : oui / non)
├── exploration/             ← lancements manuels (python -m topic_modeling / cause_extraction)
└── cache/                   ← INTERNE, ne pas lire ni modifier
    ├── ocr_doctr/               lectures OCR brutes des scans (étape 1)
    ├── embeddings/              vecteurs des descriptions, Azure OpenAI (étape 2)
    └── llm/                     réponses du LLM (étapes 2 et 3)
```

- **Dossiers numérotés** : les résultats, numérotés dans l'ordre des étapes. C'est
  ce qu'on lit dans un notebook ou une étape suivante.
- **`cache/`** : évite de refaire le travail coûteux (≈ 30 min d'OCR, appels Azure payants, sessions de l'agent). Le
  supprimer ne casse rien mais force à tout recalculer. Pour éviter ce calcul à un coéquipier, lui
  envoyer ce dossier.

L'organisation est définie dans `paths.py` (racine du repo).
