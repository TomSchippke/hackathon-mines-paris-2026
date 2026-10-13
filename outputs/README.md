# outputs/

Tout ce que produit `python main.py`. Le contenu n'est pas versionné (seul ce README l'est) :
il se régénère avec `python main.py`, ou se partage directement entre membres de l'équipe.

```
outputs/
├── 1_extraction/            ← À UTILISER
│   └── incidents_full.csv       les 1800 incidents, toutes colonnes remplies (PDF inclus)
├── 2_topics/                ← À UTILISER
│   ├── incidents_topics.csv     chaque incident avec son topic
│   ├── topic_info.csv           chaque topic : taille, mots-clés, exemples
│   ├── topics_par_site.csv      nombre d'incidents par topic et par site
│   └── barchart_*.html          mots-clés des topics (à ouvrir dans un navigateur)
├── cache/                   ← INTERNE, ne pas lire ni modifier
│   ├── ocr_doctr/               lectures OCR brutes des scans (étape 1)
│   ├── ocr_tesseract/           lectures OCR de secours (étape 1)
│   └── embeddings/              vecteurs des descriptions, Azure OpenAI (étape 2)
└── comparaison_ocr/         ← DOCUMENTATION des choix OCR, pas utilisé par la pipeline
    ├── incidents_full_doctr.csv     extraction avec docTR seul
    ├── incidents_full_tesseract.csv extraction avec Tesseract seul
    └── verite_terrain_scans.csv     40 scans relus à la main
```

- **`1_extraction/` et `2_topics/`** : les résultats, numérotés dans l'ordre des étapes. C'est
  ce qu'on lit dans un notebook ou une étape suivante.
- **`cache/`** : évite de refaire le travail coûteux (≈ 30 min d'OCR, appels Azure payants). Le
  supprimer ne casse rien mais force à tout recalculer. Pour éviter ce calcul à un coéquipier, lui
  envoyer ce dossier.
- **`comparaison_ocr/`** : sert seulement à justifier le choix docTR + Tesseract
  (`python -m pdf_pipeline.compare`, voir `pdf_pipeline/README.md`).

L'organisation est définie dans `paths.py` (racine du repo).
