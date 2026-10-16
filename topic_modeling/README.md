# Topic modeling des incidents

Regroupe les 1800 incidents (texte, PDF digitaux et scans) par thème avec **BERTopic**, à partir
de `outputs/1_extraction/incidents_full.csv` produit par `pdf_pipeline`.

## Utilisation

```bash
python -m topic_modeling                         # lancement seul, pour explorer
python -m topic_modeling --min-cluster-size 15
```

Dans la pipeline, l'agent d'analyse utilise les embeddings et les réglages UMAP de ce module, et
choisit lui-même `min_cluster_size` (outil `run_clustering`). Le lancement seul ci-dessus garde des
réglages fixes, pour explorer.

Il faut le `.env` (Azure OpenAI) pour calculer les embeddings. Une fois en cache, plus aucun appel
à l'API n'est fait.

## Étapes

| Étape | Ce qui est fait | Pourquoi |
|---|---|---|
| 1. Embeddings | Chaque description devient un vecteur (Azure OpenAI `text-embedding-3-small`). | Deux incidents décrits avec des mots différents mais un même problème ont des vecteurs proches. |
| 2. Séparation PV / éolien | Un modèle BERTopic par `asset_type`. | Le vocabulaire des deux domaines diffère : un modèle commun ferait d'abord ressortir PV contre éolien, avant les vraies causes. |
| 3. UMAP | Réduit les vecteurs de 1536 à 5 dimensions. | Le clustering par densité marche mal en grande dimension. |
| 4. HDBSCAN | Forme les topics (au moins `min_cluster_size` incidents) ; les incidents isolés sont marqués `-1` (non classés). | Pas besoin de fixer le nombre de topics à l'avance. |
| 5. Mots-clés | Mots et paires de mots fréquents dans un topic et rares ailleurs (c-TF-IDF). | Donne une première idée du thème de chaque topic. |

## Sorties (`outputs/exploration/topics/`)

| Fichier | Contenu |
|---|---|
| `incidents_topics.csv` | Les 1800 incidents avec `topic_id` (ex. `solar_3`, `wind_-1` = non classé), `topic_label` (mots-clés) et `topic_probability`. |
| `topic_info.csv` | Un topic par ligne : taille, mots-clés, 3 descriptions représentatives. |
| `topics_par_site.csv` | Nombre d'incidents par topic et par site : base pour trouver les causes dominantes de chaque site. |
| `barchart_solar.html`, `barchart_wind.html` | Mots-clés des topics, graphique interactif à ouvrir dans un navigateur. |

## Cache

Les embeddings sont stockés dans `outputs/cache/embeddings/<modèle>.npz`, indexés par l'empreinte
(SHA-1) du texte : un texte déjà vectorisé n'est jamais renvoyé à l'API, et le cache est
sauvegardé après chaque lot (une interruption ne fait rien perdre). Le clustering, lui, est
refait à chaque lancement (quelques secondes) : on peut changer les réglages librement.

## Réglages (`config.py`)

- `MIN_CLUSTER_SIZE` (10) : plus petit -> topics plus nombreux et plus fins.
- `NGRAM_RANGE` ((1, 2)) : mots seuls et paires de mots dans les mots-clés.
- `UMAP_PARAMS` : `random_state` fixé, les résultats sont reproductibles d'un lancement à l'autre.
- `FRENCH_STOPWORDS` : mots ignorés dans les mots-clés.
