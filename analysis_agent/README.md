# Étape 2 : agent d'analyse

Un agent LLM (le modèle de chat du `.env`, sur Azure OpenAI) analyse les 1800 incidents et rédige
le **rapport d'analyse** : causes racines, causes dominantes par site, association de chaque site
à son profil (A, B, C du sujet), plan de remédiation par site.

```bash
python main.py --steps analyse       # ou : python -m analysis_agent
```

## Le principe : l'agent décide, le code calcule et vérifie

L'agent ne lit pas les 1800 rapports et ne calcule rien lui-même. Il appelle des **outils** :
des fonctions Python qui font les calculs et lui renvoient des résultats résumés. Lui choisit
quoi essayer, quoi garder, comment nommer, et justifie chaque choix.

```
        ┌──────────────── Agent LLM (Azure) ────────────────┐
        │ lit les résultats, raisonne, choisit l'outil suivant │
        └───────┬─────────────────────────────────▲──────────┘
       appel d'outil (JSON)                résultat (JSON)
                ▼                                 │
        ┌──────────────────── Code Python ─────────────────────┐
        │ vérifie les arguments (bornes, ordre des étapes)       │
        │ calcule (UMAP, HDBSCAN, statistiques, LLM par incident)│
        │ refuse avec la raison si la demande est invalide       │
        └───────────────────────────────────────────────────────┘
```

Boucle commune aux deux agents : `llm/agent.py`. Chaque tour est consigné dans le journal.

## Causes et conséquences : deux clusterings reliés

Un cluster « feu » construit sur la description brute mélange le feu comme **cause** (feu de forêt,
étincelles de travaux) et le feu comme **conséquence** (un arc électrique qui enflamme la toiture).
Le LLM extrait donc de chaque rapport deux phrases courtes, et chacune a son propre clustering :

```
description ──LLM──> cause_racine  ("étincelles de meuleuse pendant des travaux")
             └─────> consequence   ("incendie de la zone de travaux, arrêt 6 h")
                          │                          │
                  clustering des CAUSES      clustering des CONSÉQUENCES
                          └───────────┬──────────────┘
                     liens_cause_consequence : quelles causes mènent à quelles
                     conséquences, à quel coût, sur quels sites
```

Quand le rapport ne donne pas la cause, le LLM écrit exactement « cause non précisée dans le
rapport » : ces incidents forment leur propre cluster, visible et chiffré.

## Les outils, dans l'ordre

| # | Outil | Ce qu'il fait | Bornes et contrôles |
|---|---|---|---|
| 1 | `overview` | incidents par site, sévérité, coûts ; profils de site du sujet ; bornes et seuils | — |
| 2 | `extract_causes(asset_type)` | cause, conséquence, composant, facteur, usure, maintenance, action corrective de chaque incident (LLM, `cause_extraction`) | le site n'est pas donné au LLM ; réponses en cache |
| 3 | `run_clustering(asset_type, dimension, min_cluster_size)` | un essai UMAP + HDBSCAN sur les causes **ou** les conséquences ; renvoie stabilité, % non classés, DBCV, nombre de clusters, mots-clés | `dimension` ∈ {cause, consequence} ; `min_cluster_size` entre **5 et 50** ; **5 essais max** par type et dimension ; exige les causes extraites |
| 4 | `select_clustering(asset_type, dimension, run_id, justification)` | fixe l'essai utilisé | refusé si **stabilité < 0.8**, **non classés > 15 %** ou nombre de clusters hors **3–40** |
| 5 | `topic_details(asset_type, dimension, topics)` | pour 1 à 10 clusters : phrases typiques, sites, sévérité, saisons, météo, facteurs, coût, cohésion | **10 clusters max** par appel |
| 6 | `define_themes(asset_type, dimension, themes)` | range des clusters dans 1 à 20 thèmes nommés ; chaque thème accepté ou refusé séparément | nom ≤ 60 caractères ; cluster déjà rangé refusé ; **fusion refusée** si la cohésion tombe sous 90 % de celle des clusters d'origine |
| 7 | `remove_theme(theme_id)` | défait un thème | — |
| 8 | `liens_cause_consequence(asset_type)` | paires cause → conséquence les plus fréquentes (part, coût moyen et total, sites), conséquences de chaque cause, causes de chaque conséquence | exige tous les clusters rangés dans un thème |
| 9 | `site_evidence(asset_type)` | par site : thèmes de causes et de conséquences, facteurs, paires, % usure, % maintenance, sévérité, coût, météo, actions fréquentes | idem |
| 10 | `write_report(...)` | **outil final** : synthèse, causes racines, chaînes cause → conséquence, et pour chaque site profil + justification + causes + plan d'actions | exactement les 6 sites ; **profils A, B, C une fois chacun** par type ; 2 à 8 actions par site, priorité haute/moyenne/basse |

Pourquoi ces bornes : le modèle (gpt-4.1-nano) est petit. Sans bornes, il pourrait lancer des
clusterings absurdes (`min_cluster_size=1`), tourner en rond, attribuer le même profil à deux
sites ou rendre un rapport incomplet. Chaque refus lui explique quoi corriger.

Pourquoi l'agent choisit le clustering : les bons réglages dépendent des données. Ils ne sont
donc pas figés dans le code : l'agent les choisit à chaque lancement, dans des bornes sûres, en
s'appuyant sur des scores calculés.

## Le rapport : texte de l'agent, chiffres du code

`write_report` ne reçoit que du texte (synthèse, justifications, actions). Les tableaux du
rapport (essais de clustering, thèmes, causes par site, paires cause → conséquence, indicateurs par site) sont calculés par
`report.py` à partir des données : aucun chiffre des tableaux ne peut être inventé. Les chiffres
cités dans le texte, eux, sont contrôlés par l'agent vérificateur (étape 3).

## Sorties (`outputs/2_analyse/`)

| Fichier | Contenu |
|---|---|
| `rapport_analyse.md` | le livrable (absent si l'agent n'a pas terminé) |
| `journal.md` | chaque tour : raisonnement, outil appelé, arguments, résultat, refus |
| `journal.json` | le même journal, lu par l'agent vérificateur |
| `incidents_analyse.csv` | chaque incident : cause, conséquence, leurs clusters et thèmes, composant, facteur, action corrective |
| `themes.csv` | thèmes de causes et de conséquences : nom, description, clusters, taille, cohésion, justification |
| `clustering_essais.csv` | scores de chaque essai de clustering |
| `decisions.json` | clustering choisi par type, contenu rédigé du rapport, scores |

## Coût et reproductibilité

- Premier lancement : environ 1800 appels pour les causes (modèle nano, 2 en parallèle) et une
  centaine de tours d'agent (historique compacté : seuls les derniers résultats sont renvoyés en entier).
- Toutes les réponses du LLM sont en cache (`outputs/cache/llm/`), avec une température de 0 :
  relancer rejoue la même session sans aucun appel. Changer les données, un prompt ou une borne
  relance l'agent à partir du premier tour concerné.

## Réglages (`config.py`)

Bornes des outils, seuils de qualité, profils de site du sujet et prompt de l'agent. Les seuils
sont des valeurs de départ : les ajuster après un premier lancement en lisant `journal.md`.
