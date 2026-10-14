# Étape 2 : workflow d'analyse

Analyse les 1800 incidents et produit le **rapport d'analyse** (`outputs/2_analyse/rapport_analyse.md`) :
causes racines, chaînes cause → conséquence, association de chaque site à son profil (A, B, C du
sujet), écarts au profil, plan de remédiation par site et priorisation à l'échelle du parc.

```bash
python main.py --steps analyse       # ou : python -m analysis_agent
```

## Le principe

- **Le code fixe l'ordre des phases.** Un petit modèle (gpt-4.1-nano) se perd dans un long plan ;
  il suit bien une consigne courte avec 2 ou 3 outils.
- **L'agent LLM décide** dans chaque phase : réglages de clustering, nom et regroupement des
  thèmes, profil de chaque site.
- **Le code calcule tous les chiffres** (`facts.py`) et refuse toute décision hors bornes.
- **Le LLM ne rédige qu'en reformulant des faits calculés**, et un validateur vérifie chaque texte
  (`writing.py`).

```
0. code       extraction : cause, type de cause, conséquence, catégorie de conséquence (LLM, en cache)
1. agent      ×4 (solar/wind × causes/conséquences) : clustering, puis nommage des thèmes
2. agent      profils : points distinctifs de chaque site -> profil A, B, C
3. code + LLM rédaction : faits Python -> reformulation LLM -> validateur -> texte Python si échec
```

Si l'agent n'atteint pas l'objectif d'une phase, le code la termine (meilleur clustering conforme ;
thème nommé par sa catégorie ou son facteur dominant ; profils selon les règles d'attentes) et le
signale dans le journal et dans le rapport (« nommé par : code », « attribution automatique »).

## Phase 0 : extraction (`cause_extraction/`)

Pour chaque incident, le LLM renvoie un JSON imposé : `cause_racine`, `cause_type` (explicite,
inférée, ou inconnue) et `indice` (extrait du rapport recopié mot pour mot, **vérifié par le code** :
sans extrait retrouvé, la cause est annulée), `consequence` et `categorie_consequence` (liste
fermée : perte de production, arrêt de sécurité, déclenchement de protection, dégât mineur, dégât
majeur, incendie...), facteur principal, composant, liens avec l'usure et la maintenance, action
corrective. Toutes les formulations de « cause non précisée » sont ramenées à une seule valeur ; ces
incidents sont **exclus du clustering des causes** et comptés à part.

## Phase 1 : clustering et thèmes (agent)

**Clustering** (`run_clustering`, `select_clustering`) : UMAP + HDBSCAN (méthode BERTopic) sur les
phrases de causes ou de conséquences, sans nombres ni mots génériques ; `min_cluster_size` entre
**5 et 50**, **5 essais max**. Chaque essai reçoit une qualité en mots fixée par des seuils (DBCV <
0.5 = séparation faible). Le choix est refusé si l'essai n'est pas conforme (stabilité < 0.8, non
classés > 15 %, clusters hors 3–40), ou s'il a une séparation faible alors qu'un essai conforme de
séparation au moins moyenne existe.

**Thèmes** : les clusters sont décrits dans la consigne (facteur ou catégorie dominant, mots-clés,
phrases typiques), avec les **fusions suggérées par le code**.

| Outil | Ce qu'il fait | Contrôles |
|---|---|---|
| `define_theme(name, topics)` | crée un thème (plusieurs clusters = fusion) | un cluster déjà rangé est ignoré ; nom de cause refusé s'il contient une conséquence, « incendie », « détection » ou « indéterminé » ; fusion refusée si les facteurs (causes) ou catégories (conséquences) dominants diffèrent, ou si la cohésion chute |
| `compare_clusters(a, b)` | similarité des centres de deux clusters, facteur dominant de chacun, exemples, avis de fusion | — |
| `merge_themes(theme_ids, name)` | fusionne des thèmes déjà créés (doublon) | mêmes contrôles que `define_theme` |

**Règle de doublon** : deux clusters ou thèmes dont les centres ont une similarité ≥ 0.8 **et** le même
facteur (ou catégorie) dominant. La similarité seule ne suffit pas : sur nos données, « maintenance »
et « usure » sont à 0.85 sans être la même cause, alors que les vrais doublons étaient à 0.80 et 0.84
avec le même facteur. La phase ne se termine que lorsqu'il ne reste **ni cluster sans thème, ni
doublon** ; sinon le code complète et fusionne lui-même (signalé).

## Phase 2 : profils (agent)

| Outil | Ce qu'il fait | Bornes et contrôles |
|---|---|---|
| `site_evidence(asset_type)` | pour chaque site : **points distinctifs** (critères où il est seul en tête ou en queue avec un écart net), causes connues les plus fréquentes, part de causes non précisées ; critères des profils mesurés sur chaque site | — |
| `assign_profiles(assignments)` | associe chaque site à un profil | les 6 sites ; **A, B, C une fois chacun** par type |

## Phase 3 : rédaction contrôlée (`facts.py`, `writing.py`)

Tout ce que le rapport affirme est calculé en Python :

| Fait | Calcul |
|---|---|
| Points distinctifs | critère où le site est seul en tête (ou en queue), avec au moins 3 points d'écart et 1.5 fois la valeur suivante ; critères à 0 % ou égaux partout ignorés |
| Causes dominantes | 3 thèmes de causes **connues** les plus fréquents du site (hors « non précisée » et « non classé ») |
| Plan d'actions | 3 causes connues au plus fort **coût total** (fréquence × coût moyen), **hors arrêts de sécurité nominaux** ; priorité selon leur part du coût du site (≥ 25 % haute, ≥ 10 % moyenne) |
| Arrêts de sécurité nominaux | arrêt automatique de gravité mineure (`config.NOMINAL_*`) : la protection a fonctionné, ce n'est pas une panne ; présentés à part comme perte structurelle liée à l'exposition du site, avec des leviers d'optimisation, et exclus des plans et de la priorisation |
| Priorisation du parc | 8 couples (site, cause) au plus fort coût, hors arrêts nominaux, avec le coût par an |
| Origine des causes | part de causes explicites / inférées parmi les causes dominantes de chaque site |
| Robustesse | meilleure association site → profil recalculée avec les **seules causes explicites** : identique ou non |
| Écarts au profil | règles « profil → attentes » (`config.PROFILE_EXPECTATIONS`) comparées aux mesures |
| Pourcentages | **toujours** en part de l'ensemble des incidents du site |

Le LLM reformule ces faits : justification du profil, causes dominantes, synthèse, et pour chaque
cause du plan une **mesure préventive ou organisationnelle à l'échelle du site** (type : programme
d'inspection, nettoyage planifié, équipement de protection, procédure, monitoring...), avec son
**mécanisme** (pourquoi elle réduit la cause). Le **validateur** rejette un texte si :
- un nombre cité n'existe pas dans les faits (arrondi toléré, « 1,4 M€ » accepté) ;
- un phénomène cité (corrosion, foudre, chaleur...) est absent des faits ;
- il contient le mot « rang » ;
- une mesure est vague (« renforcer la maintenance ») ou curative (commence par « remplacer »,
  « réparer »...), ou son mécanisme ne parle pas de la cause.

Si le LLM échoue, la mesure vient d'un **catalogue de mesures préventives par facteur**
(`config.PREVENTIVE_CATALOGUE`), qui passe lui-même le validateur.

Si le site ne vérifie nettement aucune attente de son profil, la justification doit dire qu'il est
**attribué par élimination**.

## Phase 4 : relecture par le juge (`judge.py`)

Chaque texte (justification, causes dominantes, chaque mesure, synthèse) est relu séparément par un
juge LLM avec une **grille** : affirmation absente des faits, extrapolation (« coût faible donc bonne
maintenance »), mesure générique (équipement ? défaut visé ? fréquence ? réduit vraiment la cause ?),
mécanisme circulaire, synthèse sans site ni coût, coquilles. Le juge propose une correction ; **le code
ne l'accepte que si elle passe le même validateur** que la rédaction, sinon le texte d'origine est
gardé. Chaque décision est dans le journal et le décompte dans l'en-tête du rapport.

## Validation manuelle des causes inférées

`validation_inferences.py` tire (graine fixe) 20 incidents à cause inférée par type d'installation
dans `outputs/validation/inferences_a_valider.csv`, avec une colonne vide `verdict_humain` à remplir
(« oui » / « non »). Le fichier est créé une seule fois et **jamais écrasé**. Le rapport affiche le
taux d'accord s'il est rempli, sinon « validation manuelle en cours ».

Le LLM réessaie 2 fois avec la liste des erreurs, puis le texte est remplacé par une phrase
générée en Python ; c'est consigné dans `journal.md`.

## Sorties (`outputs/2_analyse/`)

| Fichier | Contenu |
|---|---|
| `rapport_analyse.md` | le livrable (8 sections) |
| `journal.md` / `journal.json` | chaque phase : raisonnement, outils, refus, rédaction (essais, erreurs du validateur), reprises par le code |
| `incidents_analyse.csv` | chaque incident : cause, type de cause, conséquence, catégorie, clusters, thèmes, action |
| `themes.csv` | thèmes de causes et de conséquences, avec qui les a nommés |
| `clustering_essais.csv` | scores et qualité de chaque essai |
| `decisions.json` | clusterings choisis, profils, contenu du rapport |

## Modèles

Chaque usage peut avoir son propre déploiement Azure dans le `.env` (voir `.env.example`) :
`AZURE_OPENAI_CHAT_DEPLOYMENT_EXTRACTION`, `..._ANALYSIS`, `..._WRITING`, `..._REVIEW`. Sans ces
variables, `AZURE_OPENAI_CHAT_DEPLOYMENT` est utilisé partout (aujourd'hui gpt-4.1-nano).

## Coût et reproductibilité

Toutes les réponses du LLM sont en cache (`outputs/cache/llm/`), avec une température de 0 :
relancer rejoue tout sans aucun appel. Changer un prompt ou une borne relance seulement ce qui en dépend.
