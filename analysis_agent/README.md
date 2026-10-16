# Étape 2 : workflow d'analyse

Analyse les incidents et produit le **rapport d'analyse** (`outputs/2_analyse/rapport_analyse.md`) :
causes racines découvertes dans les données, causes contributives, chaînes cause → conséquence,
caractérisation de chaque site (ce qui le distingue des autres sites de son segment), **plan d'actions**
(une mesure par couple site × cause, dans la limite de 80 % du coût et de 12 mesures) et causes isolées.

L'agent ne sait **rien du problème a priori** : domaine, segments, libellés et ordre des sévérités
viennent du **cadrage** ([`framing/`](../framing/README.md)), déduit de `context/contexte.md` et des
données. Aucune liste de causes, de facteurs ou de profils n'est fixée à l'avance.

```bash
python main.py --steps analyse       # ou : python -m analysis_agent
```

Premier lancement sur de vraies données : toutes les étapes d'appel au LLM sont refaites (extraction,
affectation, composants, rédaction, juge).

## Le principe

- **Le code fixe l'ordre des phases.** Un petit modèle se perd dans un long plan ; il suit bien une consigne
  courte avec 2 ou 3 outils.
- **L'agent LLM décide** dans chaque phase : réglages de clustering, nom et regroupement des thèmes,
  chaînes prioritaires, arrêts nominaux.
- **Le code calcule tous les chiffres** (`facts.py`) et refuse toute décision hors bornes.
- **Le LLM affecte chaque phrase à un thème ou à « aucun »**, par lots (`assignment.py`) ; le code relance le
  clustering sur les phrases « aucun » tant qu'elles sont trop nombreuses.
- **Le LLM ne rédige qu'en reformulant des faits calculés**, et un validateur vérifie chaque texte
  (`writing.py`). Un juge relit ensuite, et sa correction n'est acceptée que si elle passe le même validateur
  (`judge.py`).

```
0.  code      extraction par incident : cause racine, causes contributives (avec extrait vérifié), conséquence
0b. code+LLM  composants bruts de chaque segment regroupés (normalisation, toutes les valeurs rangées)
1.  pour chaque segment × (causes, conséquences) :
      a. agent    clustering : essais de min_cluster_size, choix (select_clustering)
      b. agent    thèmes : nommer / regrouper les clusters (define_theme, merge_themes)
      c. code+LLM affectation : chaque phrase -> un thème ou « aucun » (assign_phrases) ;
                  tant que « aucun » > 5 % (au plus 2 tours) : reclustering des seules phrases « aucun »,
                  thèmes nommés par l'agent sur cet essai, réaffectation des seules phrases restées « aucun »
2.  agent     priorités : heatmap causes × conséquences -> chaînes prioritaires (select_priority_chains)
3.  agent     arrêts nominaux : thèmes de conséquences qui sont un fonctionnement normal de protection
              (flag_nominal, refusé si moins de 80 % des incidents du thème sont à la gravité la plus faible)
4.  code+LLM  rédaction : caractérisation et causes dominantes de chaque site, une mesure par couple
              (site, cause) du plan, fiche de chaque cause, synthèse ; puis relecture par le juge
```

Si l'agent n'atteint pas l'objectif d'une phase, le code la termine (meilleur clustering conforme ; thème
nommé d'après ses mots-clés ; chaînes les plus coûteuses ; aucun arrêt nominal) et le signale dans le
journal et dans le rapport (« nommé par : code », « CHOIX AUTOMATIQUE »).

## Phase 0 : extraction (`cause_extraction/`)

Pour chaque incident, le LLM renvoie un JSON imposé : `cause_racine` (le phénomène qui a déclenché la
chaîne), `cause_type` (explicite, inférée ou inconnue) et `indice` (extrait recopié du rapport), 0 à 4
**causes contributives** (conditions qui ont rendu l'incident possible ou pire, chacune avec son propre
extrait), la `consequence` en texte libre, le `composant_brut`, l'hypothèse de maintenance et l'action
corrective. Le code vérifie chaque extrait : une cause sans extrait retrouvé est annulée, une cause
contributive sans extrait est écartée et comptée (`contributives_rejetees`). Les incidents dont la cause
est inconnue ou inférée avec une confiance faible sont exclus de la découverte et comptés à part.

## Phase 1 : clustering, thèmes, affectation

**Clustering** (`run_clustering`, `select_clustering`) : UMAP + HDBSCAN sur les phrases de causes (racines
et contributives) ou de conséquences, par segment. `min_cluster_size` entre **5 et 50**, **5 essais max**.
Chaque essai reçoit une qualité en mots, fixée par des seuils (DBCV, stabilité, non classés, nombre de
clusters). Le choix est refusé si l'essai n'est pas conforme.

**Thèmes** : les clusters sont décrits dans la consigne (mots-clés, phrases typiques, fusions suggérées par
le code). L'agent appelle `define_theme` (un thème = un ou plusieurs clusters) et `merge_themes`. Un nom de
cause qui contient une conséquence (arrêt, perte, incendie...) est refusé. Une fusion est refusée si les
phrases réunies sont nettement moins semblables que dans les clusters d'origine.

**Affectation** (`assignment.py`, `tools.py`) : chaque phrase reçoit l'identifiant de son thème, ou
`aucun` si aucun thème ne décrit le même phénomène (le LLM ne doit jamais forcer). Les appels sont
groupés par lots de 20 phrases ; une réponse absente ou invalide donne « aucun ».

**Reprise** (`recluster_none`) : HDBSCAN relancé sur les seules phrases « aucun » (au moins 5 par cluster).
L'essai de reprise est thématisé par l'agent, puis seules les phrases restées « aucun » sont réaffectées.
La boucle s'arrête quand la part de « aucun » est ≤ 5 % ou après 2 reprises. Ce qui reste est présenté
comme **causes isolées** (section 9 du rapport) : ce n'est pas un défaut caché.

## Phase 2 : chaînes prioritaires (agent)

La heatmap croise les thèmes de causes **connues** (lignes) et les thèmes de conséquences (colonnes) ; une
case = nombre d'incidents et coût total (arrêts nominaux exclus).

| Outil | Ce qu'il fait | Bornes et contrôles |
|---|---|---|
| `heatmap_hotspots(site=None, top_k)` | cases les plus coûteuses, sur le segment ou sur un site | `top_k` entre 1 et 10 |
| `select_priority_chains(chains, justification)` | **outil final** : chaînes cause → conséquence prioritaires | **3 à 5** chaînes parmi les cases proposées |

S'il ne choisit pas, le code prend les 3 cases les plus coûteuses (signalé).

## Phase 3 : arrêts de sécurité nominaux (agent)

Un arrêt nominal est le fonctionnement normal d'une protection (arrêt de sécurité automatique sans dommage),
pas une panne. L'agent lit `nominal_evidence` (incidents, part à la gravité la plus faible, coût moyen,
exemples) puis appelle `flag_nominal`. Le code refuse tout thème dont moins de **80 %** des incidents sont à
la gravité la plus faible. Une liste vide est permise : sans arrêt nominal, la perte est comptée comme une panne.
Les incidents nominaux sont présentés à part (perte structurelle) et exclus du plan et de la priorisation.

## Phase 4 : rédaction contrôlée (`facts.py`, `writing.py`)

Tout ce que le rapport affirme est calculé en Python :

| Fait | Calcul |
|---|---|
| Caractérisation d'un site | critères où le site est nettement en tête ou en queue de son segment (« points distinctifs ») ; un critère par thème de cause, mesuré comme cause racine **ou** contributive |
| Causes dominantes | 3 thèmes de causes **connues** les plus fréquents du site |
| Couples du plan | par (site, thème de cause) : incidents, coût total, coût par an ; au moins 5 incidents ; hors arrêts nominaux |
| Portée | « spécifique au site » si le site est nettement en tête sur ce thème, sinon « commun au segment » |
| Plan | couples classés par coût par an, gardés jusqu'à 80 % du coût des causes connues (le couple qui franchit le seuil est gardé), au plus **12** mesures |
| Causes isolées | phrases de cause sans thème, après les reprises |
| Synthèse | coût total, causes non précisées, arrêts nominaux, couverture du plan, 3 mesures prioritaires, chaînes |

**Une mesure par couple (site, cause)** : le LLM rédige UNE action terrain sur ce site contre cette cause
(quoi, sur quel équipement, à quelle fréquence ou sur quel déclencheur), avec son mécanisme. Le
**validateur** rejette une mesure si :
- elle est curative (« remplacer », « réparer »...) ou vague (« renforcer la maintenance ») ;
- elle ne nomme pas le site (ni « ce site ») ou cite un autre site du jeu de données ;
- son mécanisme ne parle pas de la cause ;
- un nombre cité ou un phénomène (mot d'un nom de thème) est absent des faits.

Après `WRITE_RETRIES` reprises, la mesure devient **« À définir »** : le texte donne le site, la cause, les
incidents et le coût par an, sans action inventée. Les fiches de cause (`cause_sheets`) gardent une
description et les conséquences typiques, sans mesure ; elles indiquent les sites qui ont une mesure
pour leur cause.

Le validateur de texte refuse aussi le mot « rang », les nombres absents des faits et les phénomènes
absents des faits (âge et environnement du site inclus).

## Relecture par le juge (`judge.py`)

Un passage par texte (caractérisation, causes dominantes, chaque mesure non « À définir », synthèse). Le juge
reçoit le texte, ses faits et une grille de questions, et renvoie les problèmes et une correction. Le code
accepte la correction seulement si elle passe le validateur de la rédaction. Chaque décision est consignée
dans le journal.

## Sorties (`outputs/2_analyse/`)

| Fichier | Contenu |
|---|---|
| `rapport_analyse.md` | le rapport : synthèse, méthode, thèmes, causes, heatmap, fiches, sites, plan d'actions, causes isolées, arrêts nominaux, limites |
| `journal.md` / `journal.json` | déroulé complet, phase par phase |
| `decisions.json` | clusterings choisis, rapport (`sites`, `mesures`, `couverture`, `fiches`, `synthese`, `juge`), arrêts nominaux, causes isolées, composants normalisés, essais |
| `clustering_essais.csv` | scores de chaque essai (les reprises sont marquées) |
| `themes.csv` | thèmes de causes et de conséquences |
| `phrases.csv` | chaque phrase de cause ou de conséquence, avec son thème (vide = isolée) |
| `mesures.csv` | plan d'actions : une ligne par couple (site, cause) |
| `fiches_causes.csv` | fiche et criticité Pareto de chaque thème de cause |
| `heatmap_<segment>.csv/.png` | heatmap causes × conséquences (coût, incidents) |
| `incidents_analyse.csv` | chaque incident : cause, conséquence, thèmes, composant, arrêt nominal |
