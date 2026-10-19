# Pipeline agentique de segmentation

Étape `segmentation` de `main.py`. Elle regroupe les 1800 incidents en **voies de remédiation** et
cherche, pour chacune, les lieux où agir et les signaux précurseurs.

## Objectif

Un cluster fusionné = **une voie de remédiation distincte** : une action concrète qui prévient
un ensemble de modes de défaillance. Deux niveaux coexistent :

- **cluster fin** (`fine_cluster_id`) : une cause, issue de HDBSCAN sur les causes extraites ;
- **cluster fusionné** (`merged_cluster_id`) : une remédiation, issue de l'agent de fusion.

Les sites, équipements et périodes ne servent **qu'après** le clustering (profil de concentration).
Le clustering repose uniquement sur les embeddings de la **cause extraite** de chaque rapport, pas du
rapport entier : un rapport décrit aussi les dégâts, et regroupé tel quel il sépare une même cause en
plusieurs clusters selon sa gravité (sur les données réelles, 85 % des clusters solaires et 83 % des
clusters éoliens ne contenaient qu'un niveau de sévérité ; 15 % et 50 % en regroupant les causes).

## Étapes

Le traitement se fait par partition `asset_type` (`PARTITION_COL`). Les balayages, les fusions
et les escalades restent à l'intérieur d'une partition.

0. **Extraction de la cause** (`extraction.py`) : un appel LLM par incident, qui ne voit que la
   description (ni site, ni identifiants, ni sévérité ou coût). Il renvoie la cause racine, un extrait
   recopié mot pour mot (`indice`), la conséquence et le composant, avec les mots du rapport et sans
   liste fermée. Le code cherche l'extrait dans la description (`QUOTE_MIN_MATCH`, accents et
   ponctuation ignorés) ; si l'extrait est absent ou si le rapport ne dit pas la cause, l'incident va
   dans le groupe `cause_non_precisee` (`UNKNOWN_CAUSE_LABEL`) : jamais clusterisé, ni diagnostiqué, ni
   compté dans la concentration et les précurseurs, mais gardé dans la criticité.
   `--extractions <csv>` reprend des extractions déjà faites (par exemple un `incidents_segmented.csv`
   précédent) pour les incidents au texte identique ; leur extrait est revérifié de la même façon.
1. **Embeddings et réduction** : embeddings de la cause extraite, puis UMAP (`UMAP_PARAMS`).
2. **Balayage HDBSCAN** par `min_cluster_size` dans `MCS_SWEEP` (5 à 25). Sélection par DBCV :
   on retient le DBCV maximal parmi les réglages avec un taux d'isolés ≤ 35 % (`SWEEP_MAX_OUTLIER_RATE`)
   et au moins 2 clusters (`SWEEP_MIN_CLUSTERS`). À égalité, on prend le plus fin (plus petit
   `min_cluster_size`). Si aucun réglage n'est admissible, on prend le DBCV maximal global.
3. **Étiquetage LLM des clusters fins** : un titre et un mécanisme par cluster, à partir des causes
   et des conséquences représentatives.
4. **Reclassement LLM des isolés** : les incidents isolés (cause connue, mais hors des clusters denses) sont rattachés à un cluster fin de leur
   partition, ou restent `isole_atypique` (`OUTLIER_LABEL`). Ils sont ensuite exclus de la fusion,
   de la fouille de précurseurs et du profil de concentration, mais gardés dans la criticité.
5. **Agent de fusion** (`merging.py`) :
   - groupes candidats locaux : les `MERGE_NEIGHBOURS` centroïdes les plus proches et les paires de
     titres très proches (Jaccard ≥ 0,6), reliés en composantes connexes, découpées en lots de
     `MERGE_MAX_GROUP` ;
   - le LLM voit, pour chaque cluster, des causes extraites (pas des rapports entiers). Il ne fusionne
     que si **une seule action concrète** traite tous les membres ; il liste la
     conséquence évitée pour chaque membre. Il peut aussi signaler un cluster hétérogène
     (`flag_heterogeneous`) ;
   - garde-fou DBCV à trois paliers sur la chute relative : ≤ `MERGE_DBCV_MAX_DROP` (15 %) →
     acceptée ; jusqu'à `MERGE_DBCV_HARD_DROP` (35 %) → un appel **critique** doit nommer un
     composant et une action uniques couvrant chaque membre ; au-delà → veto ferme. Un ensemble de
     clusters fins rejeté n'est jamais reproposé, même reformulé (`vetoed`) ;
   - un signalement déclenche un nouveau balayage HDBSCAN sur ce seul cluster, qui est accepté s'il
     donne 2 à 4 sous-clusters avec un DBCV positif **et** si le LLM leur attribue des intitulés
     distincts (sinon HDBSCAN fragmente un mode homogène) ; pas de re-découpage récursif ;
   - au plus `MERGE_MAX_ROUNDS` = 3 tours. Toutes les décisions sont écrites dans `merge_log.json`.
6. **Profil de concentration** (`concentration.py`) : pour chaque cluster fusionné, part observée
   contre part attendue par site, équipement et trimestre (lift), test binomial unilatéral, puis
   correction de Benjamini-Hochberg au sein du cluster. Un résultat est **significatif** si
   q ≤ `CONCENTRATION_FDR` (0,10), lift ≥ `CONCENTRATION_MIN_LIFT` (1,5) et au moins
   `CONCENTRATION_MIN_OBS` (3) observations. Son niveau de preuve (`evidence_level`) est « forte » si
   q ≤ `CONCENTRATION_STRONG_Q` (0,05), « à vérifier » sinon : à confirmer sur place avant d'agir.
7. **Fouille de précurseurs** (`precursors.py`) :
   - paire (A, B) avec sev(A) ≤ sev(B) : un incident B a un précurseur si un incident A est survenu
     sur le **même équipement**, strictement avant B, dans une fenêtre de 7, 30 ou 60 jours
     (meilleure fenêtre par paire) ;
   - les paires A → A sont des **récurrences** : calculées et publiées, mais ni envoyées au LLM ni
     présentées comme règles d'alerte ;
   - support minimal : au moins 5 occurrences sur au moins 3 équipements ;
   - modèle nul : permutation des étiquettes **au sein de chaque équipement**, 10 000 permutations.
     `lift = observé / attendu`, `p = (1 + #{nul ≥ observé}) / (1 + N)` ;
   - correction de Benjamini-Hochberg (FDR 0,05) en un seul pool par type (précurseur / escalade
     versus récurrence). Un précurseur est significatif s'il est la meilleure fenêtre, q ≤ 0,05 et
     lift > 1 ;
   - **escalade** : même mécanique à l'intérieur de chaque cluster fusionné, sur les clusters fins
     qui le composent (une seule correction BH pour l'ensemble des clusters).
8. **Contrôle de plausibilité LLM** (`precursor_agent.py`) : chaque paire significative (précurseur
   ou escalade) reçoit un mécanisme physique et une règle de surveillance, ou est écartée
   (`plausible=False`) si la cause est commune et externe (par exemple la météo).
9. **Preuves par incident** : pour chaque règle plausible, liste des incidents précurseurs
   associés à chaque incident conséquent (`precursor_evidence.csv`).
10. **Diagnostic LLM final** (`diagnosis.py`) : un diagnostic par cluster fusionné, exécuté en
    dernier, avec les fiches des clusters fins, le journal de fusion, les concentrations
    significatives, les règles plausibles et les impacts. La validation est stricte :
    - les cibles d'une recommandation ne peuvent être que des valeurs de concentration significatives
      (sinon la cible est retirée, et une recommandation sans cible devient `fleet`) ; les cibles dont la
      concentration est « à vérifier » sont listées dans `targets_to_verify` ;
    - les règles d'alerte ne peuvent être que des règles validées ;
    - les identifiants inventés sont supprimés ;
    - la cause commune doit être justifiée par des citations (`common_cause_quotes`), cherchées dans les
      rapports du cluster ; sans citation trouvée, elle est remplacée par la cause extraite la plus
      représentative du plus gros cluster fin (`common_cause_source` = `extraction`, sinon `diagnostic`) ;
    - **fonctionnement nominal** : un appel séparé (`nominal_check`) montre au LLM les causes et
      conséquences représentatives et lui pose deux questions : un défaut a-t-il été constaté sur
      l'équipement (`defect_found`) ? la seule conséquence est-elle une protection ou une mise en
      sécurité déclenchée par une condition extérieure (`protection_only`) ? Le cluster est proposé
      nominal (`nominal_proposed`) si oui à la seconde et non à la première, et accepté (`nominal`)
      seulement si au moins `NOMINAL_MIN_MILD_SHARE` (80 %) de ses incidents sont au niveau de sévérité
      le plus bas. Un cluster nominal n'a pas de recommandation. Les prompts ne donnent aucun exemple de
      défaut ni de condition extérieure : les exemples tirés des résultats orienteraient la réponse.
11. **Criticité** (`criticality.py`) : classement Pareto des clusters fusionnés par coût et temps
    d'arrêt, isolés inclus.

## Littéraux du domaine et schéma d'entrée

- **Littéraux propres au domaine** : uniquement dans `config.py` : `SEVERITY_ORDER`
  (`mineure < modérée < sévère`), `FRENCH_STOPWORDS`, `PARTITION_COL`, `EQUIPMENT_COL`,
  `SEVERITY_COL`, `OUTLIER_LABEL`, `UNKNOWN_CAUSE_LABEL`. Aucun prompt ne nomme de cause, de composant ou
  de site : les causes viennent des rapports.
- **Schéma d'entrée fixe** (`REQUIRED_COLUMNS` dans `pipeline.py`) : `incident_id`, `text`, `date`,
  `equipment_id`, `site`, `asset_type`, `severity`, `estimated_cost_eur`, `downtime_hours`. Si une
  colonne manque, le pipeline s'arrête avec la liste des colonnes absentes. Il n'y a pas de
  détection automatique des colonnes.

## Sorties

Dossier `outputs/segmentation/` (schémas détaillés dans le [README racine](../README.md#lancer-la-pipeline)) :

| Fichier | Grain | Contenu principal |
|---|---|---|
| `incidents_segmented.csv` | incident | `fine_cluster_id`, `merged_cluster_id`, `cluster_id`, `problem_title`, `common_cause`, `cluster_criticality_tier`, `assignment_source` (`dense_cluster`, `llm_reassigned`, `unclassified_outlier`, `cause_unknown`), extraction : `cause_racine`, `cause_type`, `indice`, `consequence`, `composant_brut`, `cause_known`, `extraction_source` (`llm`, `import`) |
| `clusters_fine.csv` | cluster fin | `fine_cluster_id`, `partition`, `merged_cluster_id`, `size`, `fine_title`, `failure_mode`, `keywords` |
| `clusters_merged.csv` | cluster fusionné | `merged_cluster_id`, `fine_cluster_ids`, criticité, fréquence, coûts, durées, diagnostic, `common_cause_quotes`, `prevented_outcomes`, `recommendations`, `early_warning_rules` (listes JSON), `common_cause_source`, `nominal`, `nominal_proposed`, `nominal_reason` |
| `merge_log.json` | décision de fusion | `round`, `action`, `members`, `shared_remediation`, `prevented_outcomes`, `dbcv_before`, `dbcv_after`, `vetoed`, `critic_verdict` |
| `concentration.csv` | cluster × dimension × valeur | `dimension` (`site`, `equipment_id`, `period`), `observed`, `expected`, `lift`, `p_value`, `q_value`, `significant`, `evidence_level` (`forte`, `à vérifier`, vide si non significatif) |
| `precursor_pairs.csv` | paire × fenêtre | `kind` (`precursor`, `recurrence`, `escalation`), `antecedent`, `consequent`, `window_days`, `n_hits`, `n_equipments`, `lift`, `q_value`, `is_best_window`, `significant`, `plausible`, `mechanism`, `monitoring_rule` |
| `precursor_evidence.csv` | incident conséquent × règle | `rule_id`, `consequent_incident_id`, `equipment_id`, `antecedent_incident_ids`, `antecedent_dates`, `window_days` |
| `fine_sweep.csv` | partition × taille minimale | `min_cluster_size`, `n_clusters`, `outlier_rate`, `dbcv` |

Cache : chaque appel LLM est mis en cache dans `outputs/cache/agent/<cache_tag>/`, sous un hash
du prompt système et du prompt utilisateur.

## Limites

- **Chocs communs à la flotte.** La permutation au sein de chaque équipement ne neutralise pas
  les événements qui touchent toute la flotte en même temps (tempête, canicule). Le contrôle LLM
  écarte les causes communes externes évidentes, mais c'est une limite du modèle nul.
- **Puissance statistique faible.** Avec environ 33 incidents par équipement, un test non
  significatif signifie « non détectable avec ces données », et non « inexistant ».
- **Dépendance au LLM critique.** Entre 15 et 35 % de chute de DBCV, la qualité de la fusion
  repose sur le critique. Utilisé seul, il s'est montré trop conciliant (même modèle que
  l'agent de fusion), d'où le veto ferme au-delà de 35 %. Toutes les décisions sont tracées dans
  `merge_log.json` pour audit.
- **Rapports sans cause.** Les incidents dont le rapport ne dit pas pourquoi ils sont arrivés ne
  sont pas devinés : ils restent dans `cause_non_precisee`. Le seul remède est un meilleur rapport.
- **Cause et conséquence mêlées.** L'extraction peut encore glisser des mots de conséquence dans la
  cause ; c'est pourquoi le prompt la sépare explicitement et que l'étiquetage voit les deux.
- **Cache par hash.** Modifier un prompt invalide son cache : tous les appels concernés sont
  refaits au prochain lancement, avec un coût LLM.

## Validation

Tests dans `tests/` (sans réseau ni LLM, le LLM est simulé), lancés par
`.venv/bin/python -m pytest tests -q` :

| Fichier | Couvre |
|---|---|
| `test_clustering.py` | balayage HDBSCAN, règle de sélection et repli, DBCV, UMAP sur petits effectifs |
| `test_labeling.py` | fiches et titres des clusters fins, réponses LLM mal formées, réaffectation des isolés |
| `test_merging.py` | fusion de doublons, veto DBCV avec critique négatif, découpage d'un cluster bimodal, limite de 3 tours, identifiants inconnus ignorés |
| `test_concentration.py` | concentration significative, cluster uniforme sans résultat, isolés exclus |
| `test_precursors.py` | **injection synthétique** : un précurseur planté dans des données aléatoires doit ressortir significatif, un cluster sans structure (ou une machine « citron ») ne doit pas l'être ; règle de sévérité, récurrences séparées, nombre d'occurrences identique à un calcul naïf, durée d'exécution |
| `test_precursor_agent.py` | contrôle de plausibilité : réponses valides, mal formées, exceptions, seules les paires significatives sont envoyées |
| `test_extraction.py` | extraction : citation vérifiée (accents et ponctuation tolérés, invention refusée), cause inconnue, échec du LLM, le LLM ne voit ni site ni sévérité, reprise seulement si le texte est identique |
| `test_diagnosis.py` | diagnostic validé : cibles et règles inventées supprimées, périmètre des prompts borné, cause commune remplacée sans citation vérifiée, nominal refusé si le cluster n'est pas surtout de gravité minimale |
| `test_pipeline_smoke.py` | exécution complète sur un échantillon de 200 incidents avec LLM et embeddings simulés : tous les fichiers de sortie existent avec leurs colonnes ; les rapports sans cause forment leur propre groupe ; `clusters_summary.csv` n'est pas produit ; colonnes obligatoires manquantes → erreur explicite |

Le test d'injection (`test_precursors.py`) est la vérification de référence de la fouille de
précurseurs : il contrôle à la fois la détection d'un signal réel et l'absence de faux positifs
sur des données sans structure.
