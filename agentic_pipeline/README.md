# Pipeline agentique de segmentation

Étape `segmentation` de `main.py`. Elle regroupe les 1800 incidents en **voies de remédiation** et
cherche, pour chacune, les lieux où agir, les machines dont la sévérité monte et les signaux précurseurs.

## Objectif

Un cluster fusionné = **une voie de remédiation distincte** : une action concrète qui prévient
un ensemble de modes de défaillance. Deux niveaux coexistent :

- **cluster fin** (`fine_cluster_id`) : un mode de défaillance, issu de HDBSCAN ;
- **cluster fusionné** (`merged_cluster_id`) : une remédiation, issue de l'agent de fusion.

Les sites, équipements et périodes ne servent **qu'après** le clustering (profil de concentration).
Le clustering repose uniquement sur les embeddings du texte complet de chaque rapport, sans passe LLM
préalable. Limite connue : le rapport décrit aussi les dégâts, si bien qu'une même cause peut être
répartie en plusieurs clusters fins selon sa gravité ; l'agent de fusion les rassemble quand une même
correction les traite. Une extraction LLM de la cause a été essayée puis retirée : elle laissait 19 %
des incidents sans cause (rapports qui décrivent le constat sans dire pourquoi).

## Étapes

Le traitement se fait par partition `asset_type` (`PARTITION_COL`). Les balayages et les fusions
restent à l'intérieur d'une partition.

1. **Embeddings et réduction** : embeddings du texte, puis UMAP (`UMAP_PARAMS`).
2. **Balayage HDBSCAN** par `min_cluster_size` dans `MCS_SWEEP` (5 à 25). Sélection par DBCV :
   on retient le DBCV maximal parmi les réglages avec un taux d'isolés ≤ 35 % (`SWEEP_MAX_OUTLIER_RATE`)
   et au moins 2 clusters (`SWEEP_MIN_CLUSTERS`). À égalité, on prend le plus fin (plus petit
   `min_cluster_size`). Si aucun réglage n'est admissible, on prend le DBCV maximal global.
3. **Isolés = clusters singletons** : chaque incident isolé par HDBSCAN devient un cluster fin d'un seul
   incident (`{partition}_O{k}`). Il n'y a plus de reclassement LLM des isolés : le singleton suit
   le même chemin que les autres clusters fins (étiquetage, fusion). Le DBCV ignore les singletons.
4. **Étiquetage LLM des clusters fins**, singletons compris : un titre et un mode de défaillance par
   cluster, à partir des rapports représentatifs et des mots-clés (la répartition des sévérités n'est
   pas montrée).
5. **Agent de fusion** (`merging.py`) :
   - groupes candidats locaux : les `MERGE_NEIGHBOURS` centroïdes les plus proches et les paires de
     titres très proches (Jaccard ≥ 0,6), reliés en composantes connexes, découpées en lots de
     `MERGE_MAX_GROUP` ;
   - le LLM ne fusionne que si **une seule action concrète** traite tous les membres ; il liste la
     conséquence évitée pour chaque membre. Il peut aussi signaler un cluster hétérogène
     (`flag_heterogeneous`, ignoré sur un singleton) ;
   - garde-fou DBCV à trois paliers sur la chute relative : ≤ `MERGE_DBCV_MAX_DROP` (15 %) →
     acceptée ; jusqu'à `MERGE_DBCV_HARD_DROP` (35 %) → un appel **critique** doit nommer un
     composant et une action uniques couvrant chaque membre ; au-delà → veto ferme. Un ensemble de
     clusters fins rejeté n'est jamais reproposé, même reformulé (`vetoed`) ;
   - une fusion qui ne réunit pas au moins deux clusters de plus d'un incident (elle n'absorbe que des
     singletons) n'a pas de DBCV exploitable : elle est décidée par le seul LLM de fusion ;
   - un singleton jamais fusionné reçoit `merged_cluster_id` = `isole_atypique` (`OUTLIER_LABEL`) ;
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
   Les isolés sont exclus.
7. **Veille d'escalade** (`escalation.py`) : une série = tous les incidents d'un même équipement dans un
   même cluster fusionné, sur toute la période (pas de fenêtre). Elle est retenue si elle compte au
   moins `ESCALATION_MIN_INCIDENTS` (3) incidents et si, triée par date, la sévérité ne décroît jamais
   et termine strictement plus haut qu'au départ. Les séries sont classées par coût total décroissant
   (`escalations.csv`). La liste est **descriptive, sans test de significativité** : avec quelques
   incidents par série, un test par série ne peut jamais survivre à la correction pour tests multiples
   (6 incidents sur 3 niveaux donnent au plus 90 ordres distinguables, donc p ≥ ~0,011). Elle
   complète la concentration (où) par « ça empire sur la même machine ». Aucun LLM, résultat déterministe.
8. **Fouille de précurseurs** (`precursors.py`), au niveau des clusters fusionnés :
   - paire (A, B) avec sev(A) ≤ sev(B) : un incident B a un précurseur si un incident A est survenu
     sur le **même équipement**, strictement avant B, dans une fenêtre de 7, 30 ou 60 jours
     (meilleure fenêtre par paire) ;
   - les paires A → A sont des **récurrences** : calculées et publiées, mais ni envoyées au LLM ni
     présentées comme règles d'alerte ;
   - support minimal : au moins 5 occurrences sur au moins 3 équipements ;
   - modèle nul : permutation des étiquettes **au sein de chaque équipement**, 10 000 permutations.
     `lift = observé / attendu`, `p = (1 + #{nul ≥ observé}) / (1 + N)` ;
   - correction de Benjamini-Hochberg (FDR 0,05) en un seul pool par type (précurseur versus
     récurrence). Un précurseur est significatif s'il est la meilleure fenêtre, q ≤ 0,05 et lift > 1.
   L'ancienne « escalade » au niveau des clusters fins est remplacée par la veille d'escalade (étape 7).
9. **Contrôle de plausibilité LLM** (`precursor_agent.py`) : chaque paire significative reçoit un
   mécanisme physique et une règle de surveillance, ou est écartée (`plausible=False`) si la cause est
   commune et externe (par exemple la météo).
10. **Preuves par incident** : pour chaque règle plausible, liste des incidents précurseurs
    associés à chaque incident conséquent (`precursor_evidence.csv`).
11. **Diagnostic LLM final** (`diagnosis.py`) : un diagnostic par cluster fusionné, exécuté en
    dernier, avec les fiches des clusters fins, le journal de fusion, les concentrations
    significatives, les séries en escalade, les règles plausibles et les impacts. Chaque verrou est
    appliqué par le code :
    - **cibles** : une recommandation ne peut cibler que des valeurs de concentration significatives
      ou, pour la portée `equipment`, des équipements en escalade dans ce cluster (sinon la cible est
      retirée, et une recommandation sans cible devient `fleet`). `target_reason` indique l'origine de
      chaque cible (`concentration`, `escalation`, `both`) ; les cibles dont la concentration est
      « à vérifier » sont listées dans `targets_to_verify` ;
    - les règles d'alerte ne peuvent être que des règles validées ; les identifiants inventés sont supprimés ;
    - **cause commune** : justifiée par des citations (`common_cause_quotes`) cherchées dans les rapports
      du cluster ; les citations introuvables sont écartées. Sans citation vérifiée, un second appel
      reçoit jusqu'à 30 rapports du cluster (`MAX_RETRY_TEXTS`) ; s'il n'en apporte pas non plus, la
      cause est marquée `inferee` (sinon `citee`) et les recommandations ciblées (portée `site` ou
      `equipment`) reçoivent `confirm_cause_first` : confirmer la cause sur place avant d'agir ;
    - **fonctionnement nominal** : un appel séparé (`nominal_check`) montre au LLM les rapports
      représentatifs et lui demande si un défaut a été constaté (`defect_found`), si la seule
      conséquence est une protection ou une mise en sécurité (`protection_only`), et, dans ce cas, la
      condition extérieure qui l'a déclenchée avec une citation. Le cluster est proposé nominal
      (`nominal_proposed`) si `protection_only` est vrai, `defect_found` faux, et si la condition
      extérieure est citée dans un rapport du cluster (`nominal_external_condition`,
      `nominal_external_quote` ; citation vérifiée par le code). Il est accepté (`nominal`) seulement si
      au moins `NOMINAL_MIN_MILD_SHARE` (80 %) de ses incidents sont au niveau de sévérité le plus bas.
      Un cluster nominal n'a pas de recommandation. Les prompts ne donnent aucun exemple de défaut ni de
      condition extérieure : les exemples tirés des résultats orienteraient la réponse ;
    - **critique des recommandations** (clusters non nominaux) : un appel relit chaque recommandation
      (`keep`, `rewrite` ou `drop`) et doit nommer un composant et une action. Le critique ne peut pas
      changer la portée ni les cibles. Une recommandation sans composant ou sans action est écartée ;
      si toutes le sont, la recommandation de repli est « Aucune action spécifique identifiée :
      inspecter pour établir la cause. ». En cas d'erreur de l'API, les recommandations sont gardées avec
      `critic_verdict` = `non_verifie` ;
    - **ordre** des recommandations : escalade, puis ciblées (site, équipement), puis flotte.
12. **Criticité** (`criticality.py`) : classement des clusters fusionnés par **coût total décroissant**,
    isolés inclus. Les paliers sont Pareto sur le coût total cumulé : Critique jusqu'à 70 %, Élevée
    jusqu'à 90 %, Moyenne jusqu'à 98 %, Faible au-delà (le premier cluster est toujours Critique).
    Fréquence, coût moyen et durées d'arrêt sont fournis à titre informatif.

## Littéraux du domaine et schéma d'entrée

- **Littéraux propres au domaine** : uniquement dans `config.py` : `SEVERITY_ORDER`
  (`mineure < modérée < sévère`), `FRENCH_STOPWORDS`, `PARTITION_COL`, `EQUIPMENT_COL`,
  `SEVERITY_COL`, `OUTLIER_LABEL`. Aucun prompt ne nomme de cause, de composant ou de site, ni ne donne
  d'exemple tiré des résultats.
- **Schéma d'entrée fixe** (`REQUIRED_COLUMNS` dans `pipeline.py`) : `incident_id`, `text`, `date`,
  `equipment_id`, `site`, `asset_type`, `severity`, `estimated_cost_eur`, `downtime_hours`. Si une
  colonne manque, le pipeline s'arrête avec la liste des colonnes absentes. Il n'y a pas de
  détection automatique des colonnes.

## Sorties

Dossier `outputs/segmentation/` (schémas détaillés dans le [README racine](../README.md#lancer-la-pipeline)) :

| Fichier | Grain | Contenu principal |
|---|---|---|
| `incidents_segmented.csv` | incident | `fine_cluster_id`, `merged_cluster_id`, `cluster_id`, `problem_title`, `common_cause`, `cluster_criticality_tier`, `assignment_source` (`dense_cluster`, `merged_singleton`, `unclassified_outlier`) |
| `clusters_fine.csv` | cluster fin (singletons non fusionnés inclus, `merged_cluster_id` = `isole_atypique`) | `fine_cluster_id`, `partition`, `merged_cluster_id`, `size`, `fine_title`, `failure_mode`, `keywords` |
| `clusters_merged.csv` | cluster fusionné | `merged_cluster_id`, `fine_cluster_ids`, criticité, fréquence, coûts, durées, diagnostic, `common_cause_quotes`, `prevented_outcomes`, `recommendations`, `early_warning_rules` (listes JSON), `common_cause_source` (`citee`, `inferee`), `nominal`, `nominal_proposed`, `nominal_reason`, `nominal_external_condition`, `nominal_external_quote` ; chaque recommandation porte `scope`, `targets`, `target_reason`, `targets_to_verify`, `component`, `confirm_cause_first`, `critic_verdict` |
| `merge_log.json` | décision de fusion | `round`, `action`, `members`, `shared_remediation`, `prevented_outcomes`, `dbcv_before`, `dbcv_after`, `vetoed`, `critic_verdict` |
| `concentration.csv` | cluster × dimension × valeur | `dimension` (`site`, `equipment_id`, `period`), `observed`, `expected`, `lift`, `p_value`, `q_value`, `significant`, `evidence_level` (`forte`, `à vérifier`, vide si non significatif) |
| `precursor_pairs.csv` | paire × fenêtre | `kind` (`precursor`, `recurrence`), `antecedent`, `consequent`, `window_days`, `n_hits`, `n_equipments`, `lift`, `q_value`, `is_best_window`, `significant`, `plausible`, `mechanism`, `monitoring_rule` |
| `escalations.csv` | série (équipement × cluster fusionné) | `equipment_id`, `merged_cluster_id`, `partition`, `n_incidents`, `severity_sequence`, `incident_ids`, `dates`, `costs`, `total_cost_eur`, `first_date`, `last_date` ; trié par coût total |
| `precursor_evidence.csv` | incident conséquent × règle | `rule_id`, `consequent_incident_id`, `equipment_id`, `antecedent_incident_ids`, `antecedent_dates`, `window_days` |
| `fine_sweep.csv` | partition × taille minimale | `min_cluster_size`, `n_clusters`, `outlier_rate`, `dbcv` |

Diagnostic hors pipeline : `python -m scripts.umap_stability` écrit `outputs/diagnostics/umap_stability.csv`
(stabilité des clusters fins selon la graine UMAP ; embeddings déjà en cache, aucun appel à l'API).

Cache : chaque appel LLM est mis en cache dans `outputs/cache/agent/<cache_tag>/`, sous un hash
du prompt système et du prompt utilisateur.

## Limites

Chiffres mesurés sur la dernière exécution (sorties actuelles).

- **Garde-fou DBCV de la fusion.** 27 fusions proposées, 23 acceptées. 21 l'ont été sous le seuil de
  chute de 15 %, sans contrôle sémantique ; 1 est une absorption de singletons (sans DBCV de départ,
  donc sans garde-fou) ; 1 est passée par le critique (chute de 22 %). 4 fusions ont été refusées : 2 par
  veto ferme (> 35 %, chutes de 41 % et 70 %) et 2 par le critique (chutes de 31 % et 35 %), sur 3
  appels au critique au total (1 accord, 2 refus). Les 2 vetos fermes portaient sur des remédiations
  vagues ou multi-composants (« refroidissement et surveillance thermique accrue des composants
  électriques et panneaux », « vérification et calibration du système de sécurité et de protection de
  vitesse »). Tension assumée : le niveau fusionné regroupe des clusters qui diffèrent dans l'espace du
  texte, ce qui abaisse mécaniquement le DBCV ; le garde-fou accepte de perdre quelques fusions valides
  pour se protéger d'un LLM trop conciliant. Côté découpage, 36 signalements d'hétérogénéité dont 6
  découpages acceptés ; dans chacun, des sous-clusters ont ensuite été refusionnés parce qu'ils
  partagent une même remédiation : un découpage sépare des modes de défaillance, une fusion regroupe des
  correctifs, ce qui est cohérent avec la conception. Les décisions sont toutes dans `merge_log.json`.
- **Concentration : BH au sein de chaque cluster.** Un cluster fusionné correspond à une remédiation,
  donc à une décision de ciblage : la correction est faite à cet échelon, volontairement. Elle ne
  contrôle pas le FDR sur l'ensemble du plan : 24 concentrations significatives avec un BH par
  cluster, 17 avec un BH global sur 950 tests (les 17 sont de niveau « forte »). 7 résultats faibles sont
  marqués « à vérifier ».
- **Précurseurs : résultat négatif.** 0 paire significative sur 212 tests de précurseurs ; 13 ont
  p < 0,05, pour 10,6 attendues par hasard (212 × 0,05). Sur les récurrences, 28 testées, 1 significative. Le test d'injection montre que la méthode
  détecte un signal planté : avec environ 33 incidents par équipement, aucune règle d'alerte précoce
  fiable n'existe dans ces données. Un test non significatif signifie « non détectable ici », pas
  « inexistant ». La permutation au sein de chaque équipement ne neutralise pas non plus les chocs
  communs à la flotte (tempête, canicule) ; le contrôle LLM n'écarte que les causes externes évidentes.
- **Escalade : descriptive.** La veille d'escalade ne teste rien (voir étape 7) : une série qui monte
  n'est pas une preuve, c'est une liste de machines à regarder, classée par coût. Sur ce run : 20 séries
  (3 à 6 incidents ; 7 de 3, 7 de 4, 5 de 5, 1 de 6) sur 17 équipements. 5 recommandations visent une
  cible d'escalade (4 « escalade », 1 « les deux »), dont OUEST1-INV07.
- **UMAP et DBCV.** Le DBCV (≈ 0,95 solaire, ≈ 0,91 éolien) est mesuré dans l'espace UMAP
  (`min_dist` = 0) et ne sert qu'à comparer des réglages. Stabilité sur 5 graines UMAP à
  `min_cluster_size` fixé : ARI par paires moyen / minimal de 0,970 / 0,951 (solaire) et
  0,984 / 0,966 (éolien) ; DBCV de 0,88 à 0,96 selon la graine. Point faible : le `min_cluster_size`
  retenu par le balayage varie avec la graine (5 à 10 en solaire, 7 à 20 en éolien), car les DBCV des
  réglages sont très proches.
- **Même petit modèle partout.** Tous les appels (agents et critiques) passent par le même petit modèle
  (gpt-4.1-nano, défini dans `.env`). Un critique qui relit la production du même modèle peut être
  conciliant, d'où les contrôles côté code : citations vérifiées dans les rapports, cibles restreintes,
  composant et action exigés.
- **Comportements observés du petit modèle sur ce run.** Acceptés et documentés ; le rapport affiche
  chaque cluster nominal avec sa condition extérieure et sa citation pour qu'un lecteur puisse les
  contester.
  - *Faux négatif nominal.* Le cluster des arrêts automatiques de sécurité par vent excessif (`wind_M0`,
    261 incidents) n'est pas nominal : le modèle a répondu `defect_found: true` alors que sa propre
    `reason` indique qu'aucun dommage ni défaillance n'est mentionné, et sa citation recolle deux passages
    avec « ... », ce que la vérification de citation rejette.
  - *Acceptation nominale à la limite.* Le cluster de la protection survitesse (`wind_M4`, 49 incidents)
    est accepté avec la condition extérieure « augmentation inhabituelle de la vitesse de rotation »,
    qui est un symptôme machine plutôt qu'une condition extérieure ; la citation étant authentique, le
    contrôle la laisse passer. Au total 2 clusters nominaux sur 29 (l'autre : `solar_M3`, « tempête ou
    grêle »).
  - *Actions génériques.* Une action générique passe le critique dès qu'elle nomme un composant, par
    exemple sur des cibles d'escalade : « effectuer une inspection approfondie et une maintenance
    préventive » (SUD1-WTG04), « surveiller et intervenir en cas d'anomalies » (NORD2-WTG02). Sur 36
    recommandations, 29 sont réécrites par le critique, 6 conservées, 1 en repli.
  - *Cause commune.* Les 29 clusters (hors `isole_atypique`) ont une cause citée, aucune inférée.
- **Coût des singletons.** Chaque singleton coûte un appel LLM d'étiquetage (15 au taux d'isolés
  actuel, ~1 % : 3 rattachés à un cluster, 12 isolés) ; un réglage sélectionné avec un fort taux d'isolés en coûterait des centaines
  (plafond de sélection : 35 %).
- **Gravité dans le texte.** Les embeddings du rapport complet captent aussi la description des
  dégâts : des clusters fins d'une même cause peuvent se distinguer surtout par leur gravité.
- **Cache par hash.** Modifier un prompt invalide son cache : tous les appels concernés sont
  refaits au prochain lancement, avec un coût LLM.

## Validation

Tests dans `tests/` (sans réseau ni LLM, le LLM est simulé), lancés par
`.venv/bin/python -m pytest tests -q` :

| Fichier | Couvre |
|---|---|
| `test_clustering.py` | balayage HDBSCAN, règle de sélection et repli, DBCV (singletons ignorés), UMAP sur petits effectifs |
| `test_labeling.py` | fiches et titres des clusters fins (singletons compris), réponses LLM mal formées, sévérité absente du prompt d'étiquetage |
| `test_merging.py` | fusion de doublons, veto DBCV avec critique négatif, découpage d'un cluster bimodal, limite de 3 tours, identifiants inconnus ignorés, singletons (rattachement, absorption, isolé non fusionné, veto et critique maintenus, signalement ignoré) |
| `test_concentration.py` | concentration significative, cluster uniforme sans résultat, isolés exclus |
| `test_escalation.py` | veille d'escalade : série plantée retrouvée, cas exclus (sévérité qui baisse, moins de 3 incidents, isolés), pas de contamination entre séries, classement par coût, entrée vide, sévérité inconnue |
| `test_precursors.py` | **injection synthétique** : un précurseur planté dans des données aléatoires doit ressortir significatif, un cluster sans structure (ou une machine « citron ») ne doit pas l'être ; règle de sévérité, récurrences séparées (plus de type escalade), nombre d'occurrences identique à un calcul naïf, durée d'exécution |
| `test_precursor_agent.py` | contrôle de plausibilité : réponses valides, mal formées, exceptions, seules les paires significatives sont envoyées |
| `test_diagnosis.py` | diagnostic validé : cibles et règles inventées supprimées, périmètre des prompts borné, cibles d'escalade et `target_reason`, ordre escalade → ciblées → flotte, critique (réécriture sans changer portée ni cibles, repli si tout est écarté, `non_verifie` en cas d'erreur), citations vérifiées dans les rapports (accents et ponctuation tolérés), relance de la cause (30 rapports au plus) puis cause `inferee`, nominal exigeant condition extérieure citée et sévérité minimale |
| `test_equipment_metrics.py` | chronologie par équipement (écarts inter-pannes), agrégats, score « bad actor » fondé sur le seul coût et les seules heures d'arrêt (module `equipment_analysis`) |
| `test_pipeline_smoke.py` | exécution complète sur un échantillon de 200 incidents avec LLM et embeddings simulés : tous les fichiers de sortie existent avec leurs colonnes ; `clusters_summary.csv` n'est pas produit ; plus de reclassement des isolés ni de colonne `scope_cluster` ; `escalations.csv` présent ; colonnes obligatoires manquantes → erreur explicite |

Le test d'injection (`test_precursors.py`) est la vérification de référence de la fouille de
précurseurs : il contrôle à la fois la détection d'un signal réel et l'absence de faux positifs
sur des données sans structure.
