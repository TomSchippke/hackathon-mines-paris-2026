# Cause racine de chaque incident

Le topic modeling regroupe les incidents par **type d'événement** (incendie, surtension...),
parce qu'une description mélange ce qui s'est passé, pourquoi, et ce qui a été fait. Or le sujet
demande les **causes racines**. Cette étape fait lire chaque rapport au LLM et lui demande une
analyse structurée.

```bash
python -m cause_extraction          # lancement seul, pour explorer
```

## Ce que renvoie le LLM pour chaque incident

| Champ | Contenu |
|---|---|
| `composant` | composant touché, parmi une liste fermée (onduleur, pale, boîtier de jonction...) |
| `cause_racine` | **pourquoi**, en une phrase, en remontant à la cause première |
| `cause_type` | `explicite` (écrite dans la description), `inférée` (désignée par un indice concret de la description ; la météo du jour seule ne suffit pas) ou `inconnue` (sinon, et en cas de doute) |
| `indice` | extrait de la description **recopié mot pour mot** qui justifie la cause. **Vérifié par le code** : s'il ne figure pas dans le rapport (à 80 % près, sans tenir compte des accents), la cause est annulée et devient `inconnue` (la proposition du LLM est gardée dans `cause_annulee`) |
| `consequence` | **ce que l'incident a provoqué** (dégât, arrêt, risque), sans la cause ni durée ni montant |
| `categorie_consequence` | la même conséquence dans une liste fermée : perte de production, arrêt de sécurité automatique, déclenchement de protection / surtension, dégât matériel mineur, dégât majeur / destruction, incendie, aucun impact notable, autre |
| `facteur_principal` | parmi une liste fermée : corrosion saline, foudre, vent, grêle, chaleur, poussière, nuisibles, sol, usure, défaut de fabrication, maintenance, travaux, défaut électrique, indéterminé |
| `facteurs_secondaires` | autres facteurs ayant contribué |
| `lie_vieillissement`, `lie_maintenance` | l'âge ou la maintenance ont-ils joué un rôle ? |
| `action_corrective` | ce qui éviterait que ça se reproduise (base du plan de remédiation) |
| `confiance` | faible / moyenne / haute |

Choix de conception :
- **Listes fermées** (`config.py`) : on peut compter et comparer les sites. Les facteurs reprennent
  ce qui distingue les profils de site du sujet (environnement, climat, âge, maintenance).
- **Format JSON imposé** (*structured outputs*) : la réponse est toujours analysable.
- **Le site n'est pas donné au LLM** : l'analyse ne doit pas être influencée par ce qu'on sait du
  site, sinon associer ensuite un site à son profil serait circulaire.
- **Température 0 et cache** des réponses (`outputs/cache/llm/`) : 1800 appels la première fois,
  aucun ensuite, résultats identiques d'un lancement à l'autre. Changer le prompt relance les appels.

Dans la pipeline, c'est l'agent d'analyse qui appelle cette brique (outil `extract_causes`) ;
le cache des réponses est partagé, donc rien n'est jamais analysé deux fois.

## Sorties du lancement seul (`outputs/exploration/causes/`)

| Fichier | Contenu |
|---|---|
| `incidents_causes.csv` | chaque incident avec son analyse (`erreur_analyse` non vide si l'appel a échoué) |
| `facteurs_par_site.csv` | % des incidents de chaque site par facteur principal |
| `composants_par_site.csv` | % des incidents de chaque site par composant |
| `indicateurs_par_site.csv` | part liée à l'usure, à la maintenance, coût et durée moyens, part de confiance faible |

`facteurs_par_site.csv` et `indicateurs_par_site.csv` servent directement à associer chaque site
à son profil (A, B, C) : salin et tempêtes -> côtier, usure -> ancien, maintenance -> maintenance faible.
