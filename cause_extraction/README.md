# Causes de chaque incident

Le topic modeling regroupe les incidents par **type d'événement** (incendie, surtension...),
parce qu'une description mélange ce qui s'est passé, pourquoi, et ce qui a été fait. Or le sujet
demande les **causes**. Cette étape fait lire chaque rapport au LLM et lui demande une analyse
structurée : une cause racine, des causes contributives, et le reste.

```bash
python -m cause_extraction          # lancement seul, pour explorer
```

## Ce que renvoie le LLM pour chaque incident

| Champ | Contenu |
|---|---|
| `cause_racine` | **pourquoi**, en une phrase, en remontant au phénomène qui a déclenché l'enchaînement (pas au symptôme final) |
| `cause_type` | `explicite` (écrite dans la description), `inférée` (désignée par un indice concret de la description ; la météo du jour seule ne suffit pas) ou `inconnue` (sinon, et en cas de doute) |
| `indice` | extrait de la description **recopié mot pour mot** qui justifie la cause racine. **Vérifié par le code** : s'il ne figure pas dans le rapport (à 80 % près, sans tenir compte des accents), la cause est annulée et devient `inconnue` (la proposition du LLM est gardée dans `cause_annulee`) |
| `causes_contributives` | 0 à 4 **conditions** qui ont permis l'incident ou l'ont aggravé (sans être le déclencheur). Chacune : `texte`, `type` (`explicite` ou `inférée`), `indice`. **Vérifiées une à une** : un extrait absent du rapport écarte la condition |
| `consequence` | **ce que l'incident a provoqué** (dégât, arrêt, risque), sans la cause ni durée ni montant |
| `composant_brut` | le composant touché, avec les mots du rapport (non normalisé à ce stade) |
| `maintenance_suspectee`, `justification_maintenance` | hypothèse : un défaut de maintenance a-t-il pu laisser l'incident se produire, **d'après la conséquence** et l'état constaté (non / plausible / probable) ? Champ séparé : ne modifie jamais `cause_racine` ni `cause_type` |
| `action_corrective` | ce qui éviterait que ça se reproduise (base du plan de remédiation) |
| `confiance` | faible / moyenne / haute |

Règle de l'extrait, commune à la cause racine et aux causes contributives : **pas d'indice vérifié,
pas de cause**. Le code compte les causes contributives écartées (`contributives_rejetees`), ce qui
mesure la part des propositions du LLM qui ne reposent pas sur le rapport.

Choix de conception :
- **Pas de liste fermée** : causes, composants et conséquences sont écrits avec les mots du rapport.
  Les thèmes seront découverts ensuite à partir de ces textes.
- **Format JSON imposé** (*structured outputs*) : la réponse est toujours analysable.
- **Le site n'est pas donné au LLM** (ni les identifiants d'équipement et de technicien, qui le contiennent) : l'analyse ne doit pas être influencée par ce qu'on sait du
  site, sinon comparer ensuite les sites entre eux serait circulaire.
- **Température 0 et cache** des réponses (`outputs/cache/llm/`) : 1800 appels la première fois,
  aucun ensuite, résultats identiques d'un lancement à l'autre. Changer le prompt relance les appels.

Dans la pipeline, c'est l'agent d'analyse qui appelle cette brique (outil `extract_causes`) ;
le cache des réponses est partagé, donc rien n'est jamais analysé deux fois.

## Sorties du lancement seul (`outputs/exploration/causes/`)

| Fichier | Contenu |
|---|---|
| `incidents_causes.csv` | chaque incident avec son analyse (`erreur_analyse` non vide si l'appel a échoué) |
| `contributives_par_site.csv` | par site : nombre moyen de causes contributives vérifiées, % écartées, % de maintenance suspectée probable, coût et durée moyens |
