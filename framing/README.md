# Étape 1b : cadrage

Les agents ne savent **rien du problème a priori**. Ce qu'ils en savent vient de deux sources :

1. **`context/contexte.md`** : quelques lignes écrites par l'utilisateur (qui est le client, quels
   sites, ce qu'on sait d'eux, l'objectif du rapport) ;
2. **un échantillon des données** : valeurs des colonnes, sites de chaque segment, descriptions
   d'incidents tirées au hasard (graine fixe).

```bash
python main.py --steps cadrage       # ou : python -m framing
```

Le LLM en déduit le **cadre d'analyse** (JSON imposé), que le code vérifie contre les données :

| Élément | Sert à | Vérifié par le code |
|---|---|---|
| `domaine` | une phrase de contexte dans les consignes des agents | — |
| `segments` | libellé de chaque valeur de `asset_type` (rapport, webapp) | chaque valeur des données a un libellé |
| `facteurs` (+ mots-clés, mesure préventive type) | liste fermée de l'extraction ; critères mesurés sur chaque site ; validateur de rédaction ; mesure de secours | 6 à 16 facteurs distincts, pas de nom réservé |
| `composants` | liste fermée de l'extraction | 4 à 16 |
| `categories_consequence` (+ `fonctionnement_normal`) | liste fermée de l'extraction ; arrêts de sécurité nominaux exclus des plans | 4 à 9, pas toutes « normales » |
| `severites` | ordre de gravité (la plus faible = arrêts nominaux, la plus forte = incidents graves) | exactement les valeurs des données |
| `profils` (facultatif) | attribution d'un profil à chaque site, si le contexte en décrit | autant de profils que de sites ; critères des attentes connus |

Une erreur fait redemander le cadrage au LLM (avec la liste des erreurs) ; un avertissement est
corrigé par le code et affiché (par exemple une attente de profil sur un critère inconnu, ignorée).

**Relire le cadrage avant l'analyse** : `outputs/1b_cadrage/cadrage.md`. Pour le corriger, copier
`outputs/1b_cadrage/cadrage.json` en **`context/cadrage.json`** et l'éditer : ce fichier remplace la
proposition du LLM (il est vérifié de la même façon). Changer `contexte.md` ou le cadrage change les
listes fermées, donc relance l'extraction des causes (les réponses du LLM sont en cache par requête).

## Profils de site (facultatif)

Sans profils dans le contexte, chaque site est **caractérisé** par ce qui le distingue des autres
sites de son segment. Pour faire associer chaque site à un profil, les décrire dans
`context/contexte.md`, par exemple :

```
Profils des centrales solaires (un par site, à retrouver dans les données) :
- A : site récent, zone sèche aux étés chauds, bonne maintenance.
- B : zone agricole, nuisibles modérés.
- C : site ancien et côtier (embruns, poussière agricole), maintenance faible.
```

Le LLM traduit chaque description en **attentes** mesurables (« chaleur : élevé », « usure : faible »)
sur les facteurs du cadrage ; l'agent d'analyse choisit ensuite l'attribution qui en respecte le plus.
