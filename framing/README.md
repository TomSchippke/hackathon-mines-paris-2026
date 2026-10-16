# Étape 1b : cadrage

Les agents ne savent **rien du problème a priori**. Ce qu'ils en savent vient de deux sources :

1. **`context/contexte.md`** : quelques lignes écrites par l'utilisateur (qui est le client, quels
   sites, ce qu'on sait d'eux) ;
2. **un échantillon des données** : valeurs des colonnes, sites de chaque segment, descriptions
   d'incidents tirées au hasard (graine fixe).

```bash
python main.py --steps cadrage       # ou : python -m framing
```

Le LLM en déduit **trois éléments** (JSON imposé), que le code vérifie contre les données :

| Élément | Sert à | Vérifié par le code |
|---|---|---|
| `domaine` | une phrase de contexte dans les consignes des agents | — |
| `segments` | libellé de chaque valeur de `asset_type` (rapport, webapp) | chaque valeur des données a un libellé (sinon la valeur brute est utilisée, avec un avertissement) |
| `severites` | ordre de gravité, de la moins grave à la plus grave | exactement les valeurs de `severity` des données, chacune une fois (sinon erreur) |

Les causes et conséquences ne sont **pas** fixées par le cadrage : elles sont découvertes dans les
données par l'analyse.

Une erreur fait redemander le cadrage au LLM (avec la liste des erreurs) ; un avertissement est
corrigé par le code et affiché.

**Relire le cadrage avant l'analyse** : `outputs/1b_cadrage/cadrage.md`. Pour le corriger, copier
`outputs/1b_cadrage/cadrage.json` en **`context/cadrage.json`** et l'éditer : ce fichier remplace la
proposition du LLM (il est vérifié de la même façon). Changer `contexte.md` ou le cadrage peut
changer les segments et sévérités utilisés par les étapes suivantes.
