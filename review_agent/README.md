# Étape 3 : agent vérificateur

Relit le travail de l'agent d'analyse (`outputs/2_analyse/`) pour s'assurer qu'il est cohérent et
que le rapport est fiable avant de le présenter. Il est **en lecture seule** : il ne modifie rien.

```bash
python main.py --steps verification  # ou : python -m review_agent
```

## Deux niveaux de contrôle

**1. Contrôles automatiques (code)**, faits avant l'agent et donnés à l'agent dès le départ :

| Contrôle | Échec si |
|---|---|
| rapport produit | l'agent d'analyse n'a pas terminé |
| budget de clustering | plus de 5 essais pour un type et une dimension (causes, conséquences) |
| qualité du clustering choisi | stabilité < 0.8, non classés > 15 %, clusters hors 3–40 (pour les causes et pour les conséquences) |
| chaque cluster dans un thème | un incident classé n'a pas de thème de cause ou de conséquence |
| causes et conséquences extraites | moins de 95 % des incidents analysés par le LLM |
| causes non précisées | (information) part des rapports qui ne donnent pas la cause : limite des données, doit être mentionnée dans le rapport |
| profils A, B, C | un profil absent ou attribué à deux sites du même type |
| plan de remédiation | un site sans 2 à 8 actions |
| refus d'outils | (information) nombre de refus dans le journal |

**2. Relecture par un agent LLM**, avec ces outils :

| Outil | Ce qu'il fait |
|---|---|
| `read_report` | le rapport complet |
| `read_journal(start, count)` | le journal de l'agent d'analyse, par pages de 20 événements maximum |
| `site_facts(site)` | chiffres **recalculés** du site à côté de **ce que le rapport affirme** (profil, causes, actions) |
| `theme_facts(theme_id)` | contenu réel d'un thème (sites, facteurs, exemples), pour vérifier son nom |
| `record_finding(gravite, element, constat, preuve)` | consigne un point vérifié : `conforme`, ou un problème `bloquant` / `majeur` / `mineur`, avec sa preuve |
| `conclude(verdict, synthese)` | **outil final** : `valide`, `valide_avec_reserves` ou `a_refaire` |

Garde-fous du code : impossible de conclure sans avoir lu le rapport ; verdict `valide` refusé
s'il reste un contrôle en échec ou un constat bloquant ou majeur ; `valide_avec_reserves` refusé
s'il reste un contrôle en échec ou un constat bloquant.

Ce que l'agent vérifie : chiffres du texte présents dans les données, profils justifiés par des
faits (et pas une autre attribution plus cohérente), causes dominantes réellement les plus
fréquentes, actions qui répondent aux causes du site, choix du clustering justifié par les
scores, incohérences dans le journal.

## Sorties (`outputs/3_verification/`)

| Fichier | Contenu |
|---|---|
| `verification.md` | verdict, synthèse, contrôles automatiques, constats triés par gravité |
| `verification.json` | la même chose, structurée |
| `journal.md` | déroulé de la session du vérificateur |

Si le verdict est `a_refaire`, lire les constats, ajuster (prompt ou bornes dans
`analysis_agent/config.py`) puis relancer `python main.py --steps analyse verification`.
