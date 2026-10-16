# Étape 3 : agent vérificateur

Contrôle la cohérence d'ensemble du travail de l'agent d'analyse (`outputs/2_analyse/`) avant de
présenter le rapport. Les textes eux-mêmes ont déjà été relus et corrigés par le juge de l'analyse
(`analysis_agent/judge.py`) : le vérificateur ne réécrit rien, il donne un avis et un verdict.

```bash
python main.py --steps verification  # ou : python -m review_agent
```

## Trois niveaux de contrôle

**1. Contrôles automatiques (code)**, faits avant l'agent :

| Contrôle | Échec si |
|---|---|
| rapport produit | l'agent d'analyse n'a pas terminé |
| budget et qualité des clusterings choisis | plus de 5 essais ; stabilité < 0.8, non classés > 15 %, clusters hors 3–40 |
| chaque cluster dans un thème | un incident classé n'a pas de thème |
| causes et conséquences extraites | moins de 95 % des incidents analysés |
| profils (si le contexte en décrit) | un profil absent ou attribué deux fois ; attribution qui respecte moins d'attentes des profils que la meilleure possible |
| plans : mesures préventives avec mécanisme | une mesure est une réparation ponctuelle, vague, ou sans mécanisme lié à sa cause |
| noms des thèmes de causes | un nom de cause contient une conséquence (« arrêt », « perte »...) |
| (information) | causes non précisées, reprises automatiques par le code, refus d'outils |

Les problèmes trouvés par ces contrôles deviennent des **constats ciblés** (source : code), avec
une consigne de correction.

**2. Relecture par un agent LLM, exigeant par consigne.** Il reçoit le rapport, les chiffres
recalculés de chaque site et les critères mesurés sur chaque site, et doit consigner au moins un point pour
**chaque site et pour la synthèse** (sinon `conclude` est refusé).

| Outil | Ce qu'il fait |
|---|---|
| `record_finding(gravite, cible_site, cible_champ, constat, preuve, consigne)` | un point vérifié : `conforme`, ou un problème `bloquant` / `majeur` / `mineur` ; cible = site (ou synthèse) et champ (justification_profil, causes_dominantes, plan, synthese) ; **consigne obligatoire** si bloquant ou majeur |
| `site_facts(site)`, `theme_facts(theme_id)`, `read_journal(start, count)` | chiffres recalculés, contenu réel d'un thème, journal de l'analyse |
| `conclude(verdict, synthese)` | **outil final** ; refusé tant qu'un site ou la synthèse n'a aucun point ; `valide` impossible avec un problème majeur ou bloquant ou un contrôle en échec. Si l'agent ne conclut pas, le verdict est calculé par ces mêmes règles (« verdict automatique ») |

**Pourquoi le vérificateur ne fait plus réécrire.** Une première version renvoyait ses constats au
rédacteur. Avec le petit modèle, ses consignes (« préciser que le profil est basé sur la zone sèche »)
ont poussé le rédacteur à recopier la description des profils au lieu des mesures. La correction des
textes est donc confiée au juge, dont chaque correction est validée par le code.

## Sorties (`outputs/3_verification/`)

| Fichier | Contenu |
|---|---|
| `verification.md` | verdict, contrôles automatiques, problèmes (cible, consigne), points conformes |
| `verification.json` | la même chose, structurée |
| `journal.md` | déroulé de la session du vérificateur |
