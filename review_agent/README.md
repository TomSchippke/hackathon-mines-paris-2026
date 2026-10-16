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
| extraction des causes et conséquences | moins de 95 % des incidents analysés |
| clustering : budget et qualité (essais de reprise exclus) | plus de 5 essais ; clustering choisi hors seuils (stabilité < 0.8, non classés > 15 %, clusters hors 3–40) |
| plan : couverture du coût | moins de 80 % du coût des causes connues couvert, sauf plafond de 12 mesures atteint (signalé) |
| plan : nombre de mesures | plus de 12 mesures |
| arrêts nominaux | un thème nominal dont moins de 80 % des incidents sont à la sévérité la plus faible (part recalculée) |
| chaînes cause -> conséquence prioritaires | aucune chaîne choisie |
| noms des thèmes de causes | un nom de cause contient une conséquence (« arrêt », « perte »...) |
| (information) phrases rattachées à un thème | part des phrases de cause et de conséquence par segment |
| (information) causes isolées | part des phrases de cause sans thème : au-delà de 5 %, **constat mineur** |
| (information) mesures | nombre à définir et à revoir : chaque mesure fait l'objet d'un **constat majeur** |
| (information) causes contributives écartées, causes non précisées, reprises du code, refus d'outils | |

Chaque mesure du plan est contrôlée par le code : « À définir » (constat majeur avec les chiffres et
une consigne), ou action qui n'est pas préventive, sans mécanisme lié à sa cause, ou qui nomme un
autre site (constat majeur avec la consigne de réécriture).

Les problèmes trouvés par ces contrôles deviennent des **constats ciblés** (source : code), avec
une consigne de correction.

**2. Relecture par un agent LLM, exigeant par consigne.** Il reçoit le rapport, les contrôles, les
chiffres recalculés de chaque site et les critères mesurés sur chaque site (rang parmi les sites du
même segment). Il vérifie pour chaque site : la caractérisation, les causes dominantes, et chaque
mesure (action de terrain sur ce site, cause fréquente ou coûteuse là, préventive, pas un arrêt nominal).
Il consigne au moins un point **pour chaque site et pour la synthèse** (sinon `conclude` est refusé).

| Outil | Ce qu'il fait |
|---|---|
| `record_finding(gravite, cible_site, cible_champ, constat, preuve, consigne)` | un point vérifié : `conforme`, ou un problème `bloquant` / `majeur` / `mineur` ; cible = site, `synthese` ou `general` ; champ = caracterisation, causes_dominantes, mesures, synthese, autre ; **consigne obligatoire** si bloquant ou majeur |
| `site_facts(site)` | chiffres recalculés (parts des thèmes de causes et de conséquences, paires cause -> conséquence, critères du segment, maintenance suspectée, coût) et ce que le rapport dit du site, mesures comprises |
| `theme_facts(theme_id)` | contenu réel d'un thème : phrases et rôles, sites, exemples (repli sur la table des incidents) |
| `read_report`, `read_journal(start, count)` | rapport complet, journal de l'analyse par pages |
| `conclude(verdict, synthese)` | **outil final** ; refusé tant qu'un site ou la synthèse n'a aucun point ; `valide` impossible avec un problème majeur ou bloquant ou un contrôle en échec. Si l'agent ne conclut pas, le verdict est calculé par ces mêmes règles (« verdict automatique ») |

**Pourquoi le vérificateur ne fait plus réécrire.** Une première version renvoyait ses constats au
rédacteur. Avec le petit modèle, ses consignes ont poussé le rédacteur à recopier des formules au lieu
de corriger le fond. La correction des textes est donc confiée au juge, dont chaque correction est
validée par le code.

**Le plan ne couvre que les couples sélectionnés par le coût** (80 % du coût des causes connues, au
plus 12 mesures, au moins 5 incidents par couple). Un site sans mesure est normal si ses causes sont
mineures.

## Sorties (`outputs/3_verification/`)

| Fichier | Contenu |
|---|---|
| `verification.md` | verdict, contrôles automatiques, problèmes (cible, consigne), points conformes |
| `verification.json` | la même chose, structurée |
| `journal.md` | déroulé de la session du vérificateur |
