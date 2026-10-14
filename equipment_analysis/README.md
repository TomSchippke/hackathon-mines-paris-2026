# Analyse Chronologique par Équipement & Efficacité Techniciens

Ce module recentre l'analyse des incidents sur les actifs individuels (`equipment_id`) et évalue la qualité des réparations des intervenants (`technician_id`).

## Objectifs et Méthodologie

1. **Reconstitution de la timeline de chaque actif** :
   - Classement des pannes par ordre chronologique pour chaque machine.
   - Calcul des intervalles temporels entre pannes consécutives (`days_since_last_incident` et `days_until_next_incident`).
   - Détermination du MTBF (Mean Time Between Failures) et de l'historique cumulé (durée d'arrêt, coûts).

2. **Détection des récidives post-intervention** :
   - Qualification des récidives selon des fenêtres strictes (7, 15, 30 et 60 jours).
   - Détection des défaillances précoces : une panne survenant dans la fenêtre suivant une maintenance est signalée comme un rappel/défaut de maintien en état.

3. **Audit et efficacité des techniciens** :
   - Taux de rappel / récidive post-intervention par technicien.
   - Délai moyen de bon fonctionnement de l'équipement avant la panne suivante (`mean_post_survival_days`).

4. **Identification des « Bad Actors »** :
   - Scoring composite combinant coût cumulé (40 %), indisponibilité cumulée (30 %) et instabilité/taux de récidive (30 %).

## Utilisation

```bash
# Via le module direct
python -m equipment_analysis

# Via le pipeline complet
python main.py --steps equipment
```

## Fichiers générés (`outputs/equipment/`)

- `incidents_with_timeline.csv` : Chaque incident enrichi des métriques séquentielles (ordre, jours depuis la dernière panne, récidive, antécédent).
- `equipment_summary.csv` : Vue agrégée par actif (55 équipements) avec MTBF, coûts totaux, taux de récidive et score de criticité.
- `technician_performance.csv` : Évaluation des 18 techniciens (interventions totales, délai moyen de survie de la machine, taux de récidive).
