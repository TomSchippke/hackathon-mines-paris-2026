# Analyse Chronologique par Équipement & Efficacité Techniciens

Ce module recentre l'analyse des incidents sur les actifs individuels (`equipment_id`) et évalue la qualité des réparations des intervenants (`technician_id`).

## Objectifs et Méthodologie

1. **Reconstitution de la timeline de chaque actif** :
   - Classement des pannes par ordre chronologique pour chaque machine.
   - Calcul des intervalles temporels entre pannes consécutives (`days_since_last_incident` et `days_until_next_incident`).
   - Détermination du MTBF (Mean Time Between Failures) et de l'historique cumulé (durée d'arrêt, coûts).

2. **Récidives comparées au hasard** :
   - Récidive : incident survenant dans les 7, 15, 30 ou 60 jours suivant le précédent sur la même machine.
   - Une machine qui tombe souvent en panne a mécaniquement beaucoup d'intervalles courts. Chaque équipement est
     donc comparé à un modèle nul : mêmes première et dernière dates, incidents intermédiaires tirés uniformément
     entre les deux (`RECURRENCE_N_SIMULATIONS` = 2000). On publie les récidives attendues, l'excès, la p-value et
     la q-value (Benjamini-Hochberg sur les équipements, `RECURRENCE_FDR` = 0,05). Seul un excès significatif
     (`recurrence_above_chance_{w}d`) signale une réparation qui ne tient pas.

3. **Techniciens comparés à leur site** :
   - Taux de récidive après intervention, comparé au taux des **autres techniciens du même site** (mêmes
     machines, mêmes conditions), jamais à la flotte : test binomial bilatéral, puis Benjamini-Hochberg.
   - Seul technicien d'un site : pas de comparaison possible (valeurs vides).
   - Délai moyen de bon fonctionnement de l'équipement avant la panne suivante (`mean_post_survival_days`).

4. **Identification des « Bad Actors »** :
   - Score combinant coût cumulé (40 %), indisponibilité cumulée (30 %) et récidives au-delà du hasard sous
     30 jours (30 %). Si aucun équipement n'est au-dessus du hasard, le score ne repose que sur le coût et
     l'indisponibilité (poids renormalisés).

Sur les données actuelles, aucun équipement ne rechute plus que le hasard (1109 récidives sous 30 jours pour
1077 attendues) et aucun technicien ne se distingue de son site.

Limite : le modèle nul ne tient pas compte des événements qui touchent tout un site en même temps (tempête,
canicule) ; des rafales d'incidents liées à la météo peuvent donc apparaître comme des récidives.

## Utilisation

```bash
# Via le module direct
python -m equipment_analysis

# Via le pipeline complet
python main.py --steps equipment
```

## Fichiers générés (`outputs/equipment/`)

- `incidents_with_timeline.csv` : Chaque incident enrichi des métriques séquentielles (ordre, jours depuis la dernière panne, récidive, antécédent).
- `equipment_summary.csv` : Vue agrégée par actif (55 équipements) avec MTBF, coûts totaux, récidives observées et attendues au hasard (`expected_recurrences_{w}d`, `excess_recurrences_{w}d`, `recurrence_p_{w}d`, `recurrence_q_{w}d`, `recurrence_above_chance_{w}d`) et score de criticité.
- `technician_performance.csv` : Évaluation des 18 techniciens (interventions, délai moyen de survie de la machine, taux de récidive, taux des autres techniciens du site `site_recurrence_rate_{w}d`, `post_recurrence_p_vs_site_{w}d`, `post_recurrence_q_vs_site_{w}d`, `differs_from_site_{w}d`).
