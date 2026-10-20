# Analyse chronologique par équipement

Étape `equipment` de `main.py`. Elle reconstitue la chronologie des incidents de chaque actif
(`equipment_id`) et calcule des indicateurs descriptifs de fiabilité. Elle lit de préférence
`outputs/segmentation/incidents_segmented.csv`, sinon `outputs/incidents_full.csv`.

## Contenu

1. **Chronologie de chaque actif** (`timeline.py`) :
   - classement des pannes par date (ex æquo : `incident_id`), numéro d'ordre, coût et indisponibilité cumulés ;
   - intervalles entre pannes consécutives (`days_since_last_incident`, `days_until_next_incident`) ;
   - incident précédent / suivant (identifiant, sévérité, titre, cluster) et `same_cluster_as_prev`.
2. **Indicateurs par équipement** (`metrics.py`) : nombre d'incidents, période active, coût et arrêt (total,
   moyen), MTBF (moyenne des écarts), écart médian et minimal, incidents de sévérité maximale, problème le
   plus fréquent.
3. **Score « bad actor »** : coût total (4/7) et heures d'arrêt totales (3/7), chacun normalisé min-max
   entre équipements, ramené sur 0-100.

## Ce qui n'est plus calculé, et pourquoi

Les récidives à fenêtre fixe et l'audit par technicien ont été retirés.

- L'écart médian entre deux incidents d'un même équipement est de 21 jours : un taux de récidive à fenêtre
  fixe (7, 15, 30, 60 jours) est alors à peu près égal au taux de base attendu au hasard.
- Comparés au hasard, aucun équipement ne dépasse le niveau attendu, et aucun technicien ne se distingue de
  son site.
- Sans comparaison ajustée au site (machines et conditions différentes), des indicateurs par technicien
  seraient injustes.

## Utilisation

```bash
python -m equipment_analysis          # module direct
python main.py --steps equipment      # via le pipeline
```

## Fichiers générés (`outputs/equipment/`)

| Fichier | Contenu |
|---|---|
| `incidents_with_timeline.csv` | chaque incident enrichi : `incident_order`, coûts et arrêts cumulés, délais inter-pannes, incident précédent / suivant, `same_cluster_as_prev` |
| `equipment_summary.csv` | une ligne par équipement : `total_incidents`, dates, `active_span_days`, coûts et arrêts (total, moyen), `mtbf_days`, `median_days_between_incidents`, `min_days_between_incidents`, `severe_incidents`, `most_frequent_problem`, `bad_actor_score` ; trié par score décroissant |
