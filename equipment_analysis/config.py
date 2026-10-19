"""Configuration pour l'analyse chronologique par équipement et technicien."""

from pathlib import Path

from agentic_pipeline.config import SEVERITY_ORDER

DEFAULT_INPUT_SEGMENTED = Path("outputs/segmentation/incidents_segmented.csv")
DEFAULT_INPUT_FULL = Path("outputs/incidents_full.csv")
DEFAULT_INPUT_DATA = Path("data/incidents_full.csv")
DEFAULT_INPUT_FALLBACK = Path("data/incidents_students.csv")

DEFAULT_OUT_DIR = Path("outputs/equipment")

RECURRENCE_WINDOWS = [7, 15, 30, 60]
DEFAULT_RECURRENCE_WINDOW = 30

# Récidives comparées au hasard : dates d'un équipement redistribuées uniformément sur sa période d'activité
RECURRENCE_N_SIMULATIONS = 2000
RECURRENCE_FDR = 0.05  # Benjamini-Hochberg sur les équipements (resp. les techniciens), par fenêtre
RANDOM_STATE = 42

DATE_COL = "date"
EQUIPMENT_COL = "equipment_id"
TECHNICIAN_COL = "technician_id"
SEVERITY_COL = "severity"
DOWNTIME_COL = "downtime_hours"
COST_COL = "estimated_cost_eur"
SITE_COL = "site"
ASSET_COL = "asset_type"
ID_COL = "incident_id"
PROBLEM_COL = "problem_title"
CLUSTER_COL = "cluster_id"

# Niveau de sévérité le plus élevé (dernier élément de SEVERITY_ORDER)
TOP_SEVERITY = SEVERITY_ORDER[-1]
