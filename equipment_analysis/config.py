"""Configuration pour l'analyse chronologique par équipement."""

from pathlib import Path

from agentic_pipeline.config import SEVERITY_ORDER

DEFAULT_INPUT_SEGMENTED = Path("outputs/segmentation/incidents_segmented.csv")
DEFAULT_INPUT_FULL = Path("outputs/incidents_full.csv")
DEFAULT_INPUT_DATA = Path("data/incidents_full.csv")
DEFAULT_INPUT_FALLBACK = Path("data/incidents_students.csv")

DEFAULT_OUT_DIR = Path("outputs/equipment")

DATE_COL = "date"
EQUIPMENT_COL = "equipment_id"
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
