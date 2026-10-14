"""Pipeline orchestrant l'analyse par équipement et technicien."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_INPUT_DATA,
    DEFAULT_INPUT_FALLBACK,
    DEFAULT_INPUT_FULL,
    DEFAULT_INPUT_SEGMENTED,
    DEFAULT_OUT_DIR,
    RECURRENCE_WINDOWS,
)
from .metrics import compute_equipment_metrics, compute_technician_metrics
from .timeline import compute_equipment_timeline


def resolve_input_csv(custom_path: Path | str | None = None) -> Path:
    """Résout le chemin du CSV d'entrée en priorisant les données enrichies."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    for candidate in [
        DEFAULT_INPUT_SEGMENTED,
        DEFAULT_INPUT_FULL,
        DEFAULT_INPUT_DATA,
        DEFAULT_INPUT_FALLBACK,
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Aucun fichier d'incidents trouvé. Assurez-vous d'avoir exécuté la pipeline ou d'avoir les données dans data/."
    )


def run_equipment_analysis(
    input_csv: Path | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    windows: list[int] = RECURRENCE_WINDOWS,
) -> dict[str, Any]:
    """Exécute l'analyse chronologique complète et sauvegarde les résultats."""
    resolved_csv = resolve_input_csv(input_csv)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"-> Lecture des incidents depuis : {resolved_csv}")
    df = pd.read_csv(resolved_csv)
    print(f"-> {len(df)} incidents chargés. Calcul des chronologies par équipement...")

    # 1. Timeline chronologique
    timeline_df = compute_equipment_timeline(df, windows=windows)
    timeline_file = out_path / "incidents_with_timeline.csv"
    timeline_df.to_csv(timeline_file, index=False)
    print(f"   [OK] Chronologie sauvegardée dans {timeline_file}")

    # 2. Métriques par équipement
    eq_metrics_df = compute_equipment_metrics(timeline_df, windows=windows)
    eq_file = out_path / "equipment_summary.csv"
    eq_metrics_df.to_csv(eq_file, index=False)
    print(f"   [OK] Synthèse par équipement sauvegardée dans {eq_file}")

    # 3. Métriques par technicien
    tech_metrics_df = compute_technician_metrics(timeline_df, windows=windows)
    tech_file = out_path / "technician_performance.csv"
    tech_metrics_df.to_csv(tech_file, index=False)
    print(f"   [OK] Performance techniciens sauvegardée dans {tech_file}")

    print(
        f"-> Analyse terminée avec succès : {len(eq_metrics_df)} équipements et {len(tech_metrics_df)} techniciens analysés."
    )

    return {
        "timeline_df": timeline_df,
        "equipment_df": eq_metrics_df,
        "technician_df": tech_metrics_df,
        "output_dir": out_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Chemin du CSV d'entrée")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Dossier de sortie")
    args = parser.parse_args()

    run_equipment_analysis(input_csv=args.csv, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
