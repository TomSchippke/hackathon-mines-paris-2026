"""Pipeline complète, étape par étape :

  1. pdf           extraction des rapports PDF (digitaux + scans OCR) -> outputs/incidents_full.csv
  2. topics        topic modeling BERTopic sur tous les incidents     -> outputs/topics/ (optionnel)
  3. segmentation  clustering agentique des incidents                 -> outputs/segmentation/
  4. equipment     analyse par équipement                             -> outputs/equipment/
  5. report        rapport HTML autonome (le livrable)                -> outputs/rapport.html
  6. webapp        application Streamlit de visualisation (optionnel, interactif)

Chaque étape met ses résultats en cache dans outputs/cache/ : relancer ne refait que ce qui a
changé (nouveaux PDF, nouveaux textes).

Usage :
    python main.py                        # pdf, segmentation, equipment, report (défaut)
    python main.py --steps topics         # seulement le topic modeling (optionnel)
    python main.py --steps segmentation equipment webapp
    python main.py --steps topics --min-cluster-size 15
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from pdf_pipeline import config as pdf_config
from pdf_pipeline.pipeline import run as run_pdf
from topic_modeling import config as topic_config
from topic_modeling.pipeline import run as run_topics
from agentic_pipeline.pipeline import run_agentic_segmentation
from equipment_analysis.pipeline import run_equipment_analysis

STEPS = ["pdf", "topics", "segmentation", "equipment", "report", "webapp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=["pdf", "segmentation", "equipment", "report"], help="étapes à lancer")
    parser.add_argument("--csv", type=Path, default=pdf_config.DEFAULT_CSV, help="CSV des incidents")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--min-cluster-size", type=int, default=topic_config.MIN_CLUSTER_SIZE,
                        help="taille minimale des clusters (étape topics uniquement)")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)  # chemins relatifs à la racine du repo
    for i, step in enumerate([s for s in STEPS if s in args.steps], 1):
        print(f"\n{'=' * 80}\nÉtape {i} : {step}\n{'=' * 80}")
        start = time.time()
        if step == "pdf":
            run_pdf(args.csv, args.out_dir, engine="combined", dpi=pdf_config.DPI, force=False,
                    workers=os.cpu_count())
        elif step == "topics":
            run_topics(args.out_dir / "incidents_full.csv", args.out_dir, args.min_cluster_size,
                       split=True)
        elif step == "segmentation":
            incidents_file = args.out_dir / "incidents_full.csv"
            if not incidents_file.exists():
                incidents_file = Path("data/incidents_full.csv")
            run_agentic_segmentation(
                input_csv=incidents_file,
                out_dir=args.out_dir / "segmentation",
            )
        elif step == "equipment":
            incidents_file = args.out_dir / "segmentation" / "incidents_segmented.csv"
            if not incidents_file.exists():
                incidents_file = args.out_dir / "incidents_full.csv"
            run_equipment_analysis(
                input_csv=incidents_file,
                out_dir=args.out_dir / "equipment",
            )
        elif step == "report":
            from webapp.export_html import run as run_report
            run_report(args.out_dir / "rapport.html")
        elif step == "webapp":
            import subprocess
            subprocess.run(["uv", "run", "streamlit", "run", "webapp/app.py"])
        print(f"-> étape {step} terminée en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
