"""Pipeline complète, étape par étape :

  1. pdf     extraction des rapports PDF (digitaux + scans OCR) -> outputs/1_extraction/
  2. topics  topic modeling BERTopic sur tous les incidents     -> outputs/2_topics/

Chaque étape garde son cache interne dans outputs/cache/ : relancer ne refait que ce qui a
changé (nouveaux PDF, nouveaux textes). Organisation de outputs/ : voir paths.py.

Usage :
    python main.py                    # toutes les étapes
    python main.py --steps topics     # seulement le topic modeling
    python main.py --min-cluster-size 15
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from paths import DATA_CSV, INCIDENTS_FULL
from pdf_pipeline import config as pdf_config
from pdf_pipeline.pipeline import run as run_pdf
from topic_modeling import config as topic_config
from topic_modeling.pipeline import run as run_topics

STEPS = ["pdf", "topics"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="étapes à lancer")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="CSV des incidents")
    parser.add_argument("--min-cluster-size", type=int, default=topic_config.MIN_CLUSTER_SIZE)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)  # chemins relatifs à la racine du repo
    for i, step in enumerate([s for s in STEPS if s in args.steps], 1):
        print(f"\n{'=' * 80}\nÉtape {i} : {step}\n{'=' * 80}")
        start = time.time()
        if step == "pdf":
            run_pdf(args.csv, engine="combined", dpi=pdf_config.DPI, force=False,
                    workers=os.cpu_count())
        elif step == "topics":
            run_topics(INCIDENTS_FULL, args.min_cluster_size, split=True)
        print(f"-> étape {step} terminée en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
