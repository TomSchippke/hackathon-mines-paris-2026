"""Pipeline complète, étape par étape :

  1. pdf           extraction des rapports PDF (digitaux + scans OCR)  -> outputs/1_extraction/
  2. analyse       agent d'analyse : clustering, causes, thèmes, profils de site, rapport
                                                                       -> outputs/2_analyse/
  3. verification  agent vérificateur : relit le journal et le rapport -> outputs/3_verification/

Les étapes 2 et 3 appellent Azure OpenAI (il faut le .env). Chaque étape garde un cache dans
outputs/cache/ : relancer ne refait que ce qui a changé, et une session d'agent déjà faite est
rejouée à l'identique sans aucun appel. Organisation de outputs/ : voir paths.py.

Usage :
    python main.py                          # toutes les étapes
    python main.py --steps analyse verification
    python main.py --steps verification     # revérifier sans refaire l'analyse
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from analysis_agent.pipeline import run as run_analysis
from paths import DATA_CSV
from pdf_pipeline.pipeline import run as run_pdf
from review_agent import config as review_config
from review_agent.pipeline import run as run_review

STEPS = ["pdf", "analyse", "verification"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="étapes à lancer")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="CSV des incidents")
    parser.add_argument("--max-turns-analyse", type=int, default=None,
                        help="tours maximum par phase de l'analyse (défaut : analysis_agent/config.py)")
    parser.add_argument("--max-turns-verification", type=int, default=review_config.MAX_TURNS)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)  # chemins relatifs à la racine du repo
    for i, step in enumerate([s for s in STEPS if s in args.steps], 1):
        print(f"\n{'=' * 80}\nÉtape {i} : {step}\n{'=' * 80}")
        start = time.time()
        if step == "pdf":
            run_pdf(args.csv)
        elif step == "analyse":
            if not run_analysis(args.max_turns_analyse) and "verification" in args.steps:
                print("La vérification va quand même tourner : elle signalera ce qui manque.")
        elif step == "verification":
            run_review(args.max_turns_verification)
        print(f"-> étape {step} terminée en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
