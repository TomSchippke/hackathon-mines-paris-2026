"""Pipeline complète, étape par étape :

  1. pdf           extraction des rapports PDF (digitaux + scans OCR)  -> outputs/1_extraction/
  2. analyse       agent d'analyse : clustering, causes, thèmes, profils de site, rapport
                                                                       -> outputs/2_analyse/
  3. verification  contrôles du code + agent vérificateur (avis)      -> outputs/3_verification/
  4. webapp        tableau de bord Streamlit (lecture seule des sorties, à lancer à la demande)
  5. export        rapport interactif en un fichier HTML autonome, à envoyer -> outputs/2_analyse/rapport_interactif.html

Les étapes 2 et 3 appellent Azure OpenAI (il faut le .env). Chaque étape garde un cache dans
outputs/cache/ : relancer ne refait que ce qui a changé, et une session d'agent déjà faite est
rejouée à l'identique sans aucun appel. Organisation de outputs/ : voir paths.py.

Usage :
    python main.py                          # toutes les étapes
    python main.py --steps analyse verification
    python main.py --steps verification     # revérifier sans refaire l'analyse
    python main.py --steps webapp           # ouvrir le tableau de bord
    python main.py --steps export           # fichier HTML à envoyer (s'ouvre sans Python)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from analysis_agent.pipeline import run as run_analysis
from paths import DATA_CSV, REVIEW_DIR
from pdf_pipeline.pipeline import run as run_pdf
from review_agent import config as review_config
from review_agent.pipeline import run as run_review

STEPS = ["pdf", "analyse", "verification", "export", "webapp"]
DEFAULT_STEPS = ["pdf", "analyse", "verification", "export"]  # la webapp bloque le terminal : seulement sur demande


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=DEFAULT_STEPS, help="étapes à lancer")
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
            for old in REVIEW_DIR.glob("*_avant_reecriture*"):  # restes d'anciennes versions de la pipeline
                old.unlink()
            run_review(args.max_turns_verification)
        elif step == "export":
            from webapp.export_html import build
            print(f"Rapport interactif : {build()}")
        elif step == "webapp":
            subprocess.run([sys.executable, "-m", "streamlit", "run", "webapp/app.py"])
        print(f"-> étape {step} terminée en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
