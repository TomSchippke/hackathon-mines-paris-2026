"""Compare plusieurs extractions des scans (sorties de la pipeline avec différents moteurs).

- précision champ par champ sur les scans relus à la main (`outputs/verite_terrain_scans.csv`) ;
- désaccords entre les deux premiers fichiers, scan par scan ;
- qualité des descriptions : part des mots présents dans les rapports sans OCR ;
- nombre de rapports signalés dans `extraction_warnings`.

Usage :
    python -m pdf_pipeline.compare
    python -m pdf_pipeline.compare outputs/incidents_full_tesseract.csv outputs/incidents_full_doctr.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .parsing import text_quality, words

FIELDS = ["equipment_id", "date", "severity", "downtime_hours", "estimated_cost_eur",
          "weather_condition", "technician_id"]
DEFAULT_FILES = ["outputs/incidents_full_tesseract.csv", "outputs/incidents_full_doctr.csv",
                 "outputs/incidents_full.csv"]


def same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Égalité valeur à valeur, deux valeurs manquantes étant considérées comme égales."""
    return (a == b) | (a.isna() & b.isna())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--truth", type=Path, default=Path("outputs/verite_terrain_scans.csv"))
    args = parser.parse_args()

    files = [Path(f) for f in args.files if Path(f).exists()]
    if len(files) < 2:
        raise SystemExit(f"Il faut au moins deux fichiers existants parmi : {args.files}")
    full = {f.stem: pd.read_csv(f).set_index("incident_id") for f in files}
    names = list(full)
    first = full[names[0]]
    scans = first.index[first["document_format"] == "pdf_scan"]
    print(f"{len(scans)} scans, fichiers : {', '.join(names)}\n")

    if args.truth.exists():
        truth = pd.read_csv(args.truth).set_index("incident_id")
        print(f"== Précision sur {len(truth)} scans relus à la main")
        table = pd.DataFrame({n: [f"{same(d.loc[truth.index, f], truth[f]).sum()}/{len(truth)}" for f in FIELDS]
                              for n, d in full.items()}, index=FIELDS)
        table.loc["TOTAL"] = [f"{sum(same(d.loc[truth.index, f], truth[f]).sum() for f in FIELDS)}/{truth.size}"
                              for d in full.values()]
        print(table.to_string(), "\n")

    a, b = names[0], names[1]
    print(f"== Désaccords {a} / {b}")
    disagree = pd.DataFrame({f: ~same(full[a].loc[scans, f], full[b].loc[scans, f]) for f in FIELDS})
    print(f"scans avec au moins un désaccord : {disagree.any(axis=1).sum()}/{len(scans)}")
    for i in disagree.index[disagree.any(axis=1)][:40]:
        cols = disagree.columns[disagree.loc[i]]
        print(f"  incident {i}: " + ", ".join(f"{c} {full[a].at[i, c]!r} / {full[b].at[i, c]!r}" for c in cols))
    print()

    vocabulary = {w for t in first.loc[first["document_format"] != "pdf_scan", "text"].dropna() for w in words(t)}
    print("== Descriptions des scans (part de mots connus) et rapports signalés")
    for n, d in full.items():
        q = d.loc[scans, "text"].map(lambda t: text_quality(t, vocabulary))
        flagged = (d.loc[scans, "extraction_warnings"].fillna("") != "").sum()
        print(f"  {n:28s} qualité {q.mean():.4f} | descriptions < 95 % : {(q < 0.95).sum():3d} "
              f"| signalés : {flagged}")


if __name__ == "__main__":
    main()
