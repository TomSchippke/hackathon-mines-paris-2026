"""Pipeline d'extraction des rapports PDF (digitaux et scannés) vers un CSV complet.

Étapes :
  1. Lecture du CSV : chaque ligne avec un `document_path` est un rapport PDF à extraire.
  2. Couche texte : un PDF digital est lu directement ; sans texte, c'est un scan.
  3. Référence : valeurs connues et plages de coût horaire, tirées des rapports sans OCR.
  4. OCR des scans avec docTR (lectures mises en cache, indexées par le contenu du PDF).
  5. Découpage en champs, nettoyage, contrôles -> valeurs finales + avertissements.
  6. Écriture du CSV complet et récapitulatif.

Usage :
    python -m pdf_pipeline           # -> outputs/1_extraction/incidents_full.csv
    python -m pdf_pipeline --force   # ignore le cache et relit tous les scans
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm import tqdm

from paths import CACHE_DIR, DATA_CSV, INCIDENTS_FULL

from .cache import OcrCache, file_sha1
from .config import DPI, MIN_TEXT_CHARS
from .ocr_doctr import DoctrReader
from .parsing import split_fields
from .validation import build_reference, extract_fields

# Même dossier que les versions précédentes : les lectures déjà faites sont réutilisées.
OCR_CACHE_DIR = CACHE_DIR / "ocr_doctr"


@dataclass
class Report:
    incident_id: int
    path: Path
    site: str
    sha1: str = ""
    text: str | None = None       # texte du rapport : couche texte (digital) ou lecture docTR (scan)
    method: str = ""              # text_layer | ocr_doctr
    error: str | None = None


def load_reports(df: pd.DataFrame, data_root: Path) -> list[Report]:
    """Étapes 1-2 : rapports à traiter et couche texte des PDF digitaux."""
    reports = []
    for row in df[df["document_path"].notna()].itertuples():
        r = Report(int(row.incident_id), data_root / row.document_path,
                   row.site if isinstance(row.site, str) else "")
        try:
            r.sha1 = file_sha1(r.path)
            with pymupdf.open(r.path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) >= MIN_TEXT_CHARS:
                r.text, r.method = text, "text_layer"
        except Exception as e:  # fichier absent ou corrompu : signalé, n'arrête pas la pipeline
            r.error = f"{type(e).__name__}: {e}"
        reports.append(r)
    return reports


def ocr_scans(scans: list[Report], dpi: int, force: bool) -> int:
    """Étape 4 : texte docTR de chaque scan, depuis le cache si déjà lu. Renvoie le nombre de
    nouvelles lectures."""
    cache = OcrCache(OCR_CACHE_DIR)
    reader = DoctrReader(dpi)  # le modèle n'est chargé que s'il y a vraiment un scan à lire
    todo = []
    for s in scans:
        reading = None if force else cache.get(s.sha1)
        if reading is None:
            todo.append(s)
        else:
            s.text, s.method = reading["texts"][0], "ocr_doctr"
    for s in tqdm(todo, desc="OCR docTR", disable=not todo):
        try:
            reading = reader.read_scan(s.path)
            cache.put(s.sha1, reading)
            s.text, s.method = reading["texts"][0], "ocr_doctr"
        except Exception as e:
            s.error = f"docTR: {type(e).__name__}: {e}"
    return len(todo)


def run(csv_path: Path = DATA_CSV, dpi: int = DPI, force: bool = False) -> Path:
    data_root = csv_path.parent.parent  # les document_path sont relatifs à la racine du repo
    df = pd.read_csv(csv_path)

    # 1-2. Rapports PDF et couche texte
    reports = load_reports(df, data_root)
    digital = [r for r in reports if r.text is not None]
    scans = [r for r in reports if r.text is None and r.error is None]
    print(f"{len(reports)} PDF : {len(digital)} digitaux, {len(scans)} scans, "
          f"{sum(r.error is not None for r in reports)} illisibles")

    # 3. Référence (rapports sans OCR)
    for r in digital:
        if not r.site:  # rapport ajouté sans site dans le CSV : on prend celui du PDF
            r.site = split_fields(r.text).get("site", "")
    ref = build_reference(df[df["document_format"] == "text"], [(r.text, r.site) for r in digital])

    # 4. OCR des scans
    new_readings = ocr_scans(scans, dpi, force)

    # 5. Valeurs finales
    full = df.set_index("incident_id")
    full["extraction_method"] = full["document_format"].map(lambda f: "csv" if f == "text" else "")
    full["extraction_warnings"] = ""
    for r in reports:
        if r.text is None:
            full.at[r.incident_id, "extraction_method"] = "error"
            full.at[r.incident_id, "extraction_warnings"] = f"erreur: {r.error}"
            continue
        if not r.site:
            r.site = split_fields(r.text).get("site", "")
        fields, warnings = extract_fields(r.text, r.site, ref)
        for col, value in fields.items():
            full.at[r.incident_id, col] = value
        full.at[r.incident_id, "extraction_method"] = r.method
        full.at[r.incident_id, "extraction_warnings"] = " | ".join(warnings)
    full = full.reset_index()

    # 6. CSV complet
    INCIDENTS_FULL.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(INCIDENTS_FULL, index=False)
    print(f"\nNouvelles lectures docTR : {new_readings} (cache : {OCR_CACHE_DIR}/)")
    print(f"Sauvegardé : {INCIDENTS_FULL} ({len(full)} lignes)")
    print("\nMéthode d'extraction :\n" + full["extraction_method"].value_counts().to_string())
    flagged = full[full["extraction_warnings"] != ""]
    print(f"\n{len(flagged)} rapports à vérifier (colonne extraction_warnings)")
    for row in flagged.head(20).itertuples():
        print(f"  incident {row.incident_id} : {row.extraction_warnings}")
    return INCIDENTS_FULL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DATA_CSV)
    parser.add_argument("--dpi", type=int, default=DPI, help="résolution de rendu des scans")
    parser.add_argument("--force", action="store_true", help="ignore le cache et relit tous les scans")
    args = parser.parse_args()
    run(args.csv, args.dpi, args.force)
