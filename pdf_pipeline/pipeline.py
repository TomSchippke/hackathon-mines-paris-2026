"""Pipeline d'extraction des rapports PDF (digitaux et scannés) vers un CSV complet.

Étapes :
  1. Lecture du CSV : chaque ligne avec un `document_path` est un rapport PDF à extraire.
  2. Couche texte : un PDF digital est lu directement ; sans texte, c'est un scan.
  3. Référence : valeurs connues et plages de coût horaire, tirées des rapports sans OCR.
  4. OCR des scans (lectures brutes mises en cache, indexées par le contenu du PDF) :
       combined  (défaut) docTR sur tous les scans, Tesseract seulement là où docTR laisse
                 un champ vide ou un coût/durée implausible ;
       doctr     docTR seul ;
       tesseract Tesseract seul (4 lectures + vote).
  5. Nettoyage, validation et combinaison des lectures -> valeurs finales + avertissements.
  6. Écriture du CSV complet et récapitulatif.

Usage :
    python -m pdf_pipeline                     # moteur combined -> outputs/incidents_full.csv
    python -m pdf_pipeline --engine doctr      # -> outputs/incidents_full_doctr.csv
    python -m pdf_pipeline --engine tesseract  # -> outputs/incidents_full_tesseract.csv
    python -m pdf_pipeline --force             # ignore le cache et relit tous les scans
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm import tqdm

from . import ocr_tesseract
from .cache import OcrCache, file_sha1
from .config import DEFAULT_CSV, DEFAULT_OUT_DIR, DPI, MIN_TEXT_CHARS
from .ocr_doctr import DoctrReader
from .parsing import split_fields
from .resolve import build_reference, combine, final_warnings, needs_fallback, resolve_reading


@dataclass
class Report:
    incident_id: int
    path: Path
    site: str
    sha1: str = ""
    text_layer: str | None = None  # texte du PDF digital ; None pour un scan
    error: str | None = None


# --------------------------------------------------------------------------------------------
# Étapes 1-2 : rapports à traiter, couche texte
# --------------------------------------------------------------------------------------------

def load_reports(df: pd.DataFrame, data_root: Path) -> list[Report]:
    reports = []
    for row in df[df["document_path"].notna()].itertuples():
        r = Report(int(row.incident_id), data_root / row.document_path,
                   row.site if isinstance(row.site, str) else "")
        try:
            r.sha1 = file_sha1(r.path)
            with pymupdf.open(r.path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            r.text_layer = text if len(text.strip()) >= MIN_TEXT_CHARS else None
        except Exception as e:  # fichier absent ou corrompu : signalé, n'arrête pas la pipeline
            r.error = f"{type(e).__name__}: {e}"
        reports.append(r)
    return reports


# --------------------------------------------------------------------------------------------
# Étape 4 : OCR avec cache
# --------------------------------------------------------------------------------------------

def split_cached(scans: list[Report], cache: OcrCache, force: bool) -> tuple[dict[str, dict], list[Report]]:
    """Sépare les scans déjà lus (lecture prise dans le cache) de ceux à lire."""
    out, todo = {}, []
    for s in scans:
        reading = None if force else cache.get(s.sha1)
        if reading is None:
            todo.append(s)
        else:
            out[s.sha1] = reading
    return out, todo


def doctr_readings(scans: list[Report], cache: OcrCache, dpi: int, force: bool) -> dict[str, dict]:
    out, todo = split_cached(scans, cache, force)
    reader = DoctrReader(dpi)  # le modèle n'est chargé que s'il y a vraiment un scan à lire
    for s in tqdm(todo, desc="OCR docTR", disable=not todo):
        try:
            out[s.sha1] = reader.read_scan(s.path)
            cache.put(s.sha1, out[s.sha1])
        except Exception as e:
            s.error = f"docTR: {type(e).__name__}: {e}"
    return out


def tesseract_readings(scans: list[Report], cache: OcrCache, dpi: int, force: bool,
                       workers: int) -> dict[str, dict]:
    out, todo = split_cached(scans, cache, force)
    if todo and shutil.which("tesseract") is None:
        raise SystemExit("Tesseract introuvable : `brew install tesseract tesseract-lang`.")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ocr_tesseract.read_scan, str(s.path), dpi): s for s in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="OCR Tesseract",
                           disable=not todo):
            s = futures[future]
            try:
                out[s.sha1] = future.result()
                cache.put(s.sha1, out[s.sha1])
            except Exception as e:
                s.error = f"Tesseract: {type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------

def run(csv_path: Path, out_dir: Path, engine: str, dpi: int, force: bool, workers: int) -> Path:
    data_root = csv_path.parent.parent  # les document_path sont relatifs à la racine du repo
    df = pd.read_csv(csv_path)

    # 1-2. Rapports PDF et couche texte
    reports = load_reports(df, data_root)
    digital = [r for r in reports if r.text_layer is not None]
    scans = [r for r in reports if r.text_layer is None and r.error is None]
    for r in reports:
        if not r.site:  # rapport ajouté sans site dans le CSV : on prend celui du PDF
            r.site = split_fields(r.text_layer or "").get("site", "")
    print(f"{len(reports)} PDF : {len(digital)} digitaux, {len(scans)} scans, "
          f"{sum(r.error is not None for r in reports)} illisibles")

    # 3. Référence
    ref = build_reference(df[df["document_format"] == "text"], [(r.text_layer, r.site) for r in digital])

    # 4. OCR des scans
    cache_root = out_dir / "cache"
    primary = "tesseract" if engine == "tesseract" else "doctr"
    caches = {e: OcrCache(cache_root, e) for e in ("doctr", "tesseract")}
    before = {e: len(c) for e, c in caches.items()}
    if primary == "doctr":
        readings = doctr_readings(scans, caches["doctr"], dpi, force)
    else:
        readings = tesseract_readings(scans, caches["tesseract"], dpi, force, workers)

    # 5. Valeurs finales
    results: dict[int, tuple[dict, str, list[str]]] = {}
    for r in digital:
        reading = {"texts": [r.text_layer]}
        fields, _ = resolve_reading(reading, r.site, ref)
        results[r.incident_id] = (fields, "text_layer", final_warnings(fields, reading, r.site, ref))

    resolved = {}
    for s in scans:
        if s.sha1 in readings:
            if not s.site:
                s.site = split_fields(readings[s.sha1]["texts"][0]).get("site", "")
            resolved[s.sha1] = resolve_reading(readings[s.sha1], s.site, ref)

    fallback = {}
    if engine == "combined":
        to_check = [s for s in scans if s.sha1 in resolved and needs_fallback(resolved[s.sha1][0], ref)]
        if to_check:
            print(f"{len(to_check)} scans à compléter avec Tesseract")
        fallback = tesseract_readings(to_check, caches["tesseract"], dpi, force, workers)

    for s in scans:
        if s.sha1 not in resolved:
            continue
        fields, notes = resolved[s.sha1]
        method = f"ocr_{primary}"
        if s.sha1 in fallback:
            fields, used = combine(fields, resolve_reading(fallback[s.sha1], s.site, ref)[0], ref)
            if used:
                method = "ocr_doctr+tesseract"
        results[s.incident_id] = (fields, method, final_warnings(fields, readings[s.sha1], s.site, ref) + notes)

    # 6. CSV complet
    full = df.set_index("incident_id")
    full["extraction_method"] = full["document_format"].map(lambda f: "csv" if f == "text" else "")
    full["extraction_warnings"] = ""
    for r in reports:
        if r.error is not None and r.incident_id not in results:
            full.at[r.incident_id, "extraction_method"] = "error"
            full.at[r.incident_id, "extraction_warnings"] = f"erreur: {r.error}"
    for incident_id, (fields, method, warnings) in results.items():
        for col, value in fields.items():
            full.at[incident_id, col] = value
        full.at[incident_id, "extraction_method"] = method
        full.at[incident_id, "extraction_warnings"] = " | ".join(warnings)
    full = full.reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("incidents_full.csv" if engine == "combined" else f"incidents_full_{engine}.csv")
    full.to_csv(out_csv, index=False)

    # Récapitulatif
    new = {e: len(c) - before[e] for e, c in caches.items()}
    print(f"\nNouvelles lectures OCR : docTR {new['doctr']}, Tesseract {new['tesseract']} "
          f"(cache : {cache_root}/)")
    print(f"Sauvegardé : {out_csv} ({len(full)} lignes)")
    print("\nMéthode d'extraction :\n" + full["extraction_method"].value_counts().to_string())
    flagged = full[full["extraction_warnings"] != ""]
    print(f"\n{len(flagged)} rapports à vérifier (colonne extraction_warnings)")
    for row in flagged.head(20).itertuples():
        print(f"  incident {row.incident_id} : {row.extraction_warnings}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--engine", choices=["combined", "doctr", "tesseract"], default="combined")
    parser.add_argument("--dpi", type=int, default=DPI, help="résolution de rendu des scans")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processus Tesseract")
    parser.add_argument("--force", action="store_true", help="ignore le cache et relit tous les scans")
    args = parser.parse_args()
    run(args.csv, args.out_dir, args.engine, args.dpi, args.force, args.workers)
