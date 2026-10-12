"""Moteur OCR Tesseract : quatre lectures indépendantes par scan, plus la relecture des cellules
numériques. Les lectures sont départagées ensuite par vote (voir resolve.py).

Prérequis : `brew install tesseract tesseract-lang` (macOS) ou
`apt install tesseract-ocr tesseract-ocr-fra` (Linux).
"""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

# Tesseract tourne dans plusieurs processus : un seul thread chacun évite la sur-souscription CPU.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import pandas as pd
import pymupdf
from PIL import Image, ImageFilter, ImageOps

from .config import BINARIZE_THRESHOLD, DECIMALS, FIELD_LABELS
from .parsing import match_label


def binarize(img: Image.Image) -> Image.Image:
    """Débruitage + binarisation : supprime le grain du papier scanné."""
    img = ImageOps.autocontrast(img.filter(ImageFilter.MedianFilter(3)))
    return img.point(lambda p: 255 if p > BINARIZE_THRESHOLD else 0)


def tesseract(img: Image.Image, *args: str) -> str:
    buf = io.BytesIO()
    img.save(buf, "png")
    result = subprocess.run(["tesseract", "stdin", "stdout", "-l", "fra", *args],
                            input=buf.getvalue(), capture_output=True, check=True)
    return result.stdout.decode("utf-8")


def render_gray(page: pymupdf.Page, dpi: int) -> Image.Image:
    return Image.open(io.BytesIO(page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).tobytes("png")))


def tsv_lines(img: Image.Image) -> list[pd.DataFrame]:
    """Lignes de texte lues par Tesseract (psm 6), chacune avec la position de ses mots."""
    tsv = pd.read_csv(io.StringIO(tesseract(img, "--psm", "6", "tsv")), sep="\t", quoting=3,
                      keep_default_na=False, dtype={"text": str})
    tsv = tsv[(tsv["conf"].astype(float) >= 0) & (tsv["text"].str.strip() != "")]
    return [line.sort_values("left") for _, line in tsv.groupby(["block_num", "par_num", "line_num"])]


def ocr_binarized(page: pymupdf.Page, dpi: int) -> str:
    """OCR sur image binarisée, lue ligne à ligne (psm 6)."""
    return tesseract(binarize(render_gray(page, dpi)), "--psm", "6")


def ocr_main(page: pymupdf.Page, dpi: int) -> tuple[str, dict[str, str]]:
    """Lecture principale (image binarisée) + relecture ciblée des cellules numériques.

    Chaque valeur numérique est recadrée, agrandie et relue seule en n'autorisant que des chiffres :
    cela donne des candidats supplémentaires pour le contrôle de cohérence coût/durée.
    """
    gray = render_gray(page, dpi)
    lines = tsv_lines(binarize(gray))
    text = "\n".join(" ".join(line["text"]) for line in lines)
    cells = {}
    for line in lines:
        line_text = " ".join(line["text"])
        for col in DECIMALS:
            if col in cells or len(line) < 2 or match_label(line_text, FIELD_LABELS[col]) is None:
                continue
            # La valeur est après le plus grand espace horizontal de la ligne (colonne du tableau).
            gaps = line["left"].values[1:] - (line["left"] + line["width"]).values[:-1]
            value = line.iloc[gaps.argmax() + 1:]
            box = (value["left"].min() - 15, value["top"].min() - 15,
                   (value["left"] + value["width"]).max() + 15, (value["top"] + value["height"]).max() + 15)
            crop = gray.crop(box)
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            cells[col] = tesseract(binarize(crop), "--psm", "7",
                                   "-c", "tessedit_char_whitelist=0123456789.").strip()
    return text, cells


def ocr_pymupdf(page: pymupdf.Page, dpi: int) -> str:
    """Lecture sur image brute, segmentation automatique de Tesseract via PyMuPDF."""
    return page.get_text(textpage=page.get_textpage_ocr(language="fra", dpi=dpi, full=True))


def read_scan(path: str, dpi: int) -> dict:
    """Lectures brutes d'un scan. Exécuté dans un processus séparé (une lecture ≈ 1 s de CPU)."""
    with pymupdf.open(path) as doc:
        main_pages = [ocr_main(page, dpi) for page in doc]
        texts = ["\n".join(text for text, _ in main_pages),  # la plus fiable : en premier
                 "\n".join(ocr_pymupdf(page, dpi) for page in doc),
                 "\n".join(ocr_binarized(page, dpi * 2 // 3) for page in doc),
                 "\n".join(ocr_binarized(page, dpi * 4 // 3) for page in doc)]
    cells = {k: v for _, page_cells in main_pages for k, v in page_cells.items()}
    return {"engine": "tesseract", "texts": texts, "cells": cells}
