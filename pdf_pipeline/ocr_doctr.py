"""Moteur OCR docTR (deep learning, https://github.com/mindee/doctr) : une lecture par scan.

Prérequis : `pip install "python-doctr[torch]"`. Utilise le GPU (Mac : MPS, sinon CUDA) si
disponible ; sur CPU, compter ~7 s par page au lieu de ~3 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf


def page_text(page) -> str:
    """Texte d'une page docTR : une ligne par ligne détectée, dans l'ordre de lecture.

    Les lignes à la même hauteur (libellé puis valeur d'une ligne du tableau) sont gardées
    séparées mais ordonnées de gauche à droite, comme le texte d'un PDF digital.
    """
    lines = []
    for block in page.blocks:
        for line in block.lines:
            # Rectangle ((x0, y0), (x1, y1)) ou polygone (4 points) selon les options du modèle.
            pts = np.asarray(line.geometry)
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            lines.append(((y0 + y1) / 2, y1 - y0, pts[:, 0].min(), " ".join(w.value for w in line.words)))
    rows: list[list] = []
    for y, height, x, text in sorted(lines):
        if rows and abs(y - rows[-1][0]) < height / 2:
            rows[-1][1].append((x, text))
        else:
            rows.append([y, [(x, text)]])
    return "\n".join(text for _, parts in rows for _, text in sorted(parts))


class DoctrReader:
    """Modèle chargé au premier scan seulement : rien n'est chargé si tout est déjà en cache."""

    def __init__(self, dpi: int):
        self.dpi = dpi
        self.model = None
        self.device = None

    def _load(self) -> None:
        try:
            import torch
            from doctr.models import ocr_predictor
        except ImportError as e:
            raise SystemExit('docTR introuvable : pip install "python-doctr[torch]"') from e
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        # Redressement des pages : les scans sont souvent légèrement penchés, ce qui mélange
        # l'ordre des lignes si on les lit telles quelles.
        self.model = ocr_predictor(pretrained=True, assume_straight_pages=False,
                                   straighten_pages=True, detect_orientation=True)
        self.model = self.model.to(torch.device(self.device))

    def read_scan(self, path: Path) -> dict:
        if self.model is None:
            self._load()
        with pymupdf.open(path) as doc:
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=self.dpi, colorspace=pymupdf.csRGB)
                images.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3))
        pages = self.model(images).pages
        return {"engine": "doctr", "texts": ["\n".join(page_text(p) for p in pages)]}
