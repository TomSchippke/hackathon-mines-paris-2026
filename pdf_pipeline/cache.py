"""Cache des lectures OCR, indexé par le contenu du PDF.

Chaque lecture est stockée dans `<cache>/<moteur>/<sha1 du PDF>.json`. Conséquences :
- un nouveau PDF (nouvelle empreinte) est lu, un PDF déjà lu ne l'est jamais deux fois ;
- un PDF modifié a une nouvelle empreinte, il est donc relu automatiquement ;
- renommer un PDF ou changer son identifiant dans le CSV ne force pas de relecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class OcrCache:
    def __init__(self, root: Path, engine: str):
        self.dir = root / engine
        self.dir.mkdir(parents=True, exist_ok=True)

    def _file(self, sha1: str) -> Path:
        return self.dir / f"{sha1}.json"

    def get(self, sha1: str) -> dict | None:
        f = self._file(sha1)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def put(self, sha1: str, reading: dict) -> None:
        tmp = self._file(sha1).with_suffix(".tmp")
        tmp.write_text(json.dumps(reading, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._file(sha1))  # écriture atomique : pas de fichier à moitié écrit si on interrompt

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*.json"))
