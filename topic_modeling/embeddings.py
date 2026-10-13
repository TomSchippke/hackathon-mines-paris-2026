"""Embeddings Azure OpenAI des descriptions, avec cache indexé par le contenu du texte.

Un texte déjà vectorisé n'est jamais renvoyé à l'API, même si l'ordre ou le nombre d'incidents
change. Le cache est sauvegardé après chaque lot : une interruption (quota, réseau) ne fait
rien perdre.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llm.client import azure_client, env, with_retry

from .config import EMBEDDING_BATCH_SIZE


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """{sha1 du texte: vecteur}, stocké dans outputs/cache/embeddings/<déploiement>.npz."""

    def __init__(self, root: Path, deployment: str):
        self.path = root / "embeddings" / f"{deployment}.npz"
        self.vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self.vectors = dict(zip(data["keys"], data["vectors"]))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.vectors)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vectors=np.array([self.vectors[k] for k in keys]))
        tmp.replace(self.path)

    def missing(self, texts: list[str]) -> list[str]:
        return list(dict.fromkeys(t for t in texts if text_key(t) not in self.vectors))

    def matrix(self, texts: list[str]) -> np.ndarray:
        return np.array([self.vectors[text_key(t)] for t in texts])


def embed_batch(batch: list[str]) -> list[list[float]]:
    deployment = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    response = with_retry(lambda: azure_client().embeddings.create(model=deployment, input=batch))
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str], cache_root: Path) -> np.ndarray:
    """Matrice des embeddings de `texts` (une ligne par texte), en n'appelant l'API que pour les
    textes absents du cache."""
    # Un cache par modèle : des vecteurs de modèles différents ne sont pas comparables.
    cache = EmbeddingCache(cache_root, env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
    todo = cache.missing(texts)
    print(f"Embeddings : {len(texts) - len(todo)} en cache, {len(todo)} à calculer")
    if todo:
        for i in tqdm(range(0, len(todo), EMBEDDING_BATCH_SIZE), desc="Embeddings", unit="lot"):
            batch = todo[i:i + EMBEDDING_BATCH_SIZE]
            for text, vector in zip(batch, embed_batch(batch)):
                cache.vectors[text_key(text)] = np.asarray(vector, dtype=np.float32)
            cache.save()
    return cache.matrix(texts)
