"""Embeddings Azure OpenAI des descriptions, avec cache indexé par le contenu du texte.

Un texte déjà vectorisé n'est jamais renvoyé à l'API, même si l'ordre ou le nombre d'incidents
change. Le cache est sauvegardé après chaque lot : une interruption (quota, réseau) ne fait
rien perdre.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def azure_client():
    """Client Azure OpenAI configuré par le .env (voir .env.example)."""
    try:  # réseau d'entreprise avec inspection SSL : magasin de certificats de l'OS
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import find_dotenv, load_dotenv
    from openai import AzureOpenAI

    load_dotenv(find_dotenv(usecwd=True), override=True)
    try:
        client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                             api_version=os.environ["AZURE_OPENAI_API_VERSION"])
        return client, os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    except KeyError as e:
        raise SystemExit(f"Variable {e} absente du .env (voir .env.example).") from e


def embed_batch(client, deployment: str, batch: list[str], max_attempts: int = 10) -> list[list[float]]:
    """Un lot ; sur erreur 429, attend le délai indiqué par Azure puis réessaie."""
    from openai import RateLimitError

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.with_options(max_retries=0).embeddings.create(model=deployment, input=batch)
            return [item.embedding for item in response.data]
        except RateLimitError as e:
            if attempt == max_attempts:
                raise
            match = re.search(r"retry after (\d+) seconds", str(e))
            wait = int(match.group(1)) + 1 if match else 30
            print(f"Quota embeddings atteint (429), attente {wait} s ({attempt}/{max_attempts})")
            time.sleep(wait)


def embed_texts(texts: list[str], cache_root: Path) -> np.ndarray:
    """Matrice des embeddings de `texts` (une ligne par texte), en n'appelant l'API que pour les
    textes absents du cache."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=True)
    # Un cache par modèle : des vecteurs de modèles différents ne sont pas comparables.
    cache = EmbeddingCache(cache_root, os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embeddings"))
    todo = cache.missing(texts)
    print(f"Embeddings : {len(texts) - len(todo)} en cache, {len(todo)} à calculer")
    if todo:
        client, deployment = azure_client()
        for i in tqdm(range(0, len(todo), EMBEDDING_BATCH_SIZE), desc="Embeddings", unit="lot"):
            batch = todo[i:i + EMBEDDING_BATCH_SIZE]
            for text, vector in zip(batch, embed_batch(client, deployment, batch)):
                cache.vectors[text_key(text)] = np.asarray(vector, dtype=np.float32)
            cache.save()
    return cache.matrix(texts)
