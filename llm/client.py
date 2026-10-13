"""Accès Azure OpenAI partagé par toutes les étapes : client, reprise sur quota, cache des réponses.

Chaque réponse du LLM est stockée dans outputs/cache/llm/<empreinte de la requête>.json :
relancer une étape avec les mêmes entrées ne refait aucun appel (ni coût, ni attente) et redonne
exactement les mêmes réponses. Changer le prompt, le modèle ou les données change l'empreinte,
donc provoque un nouvel appel.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache

from paths import CACHE_DIR

LLM_CACHE_DIR = CACHE_DIR / "llm"


def env(name: str) -> str:
    """Variable du .env (voir .env.example)."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=True)
    if name not in os.environ:
        raise SystemExit(f"Variable {name} absente du .env (voir .env.example).")
    return os.environ[name]


@lru_cache(maxsize=1)
def azure_client():
    try:  # réseau d'entreprise avec inspection SSL : magasin de certificats de l'OS
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from openai import AzureOpenAI

    # max_retries=0 : les reprises sont gérées par with_retry, avec un message visible.
    return AzureOpenAI(api_key=env("AZURE_OPENAI_API_KEY"), azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
                       api_version=env("AZURE_OPENAI_API_VERSION"), max_retries=0)


def retry_delay(error) -> int:
    """Délai demandé par Azure après une erreur 429 : en-tête retry-after, sinon message, sinon 20 s."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
        try:
            return max(1, int(float(headers[name]) * scale) + 1)
        except (KeyError, ValueError, TypeError):
            pass
    match = re.search(r"retry after (\d+) seconds", str(error))
    return int(match.group(1)) + 1 if match else 20


def with_retry(call, max_attempts: int = 10):
    """Exécute un appel Azure ; sur quota (429) attend le délai indiqué par Azure, sur erreur
    réseau attend un peu, puis réessaie."""
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except RateLimitError as e:
            if attempt == max_attempts:
                raise
            wait = retry_delay(e)
            print(f"Quota de débit Azure atteint (429 : trop de requêtes par minute, pas un problème de "
                  f"budget), attente {wait} s puis nouvel essai ({attempt}/{max_attempts})")
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError):
            if attempt == max_attempts:
                raise
            time.sleep(5 * attempt)


def chat(messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None,
         temperature: float = 0.0) -> dict:
    """Un appel au modèle de chat, mis en cache selon le contenu exact de la requête.

    Renvoie {"content": texte ou None, "tool_calls": [{"id", "name", "arguments"}]}.
    """
    request = {"model": env("AZURE_OPENAI_CHAT_DEPLOYMENT"), "messages": messages, "temperature": temperature}
    if tools:
        request["tools"] = tools
    if response_format:
        request["response_format"] = response_format
    key = hashlib.sha1(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = LLM_CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    message = with_retry(lambda: azure_client().chat.completions.create(**request)).choices[0].message
    answer = {"content": message.content,
              "tool_calls": [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                             for c in message.tool_calls or []]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # écriture atomique : pas de fichier à moitié écrit si on interrompt
    return answer
