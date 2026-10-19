"""Client Azure OpenAI avec cache persistant sur disque et gestion de backoff."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI, RateLimitError

from .config import CACHE_DIR


def get_azure_chat_client() -> tuple[AzureOpenAI, str]:
    """Initialise le client Azure OpenAI pour le chat et retourne (client, deployment_name)."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    load_dotenv(find_dotenv(usecwd=True), override=True)
    try:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        return client, deployment
    except KeyError as e:
        raise SystemExit(f"Variable d'environnement {e} absente du fichier .env.") from e


def _hash_request(system_prompt: str, user_prompt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(b"|||")
    hasher.update(user_prompt.encode("utf-8"))
    return hasher.hexdigest()


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    cache_tag: str = "general",
    max_retries: int = 5,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Appelle le LLM avec format de sortie JSON garanti et cache disque."""
    cache_dir = CACHE_DIR / cache_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    req_hash = _hash_request(system_prompt, user_prompt)
    cache_file = cache_dir / f"{req_hash}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # En cas de fichier corrompu, on recalcule

    client, deployment = get_azure_chat_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.with_options(max_retries=0).chat.completions.create(
                model=deployment,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)

            # Sauvegarde atomique dans le cache
            tmp_file = cache_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            tmp_file.replace(cache_file)
            return parsed

        except RateLimitError as e:
            if attempt == max_retries:
                raise
            match = re.search(r"retry after (\d+) seconds", str(e), re.IGNORECASE)
            wait_time = int(match.group(1)) + 1 if match else (2 ** attempt * 2)
            print(f"  [Quota 429] Attente de {wait_time} s avant nouvelle tentative ({attempt}/{max_retries})...")
            time.sleep(wait_time)
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                raise ValueError(f"Réponse JSON invalide du LLM : {raw_content}") from e
            time.sleep(1)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  [Erreur LLM] {e}, réessai ({attempt}/{max_retries})...")
            time.sleep(2)

    return {}
