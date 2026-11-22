import time 
import os
import requests
from functools import lru_cache
import hashlib
import json
from collections import OrderedDict
from src.prompts import SYSTEM_PROMPT, build_user_prompt

@lru_cache(maxsize=100)
def _cached_answer(cache_key):
    return _answer_cache_store[cache_key]


_answer_cache_store = {}

_ANSWER_CACHE = OrderedDict()
_CACHE_MAX_SIZE = 100

def _sources(chunks):
    seen = set()
    sources = []

    for chunk in chunks:
        url = chunk.get("source_url", "")
        title = chunk.get("source_title", "Official source")

        if url and url not in seen:
            seen.add(url)

            sources.append({
                "source_title": title,
                "source_url": url
            })

    return sources


def _extract_answer(data):

    choices = data.get("choices") if isinstance(data, dict) else None

    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("The API returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None

    if isinstance(content, list):
        text_parts = []

        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        content = "\n".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The LLM returned an empty answer.")

    return content.strip()


def generate_answer(
    query,
    retrieved_chunks,
    eligibility_result=None,
    language="auto"
):

    if not retrieved_chunks:
        return {
            "answer": "No relevant scheme information was found.",
            "language": language,
            "sources": [],
            "disclaimer": (
                "Please consult the relevant official government portal."
            )
        }

    sources = _sources(retrieved_chunks)

    cache_data = {
        "query": query.strip().casefold(),
        "language": language,
        "chunks": [
            {
                "text": chunk.get("text", ""),
                "source_title": chunk.get("source_title", ""),
                "scheme_name": chunk.get("scheme_name", ""),
                "source_url": chunk.get("source_url", "")
            }
            for chunk in retrieved_chunks
        ],
        "eligibility": eligibility_result
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_data,
            sort_keys=True,
            ensure_ascii=False,
            default=str
        ).encode("utf-8")
    ).hexdigest()

    if cache_key in _ANSWER_CACHE:
        print("CACHE HIT: Reusing previous Gemini answer", flush=True)

        _ANSWER_CACHE.move_to_end(cache_key)

        return _ANSWER_CACHE[cache_key].copy()

    print("CACHE MISS: Generating a new Gemini answer", flush=True)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        return {
            "error": True,
            "answer": (
                "Relevant scheme information was retrieved, "
                "but the LLM API is not configured. "
                "Configure LLM_API_KEY, LLM_BASE_URL and LLM_MODEL."
            ),
            "language": language,
            "sources": sources,
            "disclaimer": "Preliminary guidance only."
        }

    url = (
        base_url
        if base_url.endswith("/chat/completions")
        else base_url + "/chat/completions"
    )

    try:
        user_prompt = build_user_prompt(
            query,
            retrieved_chunks,
            eligibility_result,
            language
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 600
        }

        response = None

        for attempt in range(3):
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120
            )

            if response.status_code != 503:
                break

            if attempt < 2:
                wait_time = 2 ** (attempt + 1)

                print(
                    f"Gemini temporarily overloaded. "
                    f"Retrying in {wait_time} seconds...",
                    flush=True
                )

                time.sleep(wait_time)

        response.raise_for_status()

        data = response.json()
        answer = _extract_answer(data)

        result = {
            "answer": answer,
            "language": language,
            "sources": sources,
            "disclaimer": (
                "Preliminary guidance only. "
                "Final eligibility is decided by the relevant "
                "government authority."
            )
        }

        _ANSWER_CACHE[cache_key] = result.copy()
        _ANSWER_CACHE.move_to_end(cache_key)

        if len(_ANSWER_CACHE) > _CACHE_MAX_SIZE:
            _ANSWER_CACHE.popitem(last=False)

        return result

    except requests.exceptions.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "Unknown"
        )

        if status_code == 429:
            message = (
                "The LLM API rate limit or quota has been reached. "
                "Please wait and try again."
            )
        elif status_code in (401, 403):
            message = (
                "LLM authentication failed. Check your API key "
                "and provider permissions."
            )
        elif status_code == 404:
            message = (
                "LLM model or endpoint not found. "
                "Check LLM_MODEL and LLM_BASE_URL."
            )
        else:
            message = f"LLM API returned HTTP status {status_code}."

        return {
            "error": True,
            "answer": message,
            "language": language,
            "sources": sources,
            "disclaimer": "Please verify information using official sources."
        }

    except requests.exceptions.Timeout:
        return {
            "error": True,
            "answer": "The LLM request timed out. Please try again.",
            "language": language,
            "sources": sources,
            "disclaimer": "Please consult the official government portal."
        }

    except requests.exceptions.RequestException:
        return {
            "error": True,
            "answer": (
                "Could not connect to the LLM service. "
                "Please check your internet connection and API settings."
            ),
            "language": language,
            "sources": sources,
            "disclaimer": "Please consult the official government portal."
        }

    except (ValueError, KeyError, IndexError, TypeError):
        return {
            "error": True,
            "answer": (
                "The LLM returned an unexpected response. "
                "Please try again."
            ),
            "language": language,
            "sources": sources,
            "disclaimer": "Please verify information using official sources."
        }
