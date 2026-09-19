"""Thin client for DeepSeek's OpenAI-compatible chat completions endpoint. Raw
requests, no SDK dependency — consistent with how data_pipeline/oanda_client.py
talks to OANDA.
"""
import requests

from . import config


class DeepSeekAPIError(RuntimeError):
    pass


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    config.require_credentials()
    url = f"{config.DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise DeepSeekAPIError(f"DeepSeek request failed ({resp.status_code}): {resp.text}")
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        # A 200 OK with an unexpected body (e.g. empty "choices" from content
        # filtering, or a schema change) would otherwise raise a bare
        # KeyError/IndexError with no indication of what actually went wrong.
        raise DeepSeekAPIError(
            f"DeepSeek returned 200 OK but an unexpected response shape ({e!r}): {resp.text}"
        ) from e
