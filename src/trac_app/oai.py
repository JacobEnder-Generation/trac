from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from .config import OAI_PROMPT
from .env import ENV_PATH, get_env_value

LOGGER = logging.getLogger(__name__)

DEFAULT_OAI_MODEL = "gpt-4o-mini"

class OpenAIIntegrationError(RuntimeError):
    """Raised when the OpenAI integration cannot be used."""


@lru_cache(maxsize=1)
def _load_openai_key() -> str:
    for candidate in ("OPENAI_API_KEY", "OPENAI_KEY", "OAI_KEY"):
        key = get_env_value(candidate)
        if key:
            return key
    raise OpenAIIntegrationError(
        f"OpenAI API key not found in {ENV_PATH}. Expected OPENAI_API_KEY."
    )


@lru_cache(maxsize=1)
def _get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise OpenAIIntegrationError(
            "OpenAI python package is not installed. Install 'openai' to use this feature."
        ) from exc

    api_key = _load_openai_key()
    return OpenAI(api_key=api_key)


def build_prompt(ticker: str) -> str:
    segments: list[str] = []
    if OAI_PROMPT:
        segments.append(OAI_PROMPT[0])
        segments.append(ticker)
        if len(OAI_PROMPT) > 1:
            segments.extend(OAI_PROMPT[1:])
    else:
        segments.append(ticker)
    return " ".join(seg.strip() if isinstance(seg, str) else str(seg) for seg in segments).strip()


def _extract_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
                elif isinstance(text_val, list):
                    parts.extend(str(segment) for segment in text_val)
                else:
                    content_val = item.get("content")
                    if isinstance(content_val, str):
                        parts.append(content_val)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


@lru_cache(maxsize=128)
def _fetch_analysis_cached(ticker: str, model: str) -> str:
    client = _get_client()
    prompt = build_prompt(ticker)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # pragma: no cover - API failure path
        LOGGER.exception("OpenAI request failed for ticker %s", ticker)
        raise OpenAIIntegrationError(f"OpenAI request failed: {exc}") from exc

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError) as exc:
        raise OpenAIIntegrationError(
            "Unexpected OpenAI response format."
        ) from exc

    text = _extract_message_content(getattr(message, "content", None))
    return text.strip()


def fetch_analysis(ticker: str, *, model: Optional[str] = None) -> str:
    model_name = model or DEFAULT_OAI_MODEL
    result = _fetch_analysis_cached(ticker.upper(), model_name)
    if not result:
        raise OpenAIIntegrationError("Empty analysis returned from OpenAI.")
    return result


def is_openai_available() -> bool:
    try:
        _get_client()
        return True
    except OpenAIIntegrationError:
        return False
