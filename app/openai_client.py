"""Cliente OpenAI criado sob demanda.

A construção é preguiçosa de propósito: `OpenAI()` valida a chave no
construtor, e fazer isso no import derrubaria o processo quando a variável de
ambiente não estivesse defined.
"""

from functools import lru_cache

from openai import OpenAI

from app.config import Settings


@lru_cache
def _build_client(*, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=1)


def get_client(settings: Settings) -> OpenAI:
    if not settings.is_configured:
        raise RuntimeError("OPENAI_API_KEY não configurada")

    return _build_client(
        api_key=settings.openai_api_key or "",
        timeout=settings.openai_timeout_seconds,
    )
