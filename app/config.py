"""Configuração da aplicação, lida de variáveis de ambiente.

Nada de segredo tem valor padrão no código: a aplicação sobe sem
`OPENAI_API_KEY` e apenas recusa as requisições de chat, o que evita o
`OpenAI()` explodir no import e derrubar o processo inteiro.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 30.0
    openai_max_tokens: int = 1200

    # Opcional. Quando vazio, a API responde sem exigir cabeçalho de chave.
    # Leia a nota de segurança do README antes de expor a API publicamente.
    api_key: str | None = None

    max_message_length: int = 4000
    max_history_messages: int = 12
    history_ttl_seconds: int = 3600

    requests_per_minute: int = 20

    @property
    def is_configured(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
