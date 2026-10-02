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

    # ---------------------------------------------------------------- RAG ---
    # Busca vetorial no Supabase (pgvector). Vazio = RAG desligado e o chat
    # responde só com o prompt base. O servidor nunca sobe sem chave: ele
    # apenas avisa em /health que o modo sem contexto está ativo.
    supabase_url: str | None = None
    # A chave `anon` basta para leitura se a RLS estiver habilitada (veja
    # migrations/001_rag.sql). A `service_role` também funciona, mas ignora a
    # RLS inteira e deve ficar restrita ao servidor.
    supabase_key: str | None = None
    supabase_timeout_seconds: float = 10.0

    embedding_model: str = "text-embedding-3-small"
    # Precisa bater com a coluna `VECTOR(n)` da tabela `documents`. O padrão
    # do text-embedding-3-small é 1536; reduza aqui E na tabela, juntos, se
    # quiser economizar (pgvector aceita dimensões menores que o máximo).
    embedding_dimensions: int = 1536

    # Limiar de similaridade de cosseno: abaixo disso o trecho é descartado.
    # 0.25 é conservador; 0.3 (como sugerido na especificação original) começa
    # a trazer trecho medíocre, e 0.5+ costuma devolver quase nada.
    rag_match_threshold: float = 0.25
    rag_match_count: int = 5

    # Teto de caracteres por trecho e no total. Sem isso, cinco páginas inteiras
    # de PDF estouram a janela de contexto e a resposta sai truncada ou cara.
    rag_max_chunk_chars: int = 1200
    rag_max_context_chars: int = 8000

    @property
    def is_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def is_rag_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
