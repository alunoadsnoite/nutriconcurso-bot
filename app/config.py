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
    # Orçamento por pessoa. O limite por IP segura robô; este segura um aluno
    # (ou um app modificado) de consumir a conta inteira de uma vez. `0`
    # desliga. Só vale quando a requisição traz `user_id`.
    requests_per_minute_per_user: int = 12

    # Só liga com `trust_proxy=true` se existir um proxy que reescreva o
    # cabeçalho. Ligado sem proxy, qualquer um burla o limite mandando
    # `X-Forwarded-For` diferente a cada requisição.
    trust_proxy: bool = False

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

    # Busca híbrida: vetorial + full-text em português, combinadas por RRF.
    # Não precisa de modelo extra e acha o que a vetorial erra, quando a
    # pergunta usa vocabulário diferente do documento ("merenda" contra
    # "alimentos").
    #
    # DESLIGADA por padrão, e o motivo é honesto: ela depende da coluna
    # gerada `content_tsv` e da função `match_documents_hybrid`, que só
    # existem depois de rodar migrations/002_busca_hibrida.sql. Não há teste
    # automatizado que rode contra um Postgres de verdade aqui, então ligar
    # por padrão seria entregar um caminho não verificado no caminho crítico.
    # O comentário de verificação está no fim da própria migration.
    rag_hybrid: bool = False
    rag_rrf_k: int = 60

    # Perguntas sugeridas depois da resposta. Uma chamada extra, pequena e
    # barata, em modelo de saída JSON. `0` desliga.
    suggestions_count: int = 3
    suggestions_max_tokens: int = 150

    @property
    def is_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def is_rag_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
