"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Opcional: sem ele a conversa não é guardada entre requisições.
    user_id: str | None = Field(default=None, max_length=128)
    # Gera as perguntas de continuação (uma chamada extra, pequena).
    include_suggestions: bool = True


class SourceItem(BaseModel):
    """Um documento consultado. Alimenta os badges de fonte do app."""

    source: str
    page: int | None = None


class StoredMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[StoredMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    # `False` quando o histórico foi descartado (TTL expirado, app reiniciado,
    # ou chamada sem `user_id`). O cliente usa isso para não prometer memória.
    has_history: bool = False
    # Vazio quando o RAG está desligado ou nada passou do limiar de
    # similaridade. Nesse caso o app mostra um aviso: a resposta saiu do modelo,
    # não de uma fonte oficial.
    sources: list[SourceItem] = Field(default_factory=list)
    rag_used: bool = False
    # Blocos de contexto que o modelo realmente citou, já validados contra o
    # número de trechos recuperados. Índice 1-based, igual aos rótulos `[1]`,
    # `[2]` do contexto. Inválidos foram removidos do texto.
    citations: list[int] = Field(default_factory=list)
    # Perguntas de continuação para os chips abaixo da resposta.
    suggestions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    configured: bool
    model: str
    rag_configured: bool = False
    embedding_model: str | None = None
    streaming: bool = True