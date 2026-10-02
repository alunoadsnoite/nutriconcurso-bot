"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Opcional: sem ele a conversa não é guardada entre requisições.
    user_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    reply: str
    # `False` quando o histórico foi descartado (TTL expirado, app reiniciado,
    # ou chamada sem `user_id`). O cliente usa isso para não prometer memória.
    has_history: bool = False


class HealthResponse(BaseModel):
    status: str
    configured: bool
    model: str
