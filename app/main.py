"""API do tutor de concursos em Nutrição."""

import logging

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from app.config import Settings, get_settings
from app.conversation import ConversationStore
from app.deps import RateLimiter, client_ip, require_api_key
from app.openai_client import get_client
from app.prompts import SYSTEM_PROMPT, rag_system_prompt
from app.rag import RetrievalError, build_sources, format_context, retrieve
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("nutriconcurso")

# Erros da OpenAI são traduzidos para status HTTP estáveis. O texto original
# vai para o log do servidor e nunca para o corpo da resposta: ele costuma
# carregar nomes de projeto, ids de requisição e trechos de prompt.
_UPSTREAM_ERRORS = (
    (AuthenticationError, status.HTTP_502_BAD_GATEWAY, "Credencial da OpenAI inválida no servidor."),
    (RateLimitError, status.HTTP_429_TOO_MANY_REQUESTS, "Limite de uso da OpenAI atingido. Tente novamente mais tarde."),
    (APITimeoutError, status.HTTP_504_GATEWAY_TIMEOUT, "A OpenAI demorou para responder."),
    (APIConnectionError, status.HTTP_502_BAD_GATEWAY, "Não foi possível alcançar a OpenAI."),
    (APIStatusError, status.HTTP_502_BAD_GATEWAY, "A OpenAI recusou a requisição."),
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    conversation_store = ConversationStore(
        max_messages=settings.max_history_messages,
        ttl_seconds=settings.history_ttl_seconds,
    )
    rate_limiter = RateLimiter(requests_per_minute=settings.requests_per_minute)

    app = FastAPI(
        title="NutriConcurso Bot API",
        version="1.0.0",
        description="Tutor de concursos públicos na área de Nutrição.",
    )
    app.state.conversation_store = conversation_store
    app.state.rate_limiter = rate_limiter
    app.state.settings = settings

    # `get_settings` é `lru_cache`d, então `Depends(get_settings)` devolveria uma
    # instância nova lida do ambiente em vez deste objeto. Sem esta linha, as
    # rotas e o `require_api_key` enxergariam configurações diferentes entre si
    # e a checagem de chave poderia ser ignorada silenciosamente.
    app.dependency_overrides[get_settings] = lambda: settings

    @app.exception_handler(OpenAIError)
    async def openai_error_handler(_: Request, exc: OpenAIError) -> JSONResponse:
        for exc_type, http_status, detail in _UPSTREAM_ERRORS:
            if isinstance(exc, exc_type):
                logger.warning("Falha da OpenAI (%s): %s", exc_type.__name__, exc)
                return JSONResponse(status_code=http_status, content={"detail": detail})

        logger.exception("Erro inesperado da OpenAI", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "Falha ao consultar a OpenAI."},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            configured=settings.is_configured,
            model=settings.openai_model,
            rag_configured=settings.is_rag_configured,
            embedding_model=settings.embedding_model if settings.is_rag_configured else None,
        )

    @app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat_endpoint(
        request: ChatRequest,
        http_request: Request,
        _: Settings = Depends(get_settings),
    ) -> ChatResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Mensagem vazia.")

        if len(message) > settings.max_message_length:
            raise HTTPException(
                status_code=422,
                detail=f"Mensagem maior que o limite de {settings.max_message_length} caracteres.",
            )

        rate_limiter.check(client_ip(http_request))

        if not settings.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servidor sem OPENAI_API_KEY configurada.",
            )

        # Mesmo cliente para o embedding e para a resposta: uma conexão, um
        # timeout, uma configuração.
        client: OpenAI = get_client(settings)

        # Busca vetorial antes de chamar o modelo. Uma falha aqui NÃO é
        # degradada para resposta sem contexto: nesse modo o tutor falaria
        # memória do modelo parecendo fundamento oficial, que é exatamente o
        # risco que o RAG existe para evitar. Melhor 503 e o app avisar.
        context = ""
        sources: list = []
        rag_used = False
        if settings.is_rag_configured:
            try:
                chunks = retrieve(settings, message, client)
            except RetrievalError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc

            context = format_context(
                chunks,
                max_chunk_chars=settings.rag_max_chunk_chars,
                max_total_chars=settings.rag_max_context_chars,
            )
            sources = build_sources(chunks)
            rag_used = bool(chunks)

        system_prompt = rag_system_prompt(context) if rag_used else SYSTEM_PROMPT

        user_id = (request.user_id or "").strip() or None
        history = conversation_store.append_user_message(user_id, message)

        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": system_prompt}, *history],
            temperature=0.3,
            max_tokens=settings.openai_max_tokens,
        )

        reply = completion.choices[0].message.content
        if not reply:
            logger.warning("A OpenAI devolveu resposta vazia para user_id=%s", user_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="A OpenAI devolveu uma resposta vazia.",
            )

        conversation_store.append_assistant_message(user_id, reply)

        return ChatResponse(
            reply=reply,
            has_history=user_id is not None,
            sources=sources,
            rag_used=rag_used,
        )

    @app.delete("/api/v1/chat", dependencies=[Depends(require_api_key)])
    async def clear_conversation(
        user_id: str,
        _: Settings = Depends(get_settings),
    ) -> dict[str, bool]:
        """Limpa o histórico de um usuário (usado pelo botão 'Nova conversa')."""
        store: ConversationStore = app.state.conversation_store
        return {"cleared": store.clear_user(user_id)}

    return app


app = create_app()
