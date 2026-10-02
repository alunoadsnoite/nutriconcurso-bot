"""API do tutor de concursos em Nutrição."""

import asyncio
import logging

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from app.citations import StreamCitationFilter, strip_all_citations, validate_citations
from app.config import Settings, get_settings
from app.conversation import ConversationStore
from app.deps import RateLimiter, client_ip, require_api_key
from app.openai_client import get_client
from app.prompts import SYSTEM_PROMPT, rag_system_prompt
from app.rag import RetrievalError, build_sources, format_context, retrieve
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    SourceItem,
    StoredMessage,
)
from app.streaming import MEDIA_TYPE, NO_BUFFER, sse
from app.suggestions import generate as generate_suggestions

logger = logging.getLogger("nutriconcurso")

# O SDK da OpenAI é síncrono. Consumir o stream dentro do async generator sem
# isto travaria o event loop inteiro durante a geração: enquanto um aluno
# espera a resposta, nenhum outro é atendido. `to_thread` tira a espera da
# thread do event loop.
#
# `StopIteration` não pode vazar de dentro de uma Future do asyncio, então o
# fim do stream é sinalizado por um sentinel.
_EXHAUSTED = object()


def _pull(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _EXHAUSTED

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
    user_limiter = RateLimiter(
        requests_per_minute=settings.requests_per_minute_per_user,
        namespace="user",
    )

    app = FastAPI(
        title="NutriConcurso Bot API",
        version="2.0.0",
        description="Tutor de concursos públicos na área de Nutrição, com RAG sobre legislação.",
    )
    app.state.conversation_store = conversation_store
    app.state.rate_limiter = rate_limiter
    app.state.user_limiter = user_limiter
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

    # ------------------------------------------------------------------ utils

    def _validate(message: str) -> None:
        if not message:
            raise HTTPException(status_code=400, detail="Mensagem vazia.")
        if len(message) > settings.max_message_length:
            raise HTTPException(
                status_code=422,
                detail=f"Mensagem maior que o limite de {settings.max_message_length} caracteres.",
            )

    def _charge(request: ChatRequest, http_request: Request) -> str | None:
        """Aplica os dois limites e devolve o user_id normalizado."""
        rate_limiter.check(client_ip(http_request, trust_proxy=settings.trust_proxy))
        user_id = (request.user_id or "").strip() or None
        if user_id:
            user_limiter.check(user_id)
        return user_id

    def _require_openai() -> None:
        if not settings.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servidor sem OPENAI_API_KEY configurada.",
            )

    def _gather_context(message: str, client: OpenAI) -> tuple[list, list[SourceItem], str]:
        """Busca vetorial. Falha aqui é 503, nunca resposta sem contexto."""
        chunks = retrieve(settings, message, client)
        sources = build_sources(chunks)
        context = format_context(
            chunks,
            max_chunk_chars=settings.rag_max_chunk_chars,
            max_total_chars=settings.rag_max_context_chars,
        )
        return chunks, sources, context

    def _prompt_for(context: str, chunks: list) -> str:
        return rag_system_prompt(context) if chunks else SYSTEM_PROMPT

    # --------------------------------------------------------------- endpoints

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            configured=settings.is_configured,
            model=settings.openai_model,
            rag_configured=settings.is_rag_configured,
            embedding_model=settings.embedding_model if settings.is_rag_configured else None,
            streaming=True,
        )

    @app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat_endpoint(
        request: ChatRequest,
        http_request: Request,
        _: Settings = Depends(get_settings),
    ) -> ChatResponse:
        message = request.message.strip()
        _validate(message)
        user_id = _charge(request, http_request)
        _require_openai()

        client: OpenAI = get_client(settings)

        chunks: list = []
        sources: list[SourceItem] = []
        context = ""
        if settings.is_rag_configured:
            try:
                chunks, sources, context = _gather_context(message, client)
            except RetrievalError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc

        history = conversation_store.snapshot(user_id, message)
        system_prompt = _prompt_for(context, chunks)

        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": system_prompt}, *history],
            temperature=0.3,
            max_tokens=settings.openai_max_tokens,
        )

        raw_reply = completion.choices[0].message.content
        if not raw_reply:
            logger.warning("A OpenAI devolveu resposta vazia para user_id=%s", user_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="A OpenAI devolveu uma resposta vazia.",
            )

        if chunks:
            reply, citations = validate_citations(raw_reply, block_count=len(chunks))
            if not reply:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="A resposta ficou vazia depois da validação de citações.",
                )
        else:
            reply, citations = strip_all_citations(raw_reply), []

        suggestions: list[str] = []
        if request.include_suggestions:
            suggestions = generate_suggestions(
                client, settings, question=message, reply=reply, history=history
            )

        conversation_store.commit(user_id, message, reply)

        return ChatResponse(
            reply=reply,
            has_history=user_id is not None,
            sources=sources,
            rag_used=bool(chunks),
            citations=citations,
            suggestions=suggestions,
        )

    @app.post("/api/v1/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_stream(
        request: ChatRequest,
        http_request: Request,
        _: Settings = Depends(get_settings),
    ) -> StreamingResponse:
        message = request.message.strip()
        _validate(message)
        user_id = _charge(request, http_request)
        _require_openai()

        client: OpenAI = get_client(settings)

        chunks: list = []
        sources: list[SourceItem] = []
        context = ""
        if settings.is_rag_configured:
            try:
                chunks, sources, context = _gather_context(message, client)
            except RetrievalError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc

        history = conversation_store.snapshot(user_id, message)
        if not settings.is_rag_configured:
            # Sem RAG não há contexto; `_gather_context` não foi chamado.
            context = ""
        system_prompt = _prompt_for(context, chunks)

        async def event_stream():
            if sources:
                yield sse("sources", {"sources": [s.model_dump() for s in sources]})

            # Sem contexto não existe bloco para citar, então qualquer
            # [[fonte:N]] que aparecer é inválido por definição. O filtro
            # resolve isso na hora, sem esperar o fim da resposta.
            citations_filter = StreamCitationFilter(block_count=len(chunks))

            try:
                stream = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=settings.openai_model,
                    messages=[{"role": "system", "content": system_prompt}, *history],
                    temperature=0.3,
                    max_tokens=settings.openai_max_tokens,
                    stream=True,
                )
                iterator = iter(stream)
                while True:
                    piece = await asyncio.to_thread(_pull, iterator)
                    if piece is _EXHAUSTED:
                        break
                    if not piece.choices:
                        continue
                    delta = piece.choices[0].delta
                    if not (delta and delta.content):
                        continue

                    # O texto que o aluno vê passa pelo mesmo filtro de
                    # citações da rota sem stream. Sem isso, um marcador
                    # inválido já estaria na tela quando se descobre, no fim,
                    # que aponta para um bloco inexistente.
                    safe = citations_filter.push(delta.content)
                    if safe:
                        yield sse("token", {"t": safe})
            except OpenAIError as exc:
                logger.warning("Falha da OpenAI no stream (%s): %s", type(exc).__name__, exc)
                yield sse("error", {"detail": "Falha durante a geração da resposta."})
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Erro inesperado no stream", exc_info=exc)
                yield sse("error", {"detail": "Erro interno durante o stream."})
                return

            cauda = citations_filter.flush()
            if cauda:
                yield sse("token", {"t": cauda})

            # O que vai para o histórico é exatamente o que foi exibido.
            final_reply = citations_filter.text
            citations = citations_filter.used

            if final_reply:
                conversation_store.commit(user_id, message, final_reply)

            suggestions: list[str] = []
            if request.include_suggestions:
                suggestions = generate_suggestions(
                    client, settings, question=message, reply=final_reply, history=history
                )
            if suggestions:
                yield sse("suggestions", {"suggestions": suggestions})

            yield sse("done", {"citations": citations})

        return StreamingResponse(
            event_stream(),
            media_type=MEDIA_TYPE,
            headers=NO_BUFFER,
        )

    @app.get("/api/v1/chat", response_model=HistoryResponse, dependencies=[Depends(require_api_key)])
    async def get_conversation(
        user_id: str,
        _: Settings = Depends(get_settings),
    ) -> HistoryResponse:
        """Histórico guardado, para o app restaurar a tela."""
        stored = conversation_store.history(user_id)
        messages = [
            StoredMessage(role=m["role"], content=m["content"]) for m in stored
        ]
        return HistoryResponse(messages=messages)

    @app.delete("/api/v1/chat", dependencies=[Depends(require_api_key)])
    async def clear_conversation(
        user_id: str,
        _: Settings = Depends(get_settings),
    ) -> dict[str, bool]:
        """Limpa o histórico de um usuário (usado pelo botão 'Nova conversa')."""
        return {"cleared": conversation_store.clear_user(user_id)}

    return app


app = create_app()