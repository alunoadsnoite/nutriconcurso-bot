"""Busca vetorial no Supabase (pgvector) e montagem do contexto.

O fluxo é: pergunta -> embedding -> RPC `match_documents` -> trechos.

Duas decisões que valem registro:

1. **O cliente do Supabase é preguiçoso.** `create_client` levanta se a URL ou a
   chave estiverem vazias, e rodar isso no import derrubaria o processo. Sem
   configuração de RAG a API continua funcionando sem contexto.

2. **Nada de `str(exc)` na resposta.** Uma falha do PostgREST carrega a URL do
   projeto e trechos das políticas de RLS. O original vai para o log; o cliente
   recebe uma mensagem genérica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAIError

from app.config import Settings
from app.schemas import SourceItem

logger = logging.getLogger("nutriconcurso.rag")

# Rótulo usado quando o `metadata` do trecho não informa a origem.
UNKNOWN_SOURCE = "Documento oficial"


class RetrievalError(RuntimeError):
    """Falha ao consultar a base vetorial.

    A mensagem é escrita para o usuário final e não contém URL do projeto,
    chave ou detalhe do Postgres.
    """


@dataclass(frozen=True)
class RetrievedChunk:
    """Um trecho de documento recuperado, já normalizado."""

    content: str
    source: str
    page: int | None = None
    similarity: float | None = None


@lru_cache
def _build_store(*, url: str, key: str) -> object:
    from supabase import create_client

    return create_client(url, key)


def get_store(settings: Settings) -> object:
    if not settings.is_rag_configured:
        raise RetrievalError("A base vetorial não está configurada.")

    return _build_store(
        url=settings.supabase_url or "",
        key=settings.supabase_key or "",
    )


def _embed(client, settings: Settings, text: str) -> list[float]:
    """Gera o embedding da pergunta com o modelo dimensionalmente correto."""
    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )
    except OpenAIError:
        # Deixa passar: o handler global da aplicação traduz para 502/429/504
        # igual faz na chamada de chat. Engolir aqui daria 503 para o mesmo
        # problema que lá vira 502, e o app não conseguiria diferenciar.
        raise
    except Exception as exc:  # noqa: BLE001 - cliente falso, rede, driver
        logger.warning("Falha ao gerar embedding (%s): %s", type(exc).__name__, exc)
        raise RetrievalError("Não foi possível gerar o embedding da pergunta.") from exc

    vector = response.data[0].embedding
    if len(vector) != settings.embedding_dimensions:
        # Sem isso, o Postgres rejeita o RPC com um erro de tipo confuso.
        logger.error(
            "Embedding com %d dimensões, esperado %d", len(vector), settings.embedding_dimensions
        )
        raise RetrievalError("Dimensão do embedding incompatível com a base vetorial.")

    return vector


def _normalize_page(value: object) -> int | None:
    """`page` chega como int, string ou null dependendo de quem inseriu."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _normalize_metadata(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_similarity(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_to_chunk(row: object, *, threshold: float) -> RetrievedChunk | None:
    if not isinstance(row, dict):
        return None

    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    similarity = _normalize_similarity(row.get("similarity"))
    # A função SQL já filtra por limiar. Repetimos aqui para proteger contra uma
    # função desatualizada no banco devolvendo trecho fora do corte.
    if similarity is not None and similarity <= threshold:
        return None

    metadata = _normalize_metadata(row.get("metadata"))
    source = metadata.get("source") or metadata.get("title") or UNKNOWN_SOURCE

    return RetrievedChunk(
        content=content.strip(),
        source=str(source),
        page=_normalize_page(metadata.get("page")),
        similarity=similarity,
    )


def retrieve(settings: Settings, question: str, client) -> list[RetrievedChunk]:
    """Busca os trechos mais similares à pergunta.

    O cliente OpenAI vem por parâmetro para que embedding e resposta usem a
    mesma instância (e a mesma configuração de timeout) dentro da requisição.
    """
    threshold = settings.rag_match_threshold
    store = get_store(settings)

    embedding = _embed(client, settings, question)

    try:
        rpc = store.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": settings.rag_match_count,
            },
        )
        result = rpc.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha na busca vetorial (%s): %s", type(exc).__name__, exc)
        raise RetrievalError("A base vetorial está indisponível no momento.") from exc

    rows = getattr(result, "data", None) or []
    if not isinstance(rows, list):
        logger.warning("Resposta inesperada do RPC match_documents: %s", type(rows).__name__)
        return []

    chunks = [chunk for chunk in (_row_to_chunk(row, threshold=threshold) for row in rows) if chunk]
    logger.info("Busca vetorial devolveu %d trecho(s)", len(chunks))
    return chunks


def format_context(chunks: list[RetrievedChunk], *, max_chunk_chars: int, max_total_chars: int) -> str:
    """Junta os trechos em texto para o prompt, com teto de tamanho.

    A ordem é preservada: a função SQL já devolve do mais para o menos
    similar, e os primeiros trechos pesam mais no contexto.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    used = 0

    for position, chunk in enumerate(chunks, start=1):
        if used >= max_total_chars:
            break

        body = chunk.content
        if len(body) > max_chunk_chars:
            body = body[:max_chunk_chars].rstrip() + " […]"

        origin = chunk.source if chunk.page is None else f"{chunk.source} (p. {chunk.page})"
        block = f"[{position}] Fonte: {origin}\n{body}"

        if used + len(block) > max_total_chars:
            block = block[: max(0, max_total_chars - used)].rstrip()
            if not block:
                break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


def build_sources(chunks: list[RetrievedChunk]) -> list[SourceItem]:
    """Lista de fontes para os badges do app, sem repetir documento+página."""
    sources: list[SourceItem] = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceItem(source=chunk.source, page=chunk.page))

    return sources