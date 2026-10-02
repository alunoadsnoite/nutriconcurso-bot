"""Resposta em streaming via Server-Sent Events.

Por que SSE e não WebSocket: o fluxo é de mão única (cliente pergunta, servidor
responde). SSE roda sobre HTTP comum, atravessa proxy sem upgrade de protocolo
e cabe em `fetch` normal.

Formato dos eventos, um por linha `event: <nome>` seguido de `data: <json>`:

    event: sources     -> {"sources": [...]}   chega antes do texto
    event: token       -> {"t": "trecho"}       repetido
    event: suggestions -> {"suggestions": [...]}
    event: done        -> {"citations": [1, 3]}
    event: error       -> {"detail": "..."}

Motivo de `sources` vir antes do texto: o app mostra os documentos consultados
enquanto o texto ainda está chegando, o que dá a sensação de "está consultando
a legislação".

No fim, o stream sempre fecha com `done` ou `error`. Nunca fica aberto.
"""

import json
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

# Mantém o proxy de CDN/NGINX de segurar o stream. Sem isto, respostas longas
# podem ficar em buffer e o cliente receber tudo de uma vez no fim.
NO_BUFFER = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
MEDIA_TYPE = "text/event-stream"


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def token_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Transforma um fluxo de pedaços de texto em eventos SSE."""
    async for piece in chunks:
        if piece:
            yield sse("token", {"t": piece})