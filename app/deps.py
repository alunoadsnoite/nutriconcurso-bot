"""Dependências de autenticação e limite de requisições.

Nota de segurança importante sobre `X-API-Key`: no app mobile a chave viaja
dentro do binário, então qualquer pessoa com o APK consegue extraí-la. Ela
protege contra chamada acidental ou bot ingênuo, **não** contra alguém
mal-intencionado. A proteção de verdade é `API_KEY` no servidor somada a
autenticação por usuário e orçamento gasto por usuário.
"""

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_key:
        return

    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API inválida.",
        )


class RateLimiter:
    """Janela deslizante simples, por IP. Em memória, por processo."""

    def __init__(self, *, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, client_id: str) -> None:
        now = monotonic()
        cutoff = now - 60.0

        with self._lock:
            hits = self._hits[client_id]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self._limit:
                retry_after = int(60 - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Limite de requisições excedido. Tente novamente em instantes.",
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)


def client_ip(request: Request) -> str:
    # Só confie em X-Forwarded-For se você realmente está atrás de um proxy
    # que o rewrite; caso contrário o cliente falsifica o IP e burla o limite.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"
