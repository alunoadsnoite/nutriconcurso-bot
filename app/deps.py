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
    """Janela deslizante simples, por chave. Em memória, por processo.

    Usa `len(hits) >= limit`, não `>`, e por isso o primeiro dispare de 429 vem
    na requisição `limit + 1`. Isso é intencional: `limit` pedidos podem
    passar, e o limite é de `limit` por janela, não de `limit + 1`.
    """

    def __init__(self, *, requests_per_minute: int, namespace: str = "ip") -> None:
        self._limit = requests_per_minute
        self._namespace = namespace
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, client_id: str) -> None:
        if self._limit <= 0:
            return

        now = monotonic()
        cutoff = now - 60.0
        key = f"{self._namespace}:{client_id}"

        with self._lock:
            hits = self._hits[key]
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


def client_ip(request: Request, *, trust_proxy: bool = False) -> str:
    """Endereço usado para contabilizar limite.

    `X-Forwarded-For` **só** é lido com `trust_proxy=True`. Sem isso, qualquer
    cliente manda o cabeçalho que quiser e cada requisição cai num balde
    diferente — o rate limit inteiro deixa de valer. Ative apenas se a
    aplicação estiver atrás de um proxy que **remove** o cabeçalho recebido e
    reescreve o dele (nginx `proxy_set_header X-Forwarded-For $remote_addr`;
    Cloudflare e Render já fazem isso).
    """
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "desconhecido"