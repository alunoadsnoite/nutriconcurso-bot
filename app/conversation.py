"""Histórico de conversa em memória, com TTL e limite por usuário.

Limitações assumidas e intencionais para um MVP de instância única:
- o histórico vive no processo e se perde quando o servidor reinicia;
- com mais de um worker (uvicorn --workers N) cada um tem a sua própria
  memória, então a conversa "pula" entre requisições.

Para produção isso precisa ir para Redis ou Postgres. Está anotado no README.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


@dataclass
class _Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    last_seen: float = field(default_factory=monotonic)


class ConversationStore:
    def __init__(self, *, max_messages: int, ttl_seconds: int, max_users: int = 1000) -> None:
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds
        self._max_users = max_users
        self._conversations: OrderedDict[str, _Conversation] = OrderedDict()
        self._lock = Lock()

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, v in self._conversations.items() if now - v.last_seen > self._ttl_seconds]
        for key in expired:
            del self._conversations[key]

        while len(self._conversations) > self._max_users:
            self._conversations.popitem(last=False)

    def append_user_message(self, user_id: str | None, message: str) -> list[dict[str, str]]:
        """Registra a mensagem do usuário e devolve o histórico para enviar."""
        if not user_id:
            return [{"role": "user", "content": message}]

        now = monotonic()
        with self._lock:
            self._evict_expired(now)
            conversation = self._conversations.get(user_id)

            if conversation is None or now - conversation.last_seen > self._ttl_seconds:
                conversation = _Conversation()
                self._conversations[user_id] = conversation

            conversation.last_seen = now
            conversation.messages.append({"role": "user", "content": message})
            conversation.messages = conversation.messages[-self._max_messages :]
            self._conversations.move_to_end(user_id)

            return list(conversation.messages)

    def append_assistant_message(self, user_id: str | None, message: str) -> None:
        if not user_id:
            return

        now = monotonic()
        with self._lock:
            conversation = self._conversations.get(user_id)
            if conversation is None:
                return

            conversation.last_seen = now
            conversation.messages.append({"role": "assistant", "content": message})
            conversation.messages = conversation.messages[-self._max_messages :]

    def clear_user(self, user_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(user_id.strip(), None) is not None

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()
