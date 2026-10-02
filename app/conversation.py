"""Histórico de conversa em memória, com TTL e limite por usuário.

O ponto delicateiro é **quando** gravar. A mensagem do usuário só entra no
histórico depois que a resposta foi gerada com sucesso:

- Gravar antes e falhar deixa uma mensagem órfã no histórico. O próximo retry
  do app grava a mesma pergunta de novo, então o modelo passa a ver a pergunta
  duplicada e duas mensagens de usuário seguidas sem assistente no meio — que é
  o formato que mais degrada a instrução, e ainda paga token duas vezes.
- Por isso o fluxo é `snapshot` (monta o que será enviado, sem gravar) e
  `commit` (grava o par usuário/assistente de uma vez, só no caminho feliz).

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

Role = str


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

    def _live(self, user_id: str, now: float) -> _Conversation:
        """Conversa existente e não expirada, ou uma nova. Caller segura o lock."""
        conversation = self._conversations.get(user_id)
        if conversation is None or now - conversation.last_seen > self._ttl_seconds:
            conversation = _Conversation()
            self._conversations[user_id] = conversation
        return conversation

    def snapshot(self, user_id: str | None, message: str) -> list[dict[str, str]]:
        """Histórico que será enviado ao modelo, com a pergunta já no fim.

        Não grava nada. Se a chamada ao modelo falhar depois, o histórico fica
        intacto e um retry não duplica nada.
        """
        pending = {"role": "user", "content": message}
        if not user_id:
            return [pending]

        now = monotonic()
        with self._lock:
            self._evict_expired(now)
            conversation = self._live(user_id, now)
            return list(conversation.messages) + [pending]

    def commit(self, user_id: str | None, user_message: str, assistant_message: str) -> None:
        """Grava o par usuário/assistente. Chamado só depois da resposta boa."""
        if not user_id:
            return

        now = monotonic()
        with self._lock:
            self._evict_expired(now)
            conversation = self._live(user_id, now)
            conversation.last_seen = now
            conversation.messages.append({"role": "user", "content": user_message})
            conversation.messages.append({"role": "assistant", "content": assistant_message})
            conversation.messages = conversation.messages[-self._max_messages :]
            self._conversations.move_to_end(user_id)

    def history(self, user_id: str) -> list[dict[str, str]]:
        """Histário guardado, para o app restaurar a tela. Não inclui pendência."""
        now = monotonic()
        with self._lock:
            self._evict_expired(now)
            conversation = self._conversations.get(user_id)
            if conversation is None or now - conversation.last_seen > self._ttl_seconds:
                return []
            return list(conversation.messages)

    def clear_user(self, user_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(user_id.strip(), None) is not None

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()