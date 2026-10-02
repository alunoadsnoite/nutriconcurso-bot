"""Testes da API com um cliente OpenAI falso.

Não é preciso `OPENAI_API_KEY` para rodar: o cliente é substituído por um
dublê, o que exercita o caminho real da requisição (validação, rate limit,
montagem de mensagens, tratamento de erro) sem chamar a OpenAI.
"""

import time

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.prompts import SYSTEM_PROMPT


def _openai_error(cls, message: str, status_code: int = 500) -> Exception:
    """As exceções da OpenAI exigem uma resposta httpx de verdade."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls(message, response=response, body=None)


class FakeCompletions:
    def __init__(self, *, reply: str = "Resposta de teste", delay: float = 0.0, error: Exception | None = None):
        self.reply = reply
        self.delay = delay
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error

        message = type("Msg", (), {"content": self.reply})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def build_settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "sk-fake-para-teste",
        "api_key": None,
        "requests_per_minute": 1000,
        "max_history_messages": 6,
        "history_ttl_seconds": 3600,
    }
    base.update(overrides)
    return Settings(**base)


def build_client(settings: Settings, completions: FakeCompletions) -> TestClient:
    app = create_app(settings)
    # Substitui só a camada de saída; o resto do caminho continua real.
    import app.main as main_module

    original = main_module.get_client
    main_module.get_client = lambda _settings: FakeClient(completions)
    app.state.restore = lambda: setattr(main_module, "get_client", original)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def completions():
    return FakeCompletions()


@pytest.fixture
def settings():
    return build_settings()


def test_health_reporta_configuracao(settings, completions):
    with build_client(settings, completions) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["configured"] is True
    assert response.json()["model"] == "gpt-4o-mini"


def test_health_sem_chave_avisa_que_nao_esta_configurado(completions):
    settings = build_settings(openai_api_key=None)
    with build_client(settings, completions) as client:
        response = client.get("/health")

    assert response.json()["configured"] is False


def test_chat_devolve_resposta_do_modelo(settings, completions):
    completions.reply = "A RDC 216/2004 define..."

    with build_client(settings, completions) as client:
        response = client.post("/api/v1/chat", json={"message": "O que é UAN?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "A RDC 216/2004 define..."
    assert body["has_history"] is False


def test_chat_exige_mensagem_nao_vazia(settings, completions):
    with build_client(settings, completions) as client:
        response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 400


def test_chat_rejeita_mensagem_longa(settings, completions):
    with build_client(settings, completions) as client:
        response = client.post("/api/v1/chat", json={"message": "a" * 5000})

    assert response.status_code == 422


def test_chap_sem_chave_respond_503_e_nao_vaza_detalhe(completions):
    settings = build_settings(openai_api_key=None)
    with build_client(settings, completions) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 503
    assert "API_KEY" in response.json()["detail"]


def test_prompt_de_sistema_e_enviado(settings, completions):
    with build_client(settings, completions) as client:
        client.post("/api/v1/chat", json={"message": "explique PNAE"})

    call = completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.3
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "explique PNAE"}


def test_historico_acumula_para_o_mesmo_usuario(settings, completions):
    with build_client(settings, completions) as client:
        primeira = client.post("/api/v1/chat", json={"message": "primeira", "user_id": "ana"})
        segunda = client.post("/api/v1/chat", json={"message": "segunda", "user_id": "ana"})

    assert primeira.json()["has_history"] is True
    assert segunda.json()["has_history"] is True

    mensagens = completions.calls[1]["messages"][1:]
    roles = [m["role"] for m in mensagens]
    contents = [m["content"] for m in mensagens]
    assert roles == ["user", "assistant", "user"]
    assert contents == ["primeira", "Resposta de teste", "segunda"]


def test_historico_nao_vaza_entre_usuarios(settings, completions):
    with build_client(settings, completions) as client:
        client.post("/api/v1/chat", json={"message": "segredo da ana", "user_id": "ana"})
        client.post("/api/v1/chat", json={"message": "oi", "user_id": "bruno"})

    contents = [m["content"] for m in completions.calls[1]["messages"][1:]]
    assert contents == ["oi"]


def test_historico_esqueca_apos_ttl(completions):
    settings = build_settings(history_ttl_seconds=0)
    with build_client(settings, completions) as client:
        client.post("/api/v1/chat", json={"message": "primeira", "user_id": "ana"})
        client.post("/api/v1/chat", json={"message": "segunda", "user_id": "ana"})

    contents = [m["content"] for m in completions.calls[1]["messages"][1:]]
    assert contents == ["segunda"]


def test_limite_de_requisicoes_responde_429(completions):
    settings = build_settings(requests_per_minute=2)
    with build_client(settings, completions) as client:
        respostas = [
            client.post("/api/v1/chat", json={"message": f"m{i}"})
            for i in range(4)
        ]

    codigos = [r.status_code for r in respostas]
    assert codigos == [200, 200, 429, 429]
    assert respostas[-1].headers.get("Retry-After")


def test_limite_por_ip_nao_vaza_entre_ips(completions):
    settings = build_settings(requests_per_minute=1)
    with build_client(settings, completions) as client:
        primeiro = client.post("/api/v1/chat", json={"message": "a"}, headers={"x-forwarded-for": "1.1.1.1"})
        segundo = client.post("/api/v1/chat", json={"message": "b"}, headers={"x-forwarded-for": "2.2.2.2"})
        terceiro = client.post("/api/v1/chat", json={"message": "c"}, headers={"x-forwarded-for": "1.1.1.1"})

    assert primeiro.status_code == 200
    assert segundo.status_code == 200
    assert terceiro.status_code == 429


def test_api_key_exigida_quando_configurada(completions):
    settings = build_settings(api_key="segredo")
    with build_client(settings, completions) as client:
        sem = client.post("/api/v1/chat", json={"message": "oi"})
        errada = client.post("/api/v1/chat", json={"message": "oi"}, headers={"x-api-key": "chute"})
        certa = client.post("/api/v1/chat", json={"message": "oi"}, headers={"x-api-key": "segredo"})

    assert sem.status_code == 401
    assert errada.status_code == 401
    assert certa.status_code == 200


def test_erro_da_openai_nao_vaza_detalhe_interno(completions):
    completions.error = openai.APIConnectionError(request=object())
    settings = build_settings()

    with build_client(settings, completions) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502
    detalhe = response.json()["detail"]
    assert "OpenAI" in detalhe
    assert "request=" not in detalhe
    assert "Traceback" not in detalhe


def test_erro_de_autenticacao_da_openai_vira_502(completions):
    completions.error = _openai_error(
        openai.AuthenticationError, "chave inválida sk-projeto-secreto", 401
    )
    with build_client(build_settings(), completions) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502
    assert "sk-projeto-secreto" not in response.json()["detail"]


def test_rate_limit_da_openai_vira_429(completions):
    completions.error = _openai_error(openai.RateLimitError, "limite", 429)
    with build_client(build_settings(), completions) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 429


def test_resposta_vazia_vira_502(completions):
    completions.reply = ""
    with build_client(build_settings(), completions) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502


def test_limpar_conversation_de_um_usuario_nao_afeta_outros(settings, completions):
    with build_client(settings, completions) as client:
        client.post("/api/v1/chat", json={"message": "da ana", "user_id": "ana"})
        client.post("/api/v1/chat", json={"message": "do bruno", "user_id": "bruno"})

        resposta = client.request("DELETE", "/api/v1/chat", params={"user_id": "ana"})
        assert resposta.json()["cleared"] is True

        client.post("/api/v1/chat", json={"message": "de novo", "user_id": "bruno"})

    conteudo_bruno = [m["content"] for m in completions.calls[-1]["messages"][1:]]
    assert "da ana" not in conteudo_bruno
