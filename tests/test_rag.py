"""Testes do RAG com clientes OpenAI e Supabase falsos.

Nenhuma credencial é necessária: tanto o embedding quanto o RPC são
substituídos por dublês. O caminho exercitado é o real — validação,
normalização do que volta do Postgres, montagem do contexto, prompt, tratamento
de erro — sem chamar nenhum serviço.
"""

from contextlib import contextmanager
from unittest.mock import patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.prompts import SYSTEM_PROMPT
from tests.test_api import FakeClient, FakeCompletions, build_settings

EMBEDDING_MODEL = "text-embedding-3-small"


class FakeEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddings:
    """Devolve `dimensions` floats e guarda o que foi pedido."""

    def __init__(self, *, dimensions: int = 1536, error: Exception | None = None):
        self.dimensions = dimensions
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        count = 1 if isinstance(kwargs["input"], str) else len(kwargs["input"])
        data = [FakeEmbedding([0.01] * self.dimensions) for _ in range(count)]
        return type("Resp", (), {"data": data})()


class FakeRpc:
    def __init__(self, rows, *, error: Exception | None = None):
        self.rows = rows
        self.error = error
        self.calls: list[dict] = []

    def execute(self):
        if self.error:
            raise self.error
        return type("Resp", (), {"data": self.rows})()


class FakeTable:
    def __init__(self, store):
        self.store = store

    def insert(self, rows):
        self.store.inserted.extend(rows)
        return type("Resp", (), {"data": rows})()


class FakeStore:
    def __init__(self, *, rows=None, rpc_error: Exception | None = None, rpc_result=None):
        self.rows = rows
        self.rpc_error = rpc_error
        self.rpc_result = rpc_result
        self.rpc_calls: list[dict] = []
        self.inserted: list[dict] = []

    def rpc(self, name, params):
        self.rpc_calls.append({"name": name, "params": params})
        if self.rpc_error:
            raise self.rpc_error
        return FakeRpc(self.rpc_result if self.rpc_result is not None else self.rows)

    def table(self, name):
        return FakeTable(self)


def doc(
    content="texto",
    *,
    source="RDC_216.pdf",
    page=4,
    similarity=0.82,
    metadata="auto",
):
    """Linha no formato que a função `match_documents` devolve."""
    if metadata == "auto":
        meta = {"source": source}
        if page is not None:
            meta["page"] = page
    else:
        meta = metadata
    row = {"id": 1, "content": content, "metadata": meta, "similarity": similarity}
    return row


@contextmanager
def _rag_client(settings, completions, store, embeddings):
    """Sustitui a camada de saída e devolve um TestClient já aberto.

    Só a saída é trocada: validação, normalização do que volta do Postgres,
    montagem do contexto e do prompt continuam sendo o código real.
    """
    import app.main as main_module
    import app.rag as rag_module

    fake = FakeClient(completions)
    fake.embeddings = embeddings

    with (
        patch.object(main_module, "get_client", lambda _settings: fake),
        patch.object(rag_module, "get_store", lambda _settings: store),
    ):
        with TestClient(create_app(settings), raise_server_exceptions=False) as client:
            yield client


@pytest.fixture
def completions():
    return FakeCompletions(reply="Resposta com base no contexto.")


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def rag_settings():
    return build_settings(
        supabase_url="https://projeto-falso.supabase.co",
        supabase_key="eyJhbGciOi-chave-falsa",
        embedding_dimensions=1536,
        rag_match_threshold=0.25,
        rag_match_count=5,
        requests_per_minute=1000,
    )


@pytest.fixture
def store():
    return FakeStore()


def test_health_avisa_que_o_rag_esta_ligado(completions, store, embeddings, rag_settings):
    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.get("/health")

    assert response.json()["rag_configured"] is True
    assert response.json()["embedding_model"] == EMBEDDING_MODEL


def test_saude_sem_supabase_mantem_rag_desligado(completions):
    from tests.test_api import build_client

    with build_client(build_settings(), completions) as client:
        response = client.get("/health")

    assert response.json()["rag_configured"] is False
    assert response.json()["embedding_model"] is None


def test_fluxo_completo_devolve_resposta_e_fontes(rag_settings, completions, embeddings):
    store = FakeStore(rows=[doc("A RDC 216 trata de água.", source="RDC_216.pdf", page=4)])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "O que é a RDC 216?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Resposta com base no contexto."
    assert body["rag_used"] is True
    assert body["sources"] == [{"source": "RDC_216.pdf", "page": 4}]


def test_embedding_e_gerado_antes_da_busca(rag_settings, completions, embeddings):
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        client.post("/api/v1/chat", json={"message": "fator de correção"})

    chamada = embeddings.calls[0]
    assert chamada["model"] == EMBEDDING_MODEL
    assert chamada["dimensions"] == 1536
    assert chamada["input"] == "fator de correção"


def test_rpc_recebe_os_parametros_configurados(rag_settings, completions, embeddings):
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        client.post("/api/v1/chat", json={"message": "UAN"})

    call = store.rpc_calls[0]
    assert call["name"] == "match_documents"
    assert call["params"]["match_threshold"] == 0.25
    assert call["params"]["match_count"] == 5
    assert len(call["params"]["query_embedding"]) == 1536


def test_contexto_entra_no_prompt_de_sistema(rag_settings, completions, embeddings):
    store = FakeStore(
        rows=[doc("Artigo 3: os serviços de alimentação devem...", source="RDC_216.pdf", page=4)]
    )

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        client.post("/api/v1/chat", json={"message": "o que diz o artigo 3?"})

    system = completions.calls[0]["messages"][0]["content"]
    assert system != SYSTEM_PROMPT
    assert "Artigo 3: os serviços de alimentação devem..." in system
    assert "RDC_216.pdf (p. 4)" in system
    assert "=== INÍCIO DO CONTEXTO OFICIAL ===" in system
    assert completions.calls[0]["messages"][1]["content"] == "o que diz o artigo 3?"


def test_sem_trecho_usa_o_prompt_base_e_avisa(rag_settings, completions, embeddings):
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "assunto raro"})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["rag_used"] is False
    system = completions.calls[0]["messages"][0]["content"]
    assert system == SYSTEM_PROMPT


def test_falha_do_rpc_vira_503_sem_vazar_detalhe(rag_settings, completions, embeddings):
    erro = RuntimeError(
        "connection to https://projeto-falso.supabase.co refused: key eyJhbGciOi-chave-falsa"
    )
    store = FakeStore(rpc_error=erro)

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "A base vetorial está indisponível no momento."
    assert "supabase.co" not in detail
    assert "eyJ" not in detail
    # Não pode chamar o modelo: responder sem contexto pareceria fundamento.
    assert completions.calls == []


def test_falha_nao_openai_no_embedding_vira_503(rag_settings, completions):
    """Falha que não é da OpenAI (driver, credencial do Supabase) vira 503 limpo."""
    erro = RuntimeError("postgrestAuthApiError: JWT inválido eyJhbGciOi-chave")
    embeddings = FakeEmbeddings(error=erro)
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "Não foi possível gerar o embedding da pergunta."
    assert "eyJhbGciOi" not in detail
    assert completions.calls == []


def test_erro_de_conexao_da_openai_no_embedding_vira_502(rag_settings, completions):
    """OpenAIError segue o handler global, igual à chamada de chat."""
    embeddings = FakeEmbeddings(error=openai.APIConnectionError(request=object()))
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502
    assert "OpenAI" in response.json()["detail"]


def test_erro_da_openai_no_embedding_vira_502(rag_settings, completions):
    """Exceções da OpenAI passam pelo handler global, não pelo do RAG."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    resposta = httpx.Response(500, request=request)
    segredo = "sk-projeto-indevido-vazar"
    embeddings = FakeEmbeddings(
        error=openai.APIStatusError(segredo, response=resposta, body=None)
    )
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502
    assert segredo not in response.json()["detail"]


def test_dimensao_inesperada_vira_503_legivel(rag_settings, completions):
    embeddings = FakeEmbeddings(dimensions=768)
    store = FakeStore(rows=[])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 503
    assert "Dimensão" in response.json()["detail"]


def test_fontes_duplicadas_sao_colapsadas(rag_settings, completions, embeddings):
    store = FakeStore(
        rows=[
            doc("trecho a", source="RDC_216.pdf", page=4),
            doc("trecho b", source="RDC_216.pdf", page=4),
            doc("trecho c", source="RDC_216.pdf", page=9),
            doc("trecho d", source="PNAN.pdf", page=2),
        ]
    )

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.json()["sources"] == [
        {"source": "RDC_216.pdf", "page": 4},
        {"source": "RDC_216.pdf", "page": 9},
        {"source": "PNAN.pdf", "page": 2},
    ]


def test_trechos_abaixo_do_limiar_sao_descartados(rag_settings, completions, embeddings):
    """Protege contra função SQL desatualizada devolvendo similaridade baixa."""
    store = FakeStore(
        rows=[
            doc("boa", similarity=0.9),
            doc("ruim", similarity=0.10),
        ]
    )

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.json()["sources"] == [{"source": "RDC_216.pdf", "page": 4}]
    system = completions.calls[0]["messages"][0]["content"]
    assert "boa" in system
    assert "ruim" not in system


def test_pagina_como_string_e_normalizada(rag_settings, completions, embeddings):
    store = FakeStore(rows=[doc("texto", metadata={"source": "PNAN.pdf", "page": "17"})])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.json()["sources"] == [{"source": "PNAN.pdf", "page": 17}]


def test_metadata_ausente_ou_invalida_nao_quebra(rag_settings, completions, embeddings):
    store = FakeStore(
        rows=[
            doc("a", metadata=None),
            doc("b", metadata="quebrado{{"),
            doc("", metadata={"source": "vazio.pdf"}),  # conteúdo vazio: descarta
            {"id": 9, "content": None},  # linha sem conteúdo
            "linha que nem é dict",
        ]
    )

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 200
    # As duas linhas degeneradas viram a mesma fonte, então o dedup deixa uma.
    assert response.json()["sources"] == [{"source": "Documento oficial", "page": None}]


def test_contexto_respeita_o_teto_de_caracteres(rag_settings, completions, embeddings):
    store = FakeStore(
        rows=[doc("X" * 5000, source="Manual.pdf", page=1) for _ in range(5)]
    )
    settings = rag_settings.model_copy(
        update={"rag_max_chunk_chars": 300, "rag_max_context_chars": 1200}
    )

    with _rag_client(settings, completions, store, embeddings) as client:
        client.post("/api/v1/chat", json={"message": "oi"})

    system = completions.calls[0]["messages"][0]["content"]
    bloco = system.split("=== INÍCIO DO CONTEXTO OFICIAL ===")[1]
    trecho = bloco.split("[1] Fonte: Manual.pdf (p. 1)\n")[1].split("\n\n")[0]
    assert len(trecho) <= 304  # 300 + o marcador de corte " […]"
    assert trecho.endswith("[…]")
    assert len(bloco) < 1400


def test_historico_continua_valendo_com_rag_ligado(rag_settings, completions, embeddings, store):
    with _rag_client(rag_settings, completions, store, embeddings) as client:
        client.post("/api/v1/chat", json={"message": "primeira", "user_id": "ana"})
        segunda = client.post("/api/v1/chat", json={"message": "segunda", "user_id": "ana"})

    assert segunda.json()["has_history"] is True
    contents = [m["content"] for m in completions.calls[1]["messages"][1:]]
    assert contents == ["primeira", "Resposta com base no contexto.", "segunda"]


def test_chave_de_api_continua_exigindo_no_modo_rag(completions, embeddings, store):
    settings = build_settings(
        supabase_url="https://projeto-falso.supabase.co",
        supabase_key="eyJ",
        api_key="segredo",
        requests_per_minute=1000,
    )

    with _rag_client(settings, completions, store, embeddings) as client:
        sem = client.post("/api/v1/chat", json={"message": "oi"})
        com = client.post(
            "/api/v1/chat", json={"message": "oi"}, headers={"x-api-key": "segredo"}
        )

    assert sem.status_code == 401
    assert com.status_code == 200


def test_limite_de_requisicoes_continua_valendo_com_rag(completions, embeddings):
    store = FakeStore(rows=[])
    settings = build_settings(
        supabase_url="https://projeto-falso.supabase.co",
        supabase_key="eyJ",
        requests_per_minute=2,
    )

    with _rag_client(settings, completions, store, embeddings) as client:
        codigos = [
            client.post("/api/v1/chat", json={"message": f"m{i}"}).status_code for i in range(4)
        ]

    assert codigos == [200, 200, 429, 429]


def test_resposta_vazia_com_rag_vira_502(rag_settings, embeddings):
    completions = FakeCompletions(reply="")
    store = FakeStore(rows=[doc()])

    with _rag_client(rag_settings, completions, store, embeddings) as client:
        response = client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 502