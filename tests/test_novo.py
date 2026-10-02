"""Testes dos recursos novos e dos dois bugs corrigidos.

Sem `OPENAI_API_KEY` e sem Supabase: OpenAI é dublê, o `supabase` não é usado.
O que se exercita de verdade é o caminho HTTP completo, incluindo o stream.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.citations import StreamCitationFilter, strip_all_citations, validate_citations
from app.main import create_app
from app.suggestions import parse_suggestions
from app.rag import RetrievedChunk
from tests.test_api import FakeClient, FakeCompletions, build_settings, chat_calls


# ---------------------------------------------------------------- fakes -----


class FakeStreamPiece:
    def __init__(self, content: str | None):
        delta = type("Delta", (), {"content": content})()
        self.choices = [type("C", (), {"delta": delta})()] if content else []


class FakeStream:
    def __init__(self, pieces, *, error=None):
        self.pieces = pieces
        self.error = error

    def __iter__(self):
        if self.error:
            raise self.error
        for piece in self.pieces:
            yield FakeStreamPiece(piece)


class SmartCompletions(FakeCompletions):
    """Entende `stream=True` e o prompt de sugestões."""

    def create(self, **kwargs):
        if kwargs.get("response_format"):
            body = (
                '{"perguntas": ["Como dimensionar uma UAN de creche?", '
                '"Qual o Fator de Correção?", "Gabarito comentado da questão"]}'
            )
            return type("C", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": body})()})()]})()

        if kwargs.get("stream"):
            self.calls.append(kwargs)
            return FakeStream(getattr(self, "stream_pieces", ["A ", "UAN ", "vale ", "1400 ", "kcal."]))

        return super().create(**kwargs)


def _chunks(n: int = 3) -> list:
    """Contextos de mentira. Só o RAG é dublê; o resto da rota roda inteiro."""
    from app.rag import RetrievedChunk

    return [
        RetrievedChunk(
            content=f"trecho {i} sobre UAN",
            source=f"manual-{i}.pdf",
            page=i,
            similarity=0.5,
        )
        for i in range(1, n + 1)
    ]


@contextmanager
def _client(settings, completions, *, chunks=None):
    """`chunks` liga o caminho de RAG sem precisar de Supabase de verdade."""
    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        if chunks is not None:
            settings = settings.model_copy(
                update={"supabase_url": "https://falso.supabase.co", "supabase_key": "chave-falsa"}
            )
            with patch("app.main.retrieve", return_value=list(chunks)):
                with TestClient(create_app(settings), raise_server_exceptions=False) as client:
                    yield client
        else:
            with TestClient(create_app(settings), raise_server_exceptions=False) as client:
                yield client


@pytest.fixture
def completions():
    return SmartCompletions(reply="A UAN vale 1400 kcal. [[fonte:1]]")


# ------------------------------------------------- bug 1: histórico órfão ----


def test_pergunta_nao_e_duplicada_quando_a_chamada_falha(completions):
    """Regressão: a mensagem do usuário só entra no histórico após o sucesso.

    Antes, `append_user_message` gravava antes de chamar o modelo. Uma falha
    deixava a mensagem órfã e o retry do app gravava a mesma pergunta de novo,
    então o modelo via a pergunta duplicada e sem assistente entre as duas.
    """
    settings = build_settings()
    store_holder = {}

    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        app = create_app(settings)
        store_holder["store"] = app.state.conversation_store
        with TestClient(app, raise_server_exceptions=False) as client:
            completions.error = RuntimeError("openai caiu")
            client.post("/api/v1/chat", json={"message": "o que e UAN?", "user_id": "ana", "include_suggestions": False})

            completions.error = None
            client.post("/api/v1/chat", json={"message": "o que e UAN?", "user_id": "ana", "include_suggestions": False})

    history = store_holder["store"].history("ana")
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant"], f"esperado par limpo, veio {roles}"
    assert history[0]["content"] == "o que e UAN?"


def test_falha_no_stream_nao_poe_no_historico(completions):
    settings = build_settings()
    holder = {}

    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        app = create_app(settings)
        holder["store"] = app.state.conversation_store

        with TestClient(app, raise_server_exceptions=False) as client:
            completions.reply = ""
            client.post("/api/v1/chat", json={"message": "oi", "user_id": "ana", "include_suggestions": False})

    assert holder["store"].history("ana") == []


# --------------------------------------------- bug 2: X-Forwarded-For -------


def test_xff_ignorado_por_padrao(completions):
    settings = build_settings(requests_per_minute=2)
    with _client(settings, completions) as client:
        codigos = [
            client.post(
                "/api/v1/chat",
                json={"message": f"m{i}", "include_suggestions": False},
                headers={"x-forwarded-for": f"203.0.113.{i}"},
            ).status_code
            for i in range(4)
        ]
    assert codigos == [200, 200, 429, 429]


# ----------------------------------------------- limite por user_id ---------


def test_limite_por_usuario_derruba_o_uso_de_uma_pessoa(completions):
    """O limite por IP não segura um aluno consumindo a conta inteira."""
    settings = build_settings(requests_per_minute=1000, requests_per_minute_per_user=2)
    with _client(settings, completions) as client:
        mesmo_usuario = [
            client.post("/api/v1/chat", json={"message": f"m{i}", "user_id": "ana", "include_suggestions": False}).status_code
            for i in range(3)
        ]
        outro_usuario = client.post(
            "/api/v1/chat", json={"message": "oi", "user_id": "bruno", "include_suggestions": False}
        ).status_code

    assert mesmo_usuario == [200, 200, 429]
    # O bloqueio é por pessoa, não global.
    assert outro_usuario == 200


def test_sem_user_id_so_sofre_o_limite_por_ip(completions):
    settings = build_settings(requests_per_minute=1000, requests_per_minute_per_user=1)
    with _client(settings, completions) as client:
        codigos = [
            client.post("/api/v1/chat", json={"message": f"m{i}", "include_suggestions": False}).status_code
            for i in range(3)
        ]
    assert codigos == [200, 200, 200]


# --------------------------------------------------- histórico via GET -------


def test_get_devolve_o_historico_guardado(completions):
    settings = build_settings()
    with _client(settings, completions) as client:
        client.post("/api/v1/chat", json={"message": "o que e UAN?", "user_id": "ana", "include_suggestions": False})
        resposta = client.get("/api/v1/chat", params={"user_id": "ana"})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert [m["role"] for m in corpo["messages"]] == ["user", "assistant"]
    assert corpo["messages"][0]["content"] == "o que e UAN?"


def test_get_de_usuario_desconhecido_vazio(completions):
    with _client(build_settings(), completions) as client:
        resposta = client.get("/api/v1/chat", params={"user_id": "ninguem"})
    assert resposta.status_code == 200
    assert resposta.json()["messages"] == []


# ------------------------------------------------------------- citações ------


def test_citacao_valida_e_mantida_no_texto():
    limpo, usadas = validate_citations("A UAN vale 1400 kcal. [[fonte:2]]", block_count=3)
    assert limpo == "A UAN vale 1400 kcal. [[fonte:2]]"
    assert usadas == [2]


def test_citacao_para_bloque_inexistente_e_removida():
    limpo, usadas = validate_citations("Texto [[fonte:9]] fim", block_count=3)
    # O espaço duplo fica: o stream não pode reescrever o que já foi enviado,
    # então os dois caminhos preservam o espaçamento original.
    assert limpo == "Texto  fim"
    assert usadas == []
    assert "[[" not in limpo


def test_citacao_malformada_e_removida():
    for texto in ("a [[fonte:abc]] b", "a [[fonte:]] b", "a [[fonte: b", "a [[fonte]] b"):
        limpo, usadas = validate_citations(texto, block_count=3)
        assert "[[" not in limpo, texto
        assert usadas == []


def test_citacao_repetida_aparece_uma_vez():
    limpo, usadas = validate_citations("A [[fonte:1]] e B [[fonte:1]] e C [[fonte:2]]", block_count=2)
    assert limpo.count("[[fonte:1]]") == 1
    assert usadas == [1, 2]


def test_inciso_legal_com_colchetes_simples_nao_e_tocado():
    """`[1]` é inciso de lei, não é marcador de fonte."""
    texto = " conforme o inciso [1] do art. 5º da RDC 275"
    limpo, usadas = validate_citations(texto, block_count=2)
    assert limpo == texto
    assert usadas == []


def test_sem_contexto_todo_marcador_some():
    limpo = strip_all_citations("Resposta [[fonte:1]] solta [[fonte:2]]")
    assert "[[" not in limpo


# ---------------------------------------------------------- sugestões --------


def test_parse_sugestoes_do_json_esperado():
    bruto = '{"perguntas": ["Primeira pergunta?", "Segunda pergunta?", "Terceira pergunta?"]}'
    assert parse_suggestions(bruto, limit=3) == [
        "Primeira pergunta?",
        "Segunda pergunta?",
        "Terceira pergunta?",
    ]


def test_parse_sugestoes_aceita_cerca_de_codigo():
    bruto = '```json\n{"perguntas": ["Uma pergunta?"]}\n```'
    assert parse_suggestions(bruto, limit=3) == ["Uma pergunta?"]


def test_parse_sugestores_aceita_array_solto():
    assert parse_suggestions('["Uma pergunta?", "Outra pergunta?"]', limit=3) == [
        "Uma pergunta?",
        "Outra pergunta?",
    ]


def test_parse_sugestoes_descarta_texto_nao_json():
    bruto = "Desculpe, não consegui gerar as perguntas."
    assert parse_suggestions(bruto, limit=3) == []


def test_parse_sugestoes_deduplica_e_respeita_o_limite():
    bruto = '{"perguntas": ["Repetida?", "Repetida?", "Outra?"]}'
    assert parse_suggestions(bruto, limit=5) == ["Repetida?", "Outra?"]


def test_chat_devolve_sugestoes(completions):
    with _client(build_settings(), completions) as client:
        resposta = client.post("/api/v1/chat", json={"message": "o que e UAN?"})

    corpo = resposta.json()
    assert len(corpo["suggestions"]) == 3
    assert corpo["suggestions"][0].startswith("Como dimensionar")


def test_sugestoes_desligadas_nao_fazem_chamada_extra(completions):
    with _client(build_settings(), completions) as client:
        resposta = client.post("/api/v1/chat", json={"message": "oi", "include_suggestions": False})

    assert resposta.json()["suggestions"] == []
    assert all(c.get("response_format") is None for c in completions.calls)


def test_falha_na_geracao_de_sugestoes_nao_derruba_a_resposta():
    """Perder os chips é aceitável; perder a resposta não."""
    completions = FakeCompletions(reply="resposta boa")

    def create(**kwargs):
        if kwargs.get("response_format"):
            raise RuntimeError("sugestoes fora do ar")
        completions.calls.append(kwargs)
        msg = type("M", (), {"content": completions.reply})()
        return type("C", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    completions.create = create

    with _client(build_settings(), completions) as client:
        resposta = client.post("/api/v1/chat", json={"message": "oi"})

    assert resposta.status_code == 200
    assert resposta.json()["reply"] == "resposta boa"
    assert resposta.json()["suggestions"] == []


# ------------------------------------------------------------- streaming -----


def _eventos(resposta) -> list[tuple[str, dict]]:
    import json

    achados = []
    for bruto in resposta.iter_lines():
        if bruto.startswith("event: "):
            evento = bruto[len("event: ") :].strip()
        elif bruto.startswith("data: "):
            dados = json.loads(bruto[len("data: ") :])
            achados.append((evento, dados))
    return achados


def test_stream_entrega_fontes_token_e_done(completions):
    with _client(build_settings(), completions) as client:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
        ) as resposta:
            assert resposta.status_code == 200
            assert resposta.headers["content-type"].startswith("text/event-stream")
            eventos = _eventos(resposta)

    nomes = [nome for nome, _ in eventos]
    assert nomes[-1] == "done"
    assert "token" in nomes

    texto = "".join(dados["t"] for nome, dados in eventos if nome == "token")
    assert texto == "A UAN vale 1400 kcal."
    assert eventos[-1][1] == {"citations": []}


def test_stream_persiste_o_historico(completions):
    settings = build_settings()
    holder = {}

    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        app = create_app(settings)
        holder["store"] = app.state.conversation_store
        with TestClient(app, raise_server_exceptions=False) as client:
            with client.stream(
                "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
            ) as resposta:
                list(resposta.iter_lines())

    history = holder["store"].history("ana")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "A UAN vale 1400 kcal."


def test_stream_exige_mensagem_nao_vazia(completions):
    with _client(build_settings(), completions) as client:
        resposta = client.post("/api/v1/chat/stream", json={"message": "   "})
    assert resposta.status_code == 400


def test_stream_sem_chave_responde_503(completions):
    with _client(build_settings(openai_api_key=None), completions) as client:
        resposta = client.post("/api/v1/chat/stream", json={"message": "oi"})
    assert resposta.status_code == 503


def test_stream_exige_api_key(completions):
    settings = build_settings(api_key="segredo")
    with _client(settings, completions) as client:
        sem = client.post("/api/v1/chat/stream", json={"message": "oi"})
        com = client.post("/api/v1/chat/stream", json={"message": "oi"}, headers={"x-api-key": "segredo"})
    assert sem.status_code == 401
    assert com.status_code == 200


def test_health_anuncia_streaming(completions):
    with _client(build_settings(), completions) as client:
        corpo = client.get("/health").json()
    assert corpo["streaming"] is True

# ------------------------- stream x caminho normal: mesma resposta ---------


def _filtrar(pieces, *, block_count):
    """Roda os pedaços pelo filtro e devolve o texto exibido + citados."""
    f = StreamCitationFilter(block_count=block_count)
    visivel = "".join(f.push(p) for p in pieces)
    visivel += f.flush()
    return visivel, f.used


@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param(["A UAN vale 1400 kcal. ", "[[fonte:", "1]]", " Fim."], id="marcador-partido-em-3"),
        pytest.param(["[[fonte:1]]", " e ", "[[fonte:2]]"], id="marcador-inteiro"),
        pytest.param(["Texto ", "[[fon", "te:2", "]]", " depois"], id="marcador-muito-quebrado"),
    ],
)
def test_stream_entrega_o_mesmo_texto_que_a_rota_normal(pieces):
    """O que o stream mostra tem que ser idêntico ao que a rota sem stream
    devolveria, senão o app mostra uma coisa e o histórico guarda outra."""
    reply = "".join(pieces)
    esperado, citados = validate_citations(reply, block_count=3)
    visivel, citados_stream = _filtrar(pieces, block_count=3)

    assert visivel == esperado
    assert citados_stream == citados


def test_stream_remove_citacao_invalida_antes_de_exibir():
    visivel, citados = _filtrar(["ok ", "[[fonte:99]]", " texto"], block_count=3)
    assert "[[" not in visivel
    assert citados == []


def test_stream_nao_exibe_citacao_repetida():
    visivel, citados = _filtrar(["a [[fonte:1]]", " b [[fonte:1]]"], block_count=2)
    assert visivel.count("[[fonte:1]]") == 1
    assert citados == [1]


def test_stream_nao_espera_o_marcador_para_passar_texto():
    """Texto comum tem que sair já, senão o app vira resposta só no fim."""
    f = StreamCitationFilter(block_count=3)
    assert f.push("A UAN vale ") == "A UAN vale "


def test_stream_segura_ate_o_marcador_fechar():
    f = StreamCitationFilter(block_count=3)
    assert f.push("valor 1400 [[fonte:") == "valor 1400 "
    assert f.push("1]]") == "[[fonte:1]]"
    assert f.used == [1]


def test_stream_sem_contexto_nao_deixa_marcador_vazar():
    visivel, citados = _filtrar(["Resposta ", "[[fonte:1]]", " solta"], block_count=0)
    assert "[[" not in visivel
    assert citados == []


def test_stream_limita_o_buffer_de_marcador_aberto():
    """Modelo que nunca fecha o marcador não pode segurar memória sem fim."""
    f = StreamCitationFilter(block_count=3)
    liberado = "".join(f.push("[[fonte:" + "x" * 400) for _ in range(3))
    assert len(liberado) > 0


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("Texto com [[nota de rodape]] no meio", id="colchete-que-nao-e-marker"),
        pytest.param("Fim truncado [[fonte:1", id="marcador-sem-fechamento"),
        pytest.param("Antes [[fonte:1]] e [[fonte:2]] depois", id="marcadores-ok"),
        pytest.param("Nenhum marcador aqui", id="sem-marcador"),
        pytest.param("Repetido [[fonte:1]] [[fonte:1]]", id="repetido"),
        pytest.param("Fora [[fonte:7]] de faixa", id="fora-de-faixa"),
        pytest.param("Malformado [[fonte:x]] fim", id="malformado"),
        pytest.param("Comeca com [[fonte:1]] e nada mais", id="marcador-sozinho"),
        pytest.param("Meio [[fonte:2]]", id="marcador-no-fim"),
        pytest.param("Colchete duplo [[no fim", id="colchete-duplo-no-fim"),
        pytest.param("Termina em [[fon", id="prefixo-truncado-no-fim"),
    ],
)
def test_stream_bate_com_a_rota_normal_em_todo_caso(reply):
    """Propriedade que importa: para a MESMA resposta, os dois caminhos
    devolvem exatamente o mesmo texto e os mesmos números citados.

    Se isso divergir, o aluno lê uma coisa na tela e o histórico guarda outra,
    e a próxima pergunta do modelo é montada sobre o texto errado.
    """
    esperado, citados = validate_citations(reply, block_count=3)

    # Vários cortes possíveis, para provar que a partição não importa.
    visivel, citados_stream = _filtrar(list(reply), block_count=3)
    assert visivel == esperado, "um pedaço só"
    assert citados_stream == citados

    visivel, citados_stream = _filtrar([reply[i : i + 3] for i in range(0, len(reply), 3)], block_count=3)
    assert visivel == esperado, "de 3 em 3"
    assert citados_stream == citados

    visivel, citados_stream = _filtrar(list(reply), block_count=3)
    assert visivel == esperado, "um caractere por vez"
    assert citados_stream == citados

    visivel, citados_stream = _filtrar(list(reply), block_count=0)
    assert visivel == strip_all_citations(reply), "sem contexto"
    assert citados_stream == []


# ------------------ o filtro de citações está ligado na rota de verdade ------


def _coletar(resposta):
    """Lê o stream UMA vez e devolve (texto exibido, eventos).

    Ler duas vezes estoura: o corpo já foi consumido.
    """
    eventos = _eventos(resposta)
    texto = "".join(dados["t"] for nome, dados in eventos if nome == "token")
    return texto, eventos


def _tokens_de(resposta) -> str:
    return _coletar(resposta)[0]


def test_stream_nao_entrega_citacao_que_aponta_para_bloque_inexistente(completions):
    """Regressão do defeito mais chato: o modelo escrevia [[fonte:99]] e o
    texto já chegava na tela do aluno antes de haver como saber que o bloco 99
    nem existe. Com o filtro, o marcador nunca sai."""
    completions.stream_pieces = ["A UAN vale 1400 kcal ", "[[fonte:99]]", " conforme o manual."]

    with _client(build_settings(), completions, chunks=_chunks(3)) as client:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
        ) as resposta:
            texto, eventos = _coletar(resposta)

    assert "[[fonte:99]]" not in texto
    assert "99" not in texto
    assert eventos[-1] == ("done", {"citations": []})


def test_stream_guarda_no_historico_exatamente_o_que_exibiu(completions):
    """Se o histórico guardasse um texto diferente do mostrado, a próxima
    pergunta do modelo seria montada sobre a resposta errada."""
    completions.stream_pieces = ["A UAN vale 1400 kcal ", "[[fonte:99]]", " fim."]

    settings = build_settings()
    holder = {}
    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        app = create_app(settings)
        holder["store"] = app.state.conversation_store
        with TestClient(app, raise_server_exceptions=False) as client:
            with client.stream(
                "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
            ) as resposta:
                texto = _tokens_de(resposta)

    guardado = holder["store"].history("ana")[1]["content"]
    assert guardado == texto
    assert "[[" not in guardado


def test_stream_marca_citacao_valida_e_devolve_os_numeros(completions):
    completions.stream_pieces = ["A UAN vale 1400 [[fon", "te:1]]", " e 800 no lanche."]

    with _client(build_settings(), completions, chunks=_chunks(3)) as client:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
        ) as resposta:
            texto, eventos = _coletar(resposta)

    assert "[[fonte:1]]" in texto
    assert eventos[-1] == ("done", {"citations": [1]})


def test_stream_e_rota_normal_dao_o_mesmo_texto(completions):
    """A mesma resposta pelos dois caminhos tem que sair idêntica. É o que
    garante que ativar o streaming no app não muda o que o usuário lê nem o
    que fica no histórico."""
    completions.stream_pieces = ["A UAN vale 1400 ", "[[fonte:9]]", "kcal. ", "[[fonte:2]]", " fim."]
    completions.reply = "".join(completions.stream_pieces)

    with _client(build_settings(), completions, chunks=_chunks(3)) as client:
        normal = client.post(
            "/api/v1/chat", json={"message": "o que e UAN?", "user_id": "ana", "include_suggestions": False}
        ).json()
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
        ) as resposta:
            streamed, eventos = _coletar(resposta)

    assert "fonte:9" not in streamed, "bloco 9 não existe em um contexto de 3"
    assert streamed == normal["reply"], "os dois caminhos têm que concordar"
    assert eventos[-1] == ("done", {"citations": normal["citations"]})


# ------------------------------ o stream com RAG ligado (a 1ª versão crashava) ----


def test_stream_com_rag_anuncia_as_fontes_antes_do_texto(completions):
    """Regressão: a rota de stream desempacotava dois valores de
    `_gather_context`, que devolve três. Com Supabase configurado, o stream
    respondia 500 antes de gerar a primeira letra."""
    completions.stream_pieces = ["A UAN vale 1400 kcal."]

    with _client(build_settings(), completions, chunks=_chunks(3)) as client:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "o que e UAN?", "user_id": "ana"}
        ) as resposta:
            assert resposta.status_code == 200
            texto, eventos = _coletar(resposta)

    nomes = [nome for nome, _ in eventos]
    assert nomes[0] == "sources", "as fontes vêm antes do texto, para o app mostrar o que foi consultado"
    fontes = eventos[0][1]["sources"]
    assert [f["source"] for f in fontes] == ["manual-1.pdf", "manual-2.pdf", "manual-3.pdf"]
    assert eventos[1][1]["t"] == "A UAN vale 1400 kcal."


def test_stream_devolve_503_quando_o_rag_falha(completions):
    """Busca falhando é 503, nunca uma resposta sem contexto: responder sobre
    lei sem documento é pior do que não responder."""
    from app.rag import RetrievalError

    settings = build_settings().model_copy(
        update={"supabase_url": "https://falso.supabase.co", "supabase_key": "chave-falsa"}
    )
    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        with patch("app.main.retrieve", side_effect=RetrievalError("indisponível")):
            with TestClient(create_app(settings), raise_server_exceptions=False) as client:
                resposta = client.post("/api/v1/chat/stream", json={"message": "o que e UAN?"})

    assert resposta.status_code == 503


def test_stream_com_rag_nao_aceita_citacao_de_bloco_que_nao_veio(completions):
    """A citação é válida para o contexto que o stream recebeu, não para o
    documento inteiro. `[[fonte:7]]` num contexto de 3 blocos some."""
    completions.stream_pieces = ["A UAN vale 1400 [[fonte:7]]", " kcal."]

    with _client(build_settings(), completions, chunks=_chunks(3)) as client:
        with client.stream("POST", "/api/v1/chat/stream", json={"message": "oi"}) as resposta:
            texto, eventos = _coletar(resposta)

    assert texto == "A UAN vale 1400  kcal."
    assert eventos[-1] == ("done", {"citations": []})
