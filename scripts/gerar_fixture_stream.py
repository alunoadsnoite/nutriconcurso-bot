#!/usr/bin/env python
"""Regera o fixture de stream consumido pelo app.

O app tem um teste de contrato que lê
`nutriconcurso/tests/fixtures/stream-exemplo.txt`. Esse arquivo tem que ser a
saída REAL da rota `/api/v1/chat/stream`, não um exemplo escrito à mão: exemplo
inventado concorda com o formato antigo, e o teste passa enquanto o app já
quebrou em produção.

Então isto roda a rota de verdade, com o OpenAI dublê, e grava o que saiu.

    python -m scripts.gerar_fixture_stream

Rode depois de mexer no formato dos eventos SSE. Se a intenção for quebrar o
contrato de propósito, o app tem de falhar junto — e o `npm test` mostra.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.rag import RetrievedChunk
from tests.test_api import FakeClient, build_settings
from tests.test_novo import SmartCompletions

DESTINO = (
    Path(__file__).resolve().parents[2]
    / "nutriconcurso"
    / "tests"
    / "fixtures"
    / "stream-exemplo.txt"
)

# Cada pedaço corta num lugar diferente de propósito:
#  - um token termina no meio do marcador (`[[fon` / `te:1]]`);
#  - outro traz citação para um bloco que não existe no contexto;
#  - outro traz `[1]`, que é inciso de lei e não pode ser confundido com fonte.
PEDACOS = [
    "A UAN (Unidade de Alimentação ",
    "e Nutrição) equivale a 1.400 kcal [[fon",
    "te:1]] por pessoa ao dia. Este trecho [[fonte:9]]",
    " não devia aparecer. [1] do inciso continua.",
]


def contexto() -> list:
    return [
        RetrievedChunk(content=f"trecho {i} sobre UAN", source=f"manual-{i}.pdf", page=i, similarity=0.5)
        for i in range(1, 4)
    ]


def main() -> None:
    completions = SmartCompletions(reply="resposta sem uso direto")
    completions.stream_pieces = PEDACOS

    settings = build_settings().model_copy(
        update={"supabase_url": "https://exemplo.supabase.co", "supabase_key": "chave-falsa"}
    )

    with patch("app.main.get_client", lambda _s: FakeClient(completions)):
        with patch("app.main.retrieve", return_value=contexto()):
            with TestClient(create_app(settings), raise_server_exceptions=False) as client:
                resposta = client.post(
                    "/api/v1/chat/stream",
                    json={"message": "o que e UAN?", "user_id": "ana"},
                )

    if resposta.status_code != 200:
        print(f"A rota devolveu {resposta.status_code}: {resposta.text[:400]}")
        sys.exit(1)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(resposta.text, encoding="utf-8")

    print(f"Gravado em {DESTINO}")
    print(resposta.text)


if __name__ == "__main__":
    main()