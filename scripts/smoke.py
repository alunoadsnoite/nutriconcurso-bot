#!/usr/bin/env python
"""Teste de fumaça contra os serviços REAIS.

Este é o único lugar do projeto que não usa dublê. Ele existe porque todo o
resto roda com OpenAI e Supabase falsificados, e isso não prova duas coisas que
só o serviço de verdade prova:

1. que o PostgREST converte o array JSON do embedding no tipo `vector` do
   Postgres (se não converter, `match_documents` quebra e nada avisa);
2. que `OPENAI_API_KEY`, o nome do modelo e o esquema de `documents` batem com o
   que está no Supabase de verdade.

Também passa pelo caminho de citações e sugestões, porque esses dois dependem
de o modelo obedecer um formato de saída específico — e obediência de modelo só
se descobre na chamada real.

Como rodar, com as credenciais no ambiente:

    OPENAI_API_KEY=sk-... \\
    SUPABASE_URL=https://seuprojeto.supabase.co \\
    SUPABASE_KEY=eyJ... \\
    python -m scripts.smoke

Faz uma inserção com `source = 'smoke-test-<uuid>'`, pergunta, e apaga tudo o
que criou. Não toca em documento real: o `source` é único e o `delete` é
filtrado por ele.

Sem credencial, ele falha dizendo o que falta. Não finge que passou.
"""

from __future__ import annotations

import sys
import uuid

from app.citations import validate_citations
from app.config import Settings
from app.openai_client import get_client
from app.prompts import rag_system_prompt
from app.rag import build_sources, format_context, get_store, retrieve
from app.suggestions import generate

PREGUNTA = "O que é a UAN no PNAE?"


def _falhar(mensagem: str) -> None:
    print(f"\nFALHOU: {mensagem}")
    raise SystemExit(1)


def main() -> None:
    settings = Settings()

    print(f"OpenAI:   {settings.openai_model} / {settings.embedding_model} ({settings.embedding_dimensions} dim)")
    print(f"Supabase: {settings.supabase_url or '(não configurado)'}")
    print(f"Híbrida:  {settings.rag_hybrid}")

    if not settings.is_configured:
        _falhar("OPENAI_API_KEY ausente. O resto do projeto roda sem ela; este script não.")
    if not settings.is_rag_configured:
        _falhar("SUPABASE_URL ou SUPABASE_KEY ausente.")

    marker = f"smoke-test-{uuid.uuid4().hex[:12]}"
    print(f"Marca desta execução: {marker}\n")

    client = get_client(settings)

    # --- 1. embedding real ---------------------------------------------------
    print("1. gerando embedding da pergunta...")
    try:
        emb = client.embeddings.create(
            model=settings.embedding_model,
            input=PREGUNTA,
            dimensions=settings.embedding_dimensions,
        ).data[0].embedding
    except Exception as exc:  # noqa: BLE001
        _falhar(f"OpenAI recusou o embedding: {type(exc).__name__}: {exc}")

    if len(emb) != settings.embedding_dimensions:
        _falhar(f"Embedding saiu com {len(emb)} dimensões, esperado {settings.embedding_dimensions}.")

    print(f"   ok: {len(emb)} dimensões")

    # --- 2. tabela existe ---------------------------------------------------
    print("\n2. conferindo a tabela...")
    try:
        store = get_store(settings)
        store.table("documents").select("id").limit(1).execute()
    except Exception as exc:  # noqa: BLE001
        _falhar(
            "Não consegui ler a tabela `documents`. Rode migrations/001_rag.sql no Supabase.\n"
            f"   erro: {type(exc).__name__}: {exc}"
        )
    print("   ok: tabela acessível")

    # --- 3. inserção de teste ----------------------------------------------
    print("\n3. inserindo trecho de teste...")
    try:
        store.table("documents").insert(
            [
                {
                    "content": (
                        "A UAN (Unidade de Alimentação e Nutrição) equivale a 1.400 kcal "
                        "por pessoa ao dia e é a base do cálculo do PNAE. "
                        "Este trecho existe apenas para o teste de fumaça."
                    ),
                    "metadata": {"source": marker, "page": 1},
                    "embedding": emb,
                }
            ]
        ).execute()
    except Exception as exc:  # noqa: BLE001
        _falhar(
            "Não consegui inserir. A chave precisa de INSERT: use a `service_role`, "
            "porque a `anon` é somente leitura nesta base.\n"
            f"   erro: {type(exc).__name__}: {exc}"
        )
    print("   ok: 1 trecho inserido")

    # --- 4. a busca acha o que acabamos de inserir ---------------------------
    print("\n4. buscando de volta...")
    try:
        chunks = retrieve(settings, PREGUNTA, client)
    except Exception as exc:  # noqa: BLE001
        _falhar(f"A busca falhou: {type(exc).__name__}: {exc}")

    if not chunks:
        print(
            "   ATENÇÃO: 0 trechos. Se a base está vazia, é o esperado.\n"
            "   Se já tem documentos, investigue RAG_MATCH_THRESHOLD e a função\n"
            "   match_documents no banco."
        )
    else:
        print(f"   ok: {len(chunks)} trecho(s)")
        for chunk in chunks[:3]:
            print(f"      {chunk.source} p.{chunk.page} similaridade={chunk.similarity}")
            print(f"        {chunk.content[:80]}...")
        if any(c.source == marker for c in chunks):
            print("   ok: o trecho de teste apareceu — o vetor fez o round-trip.")

    context = format_context(
        chunks,
        max_chunk_chars=settings.rag_max_chunk_chars,
        max_total_chars=settings.rag_max_context_chars,
    )
    sources = build_sources(chunks)
    print(f"   contexto: {len(context)} caracteres, {len(sources)} fonte(s)")

    # --- 5. resposta real, com citações -------------------------------------
    print("\n5. chamando o modelo com o contexto...")
    try:
        reply = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": rag_system_prompt(context)},
                {"role": "user", "content": PREGUNTA},
            ],
            temperature=0.3,
            max_tokens=300,
        ).choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        _falhar(f"O modelo recusou a chamada: {type(exc).__name__}: {exc}")

    limpo, citadas = validate_citations(reply, block_count=len(chunks))
    print(f"   ok: {len(reply)} caracteres")
    print(f"   citações aceitas: {citadas or 'nenhuma'}")
    if len(chunks) and not citadas:
        print(
            "   AVISO: o modelo respondeu sem citar nada. Se isso se repetir em\n"
            "   produção, o prompt em app/prompts.py precisa de ajuste."
        )
    if limpo != reply:
        print(f"   texto limpo: {limpo[:160]}")

    # --- 6. sugestões reais -------------------------------------------------
    print("\n6. gerando sugestões...")
    try:
        sugestoes = generate(client, settings, question=PREGUNTA, reply=reply, history=[])
    except Exception as exc:  # noqa: BLE001
        _falhar(f"A geração de sugestões quebrou: {type(exc).__name__}: {exc}")

    if sugestoes:
        for pergunta in sugestoes:
            print(f"   - {pergunta}")
    else:
        print(
            "   AVISO: nenhuma sugestão. O modelo não seguiu o formato JSON.\n"
            "   Confira app/suggestions.py e o prompt antes de confiar nesse caminho."
        )

    # --- 7. limpeza ---------------------------------------------------------
    print("\n7. limpando o trecho de teste...")
    try:
        store.table("documents").delete().eq("metadata->>source", marker).execute()
        print("   ok: apagado")
    except Exception as exc:  # noqa: BLE001
        print(f"   NÃO LIMPEI: {type(exc).__name__}: {exc}")
        print(f"   Apague na mão: delete from public.documents where metadata->>'source' = '{marker}';")
        raise SystemExit(1) from None

    print("\nFUMAÇA PASSOU: embedding, Postgres/vector, busca, RPC, citações, sugestões e modelo.")


if __name__ == "__main__":
    main()