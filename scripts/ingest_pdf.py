#!/usr/bin/env python
"""Popula a base vetorial a partir de PDFs de legislação.

    python -m scripts.ingest_pdf Manual_PNAE_2023.pdf

    python -m scripts.ingest_pdf --recursive pasta/  # vários arquivos em subpastas

O `metadata` de cada trecho guarda `source` (nome do arquivo) e `page` (página
1-based), que é o que a API devolve nos badges de fonte. Se o seu extrator já
gravar outras chaves, elas atravessam intactas no `metadata`.

Requer as mesmas variáveis do servidor: `OPENAI_API_KEY`, `SUPABASE_URL` e
`SUPABASE_KEY` (a `service_role` para escrever, já que anon não tem insert).

Custo: embeddings custam ~US$ 0,02 por 1M de tokens, então reingerir o mesmo
documento paga de novo. Delete os trechos antigos antes quando corrigir a
base:

    delete from public.documents where metadata->>'source' = 'Manual_PNAE_2023.pdf';
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import Settings
from app.openai_client import get_client
from app.rag import get_store

logger = logging.getLogger("nutriconcurso.ingest")

# Um pedaço por página é o mais simples e funciona bem para legislação, onde a
# página é a unidade de referência que o estudante reconhece. Se a sua extração
# de texto vier embaralhada, troque por pdfplumber e valide a saída antes.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    sys.exit("Falta a dependência de leitura de PDF. Instale com: pip install pypdf")

BATCH_SIZE = 64


def extract_pages(path: Path) -> list[str]:
    """Texto por página. Página em branco vira string vazia e é descartada."""
    reader = PdfReader(str(path))
    pages: list[str] = []

    for number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:  # noqa: BLE001 - página quebrada não derruba o lote
            logger.warning("Não consegui ler a página %d de %s: %s", number, path.name, exc)
            text = ""

        if text:
            pages.append(text)

    return pages


def embed_all(settings: Settings, texts: list[str]) -> list[list[float]]:
    """Vetoriza em lotes: a API aceita vários textos por chamada."""
    client = get_client(settings)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        vectors.extend(item.embedding for item in response.data)
        logger.info("Vetorizados %d/%d", len(vectors), len(texts))

    return vectors


def ingest(settings: Settings, path: Path) -> int:
    pages = extract_pages(path)
    if not pages:
        logger.warning("%s não tem texto extraível (PDF digitalizado?).", path.name)
        return 0

    vectors = embed_all(settings, pages)
    if len(vectors) != len(pages):
        raise RuntimeError(
            f"A OpenAI devolveu {len(vectors)} vetores para {len(pages)} páginas. "
            "Verifique EMBEDDING_DIMENSIONS."
        )

    rows = [
        {
            "content": text,
            "metadata": {"source": path.name, "page": number},
            "embedding": vector,
        }
        for number, (text, vector) in enumerate(zip(pages, vectors), start=1)
    ]

    store = get_store(settings)
    store.table("documents").insert(rows)

    logger.info("%s: %d trechos inseridos", path.name, len(rows))
    return len(rows)


def collect_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in target.glob(pattern) if p.is_file())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alvo", type=Path, help="PDF ou pasta com PDFs")
    parser.add_argument("--recursive", action="store_true", help="Descer em subpastas")
    args = parser.parse_args()

    if not args.alvo.exists():
        sys.exit(f"Caminho não encontrado: {args.alvo}")

    settings = Settings()
    if not settings.is_rag_configured:
        sys.exit("Faltam SUPABASE_URL e SUPABASE_KEY no .env.")
    if not settings.is_configured:
        sys.exit("Falta OPENAI_API_KEY no .env.")

    arquivos = collect_files(args.alvo, args.recursive)
    if not arquivos:
        sys.exit("Nenhum PDF encontrado.")

    total = 0
    for path in arquivos:
        try:
            total += ingest(settings, path)
        except Exception as exc:  # noqa: BLE001 - um PDF ruim não para o lote
            logger.error("Falha em %s: %s", path.name, exc)

    logger.info("Concluído. %d trechos em %d arquivo(s).", total, len(arquivos))


if __name__ == "__main__":
    main()