-- Busca híbrida: vetorial + full-text em português, fundidas por RRF.
-- Rode DEPOIS de 001_rag.sql. É idempotente.

-- --------------------------------------------------------------------------
-- Coluna de busca textual
--
-- `to_tsvector('portuguese', ...)` com o dicionário explícito: a forma de
-- dois argumentos é apenas STABLE, e coluna gerada exige IMMUTABLE. Sem o
-- 'portuguese' explícito o Postgres recusa o CREATE TABLE.
--
-- `generated always as ... stored` mantém a coluna em sincronia com `content`
-- sem trigger e sem espaço duplicado no disco.
-- --------------------------------------------------------------------------
alter table public.documents
    add column if not exists content_tsv tsvector
    generated always as (to_tsvector('portuguese', content)) stored;

create index if not exists documents_content_tsv_idx
    on public.documents using gin (content_tsv);

-- --------------------------------------------------------------------------
-- Busca híbrida
--
-- Por que RRF (Reciprocal Rank Fusion) e não soma de scores: similarity de
-- cosseno (0..1) e ts_rank_cd (0..inf, escala arbitrária dependente do
-- documento) não são comparáveis. Normalizar exigiria calibração por
-- corpus, que não se mantém. RRF só olha a POSIÇÃO de cada resultado, então as
-- duas buscas se combinam direto: score = Σ 1/(k + rank).
--
-- `websearch_to_tsquery` no lugar de `to_tsquery`: aceita texto do usuário
-- como é e nunca falha com erro de sintaxe, mesmo com aspas ou operador mal
-- digitado. `to_tsquery` quebraria a requisição.
--
-- IMPORTANTE sobre o contrato com a API: a função devolve DOIS scores.
--   similarity = similaridade de cosseno, ou NULL se o trecho entrou só pela
--                busca textual (nesse caso não há cosseno, e não se pode
--                fingir que há).
--   score      = valor fundido do RRF, usado só para ORDENAR.
-- A API continua filtrando e exibindo `similarity`, e usa `score` para
-- ordenar. Confundir os dois faria todo trecho sumir, porque score RRF fica em
-- torno de 0,02 e o limiar padrão é 0,25.
-- --------------------------------------------------------------------------
create or replace function public.match_documents_hybrid (
    query_embedding vector(1536),
    query_text       text,
    match_threshold  float,
    match_count      int,
    rrf_k            int default 60
)
returns table (
    id        bigint,
    content   text,
    metadata  jsonb,
    similarity float,
    score      float
)
language sql
stable
security definer
set search_path = public
as $$
    with semantic as (
        select
            d.id,
            row_number() over (order by d.embedding <=> query_embedding) as rank,
            1 - (d.embedding <=> query_embedding) as cosine
        from public.documents d
        where 1 - (d.embedding <=> query_embedding) > match_threshold
        order by d.embedding <=> query_embedding
        limit 50
    ),
    lexical as (
        select
            d.id,
            row_number() over (order by ts_rank_cd(d.content_tsv, q) desc) as rank
        from public.documents d
            cross join websearch_to_tsquery('portuguese', query_text) q
        where d.content_tsv @@ q
        order by ts_rank_cd(d.content_tsv, q) desc
        limit 50
    ),
    fused as (
        select
            coalesce(s.id, l.id) as id,
            s.cosine as cosine,
            coalesce(1.0 / (rrf_k + s.rank), 0.0)
              + coalesce(1.0 / (rrf_k + l.rank), 0.0) as score
        from semantic s
        full outer join lexical l on l.id = s.id
    )
    select
        d.id,
        d.content,
        d.metadata,
        f.cosine as similarity,
        f.score
    from fused f
    join public.documents d on d.id = f.id
    where f.cosine is null or f.cosine > match_threshold
    order by f.score desc, d.id
    limit least(greatest(match_count, 1), 20);
$$;

revoke all on function public.match_documents_hybrid(vector, text, float, int, int) from public;
grant execute on function public.match_documents_hybrid(vector, text, float, int, int) to anon, authenticated;

-- --------------------------------------------------------------------------
-- Como conferir que deu certo
--
--   -- 1. a coluna foi criada e está preenchida
--   select count(*) filter (where content_tsv is not null) from public.documents;
--
--   -- 2. a busca textual acha o que a vetorial erra
--   select id, round(score::numeric, 4) as score, similarity
--   from public.match_documents_hybrid(
--       (select embedding from public.documents limit 1),
--       'merenda escolar', 0.25, 5
--   );
--
-- Se `content_tsv is not null` vier 0, a coluna não foi populada; refaça o
-- ingestion, porque o `update` de `content` é que dispara o recálculo.
-- --------------------------------------------------------------------------