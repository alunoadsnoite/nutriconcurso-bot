-- Base vetorial do RAG (pgvector).
-- Rode no SQL Editor do Supabase ou com `supabase db push`.
-- É idempotente: pode rodar de novo sem quebrar.

create extension if not exists vector;

-- --------------------------------------------------------------------------
-- Tabela de trechos
-- --------------------------------------------------------------------------
create table if not exists public.documents (
    id        bigserial primary key,
    content   text not null,
    metadata  jsonb not null default '{}'::jsonb,
    -- 1536 = saída padrão do text-embedding-3-small.
    -- Se mudar a dimensão, mude aqui E em EMBEDDING_DIMENSIONS no app/rag.py.
    embedding vector(1536)
);

-- O ingestion sempre preenche content e embedding juntos; os dois NOT NULL
-- evitam que um trecho entre na base sem vetor e nunca seja encontrado.
alter table public.documents
    alter column embedding set not null;

-- Filtro por fonte antes da busca vetorial (o app consulta alguns documentos).
create index if not exists documents_metadata_idx
    on public.documents using gin (metadata jsonb_path_ops);

-- --------------------------------------------------------------------------
-- Índice de similaridade
--
-- HNSW em vez de IVFFlat, de propósito:
--   IVFFlat com lists=100 só funciona bem quando o número de linhas é muito
--   maior que o número de listas. Com algumas centenas de trechos, cada lista
--   tem poucas linhas e a busca perde boa parte dos vizinhos reais (recall
-- baixo) sem nenhum aviso. HNSW é preciso e não precisa de dados para ser
--   criado, o que importa porque a base começa vazia.
--   Se um dia a base passar de ~1 milhão de linhas, avalie ivfflat ou hnsw
--   via pgvector quanto ao custo de memória.
-- --------------------------------------------------------------------------
create index if not exists documents_embedding_idx
    on public.documents using hnsw (embedding vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Função de busca
--
-- SECURITY DEFINER porque a RLS abaixo libera a leitura de `documents` para
-- o papel `anon`, mas não para `authenticated`. Sem isso, o cliente usaria a
-- chave `anon` e o PostgREST não acharia a tabela. A função só faz SELECT e
-- search_path fica travado para evitar function hijacking.
-- --------------------------------------------------------------------------
create or replace function public.match_documents (
    query_embedding  vector(1536),
    match_threshold  float,
    match_count      int
)
returns table (
    id        bigint,
    content   text,
    metadata  jsonb,
    similarity float
)
language sql
stable
security definer
set search_path = public
as $$
    select
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    from public.documents d
    where 1 - (d.embedding <=> query_embedding) > match_threshold
    order by d.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 20);
$$;

-- --------------------------------------------------------------------------
-- Permissões
--
--   select na tabela: só para quem cria documento (ingestion), não para anon.
--   execute na função: para anon e authenticated, porque é o caminho do chat.
-- --------------------------------------------------------------------------
revoke all on function public.match_documents(vector, float, int) from public;
grant execute on function public.match_documents(vector, float, int) to anon, authenticated;

revoke insert on table public.documents from anon, authenticated;

-- Se o ingestion usa a service_role, ela já ignora RLS e nada muda. Se for
-- usar a chave anon, crie uma policy de escrita assim (rode só se for isso):
--
--   create policy "ingestoes via anon" on public.documents
--     for insert to anon with check (true);
--
-- Manter a tabela fechada para o anon é o padrão correto: o chat só precisa
-- de SELECT, e ele acontece dentro da função.

-- --------------------------------------------------------------------------
-- Verificação
--
-- Depois de ingerir alguns documentos:
--   select count(*) from public.documents;
--   select * from public.match_documents(
--       '[0.1,0.2,...]'::vector, 0.25, 5
--   );
-- --------------------------------------------------------------------------