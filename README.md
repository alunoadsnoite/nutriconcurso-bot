# NutriConcurso Bot API

API do tutor de concursos públicos na área de **Nutrição** (PNAE, PNAN, SUS, RDC
216/2004, UAN, Nutrição Clínica, Materno-Infantil, Esportiva e Saúde Coletiva),
pensada para responder dúvidas e corrigir questões de FGV, Vunesp, Cebraspe,
IBFC e FCC.

O app que consome esta API é o
[NutriConcurso](https://github.com/alunoadsnoite/nutriconcurso) (Expo/React Native).

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # preencha OPENAI_API_KEY
uvicorn app.main:app --reload
```

- Documentação interativa: <http://127.0.0.1:8000/docs>
- Health check: `GET /health`

Rode os testes com `pip install -r requirements-dev.txt` e depois `pytest`.
Eles **não precisam de `OPENAI_API_KEY`**: o cliente OpenAI é substituído por um
dublê, então o caminho real da requisição é exercitado sem gastar token.

## Endpoints

### `POST /api/v1/chat`

```jsonc
//requisição
{ "message": "Qual o Fator de Correção do PNAE para creche?", "user_id": "ana" }

//resposta com RAG ligado
{
  "reply": "...",
  "has_history": true,
  "rag_used": true,
  "sources": [ { "source": "Resolucao_CD_FNDE.pdf", "page": 12 } ]
}

//resposta sem RAG (ou sem trecho acima do limiar)
{ "reply": "...", "has_history": true, "rag_used": false, "sources": [] }
```

`user_id` é opcional. Sem ele a chamada é isolada; com ele a API guarda o
histórico da conversa e o reenvia ao modelo nas chamadas seguintes.

`rag_used: false` com `sources: []` significa que a resposta saiu do modelo sem
documento de apoio. O app mostra um aviso nessa situação — é o que evita o
estudante ler memória do modelo como se fosse norma.

### `DELETE /api/v1/chat?user_id=ana`

Limpa o histórico de um usuário. Usado pelo botão "Nova conversa" do app.

### `GET /health`

```json
{
  "status": "ok",
  "configured": true,
  "model": "gpt-4o-mini",
  "rag_configured": true,
  "embedding_model": "text-embedding-3-small"
}
```

`configured: false` significa que `OPENAI_API_KEY` não está definida — o
servidor sobe normalmente, mas o chat responde `503`.
`rag_configured: false` significa que `SUPABASE_URL`/`SUPABASE_KEY` não estão
definidas: o chat funciona, porém sem base oficial.

## Códigos de resposta

| Código | Quando |
| --- | --- |
| `200` | Resposta do modelo |
| `400` | Mensagem vazia |
| `401` | `X-API-Key` ausente ou incorreta (quando `API_KEY` está definida) |
| `422` | Payload inválido |
| `429` | Rate limit local, ou limite de uso da OpenAI |
| `502` | A OpenAI recusou a requisição ou devolveu resposta vazia |
| `503` | `OPENAI_API_KEY` não configurada, ou base vetorial indisponível |
| `504` | Timeout da OpenAI |

## RAG com pgvector (Supabase)

Com `SUPABASE_URL` e `SUPABASE_KEY` definidas, o `/api/v1/chat` passa a buscar
trechos oficiais antes de chamar o modelo:

1. embedding da pergunta com `text-embedding-3-small` (1536 dimensões);
2. busca por similaridade de cosseno na função `match_documents`;
3. os trechos viram contexto, com documento e página de cada um;
4. `gpt-4o-mini` responde com esse contexto injetado;
5. a resposta volta com `sources` para o app mostrar os badges.

### Ligar

**1. Crie a base.** No SQL Editor do Supabase, rode [`migrations/001_rag.sql`](migrations/001_rag.sql).
É idempotente e cria a extensão, a tabela, o índice e a função.

**2. Configure o `.env`.**

```bash
SUPABASE_URL=https://seuprojeto.supabase.co
SUPABASE_KEY=eyJ...        # a chave `anon` basta para o chat
```

**3. Ingira os PDFs.**

```bash
python -m scripts.ingest_pdf Manual_PNAE_2023.pdf
python -m scripts.ingest_pdf --recursive pasta/legislacao/
```

O script fatia por página e grava `metadata.source` (nome do arquivo) e
`metadata.page`, que é o que alimenta os badges. Para corrigir um documento,
apague os trechos antigos antes — reingerir paga o embedding de novo:

```sql
delete from public.documents where metadata->>'source' = 'Manual_PNAE_2023.pdf';
```

### Ajustes

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `RAG_MATCH_THRESHOLD` | `0.25` | Similaridade mínima. Acima de ~0.5 costuma não voltar nada |
| `RAG_MATCH_COUNT` | `5` | Quantos trechos entram no contexto |
| `RAG_MAX_CHUNK_CHARS` | `1200` | Corte por trecho |
| `RAG_MAX_CONTEXT_CHARS` | `8000` | Teto do contexto inteiro |
| `EMBEDDING_DIMENSIONS` | `1536` | Tem que bater com a coluna `VECTOR(n)` |

A busca já vem ordenada por similaridade, então os primeiros trecho pesam mais:
quando o total estourar `RAG_MAX_CONTEXT_CHARS`, o corte acontece no fim, que é
o certo.

### Falha na base não vira resposta sem fonte

Se a busca vetorial falhar com o RAG ligado, a API responde `503` em vez de
responder do mesmo jeito sem contexto. Responder sem documento deixaria o tutor
apresentar memória do modelo parecendo fundamento oficial — exatamente o risco
que o RAG existe para reduzir. Prefira o erro visível.

## Segurança

**`X-API-Key` não é segurança real no app mobile.** A chave viaja dentro do
binário do APK, então qualquer pessoa com o arquivo consegue extraí-la. Ela
protege contra chamada acidental e contra bot ingênuo — nada além disso.

O que a API já faz para reduzir dano:

- **A chave do Supabase nunca vai para o app.** Ela fica só no servidor. Para o
  chat basta a chave `anon`, porque o `SELECT` acontece dentro de
  `match_documents`, que é `security definer` com `search_path` travado. O
  `insert` fica restrito a quem tem permissão — o ingestion usa a
  `service_role`, e ela nunca deve ser versionada nem digitada no app.
- **Rate limit por IP** (`REQUESTS_PER_MINUTE`, padrão 20/min, janela
  deslizante) com `Retry-After` no `429`.
- **Erros da OpenAI não vazam para o cliente.** A mensagem original vai só para
  o log do servidor; o corpo da resposta recebe um texto genérico e um status
  estável. Isso evita expor nomes de projeto, ids de requisição e trechos de
  prompt, que costuma vir no `str(exc)`.
- **Tamanho de mensagem limitado** (`MAX_MESSAGE_LENGTH`, padrão 4000).

O que falta para produção, em ordem de prioridade:

1. **Autenticação por usuário** no servidor (login/sessão) e orçamento gasto por
   usuário. Sem isso, uma chave vazada vira uma fatura aberta.
2. **Histórico persistente** — hoje é em memória (ver abaixo).
3. **TLS** obrigatório e `X-Forwarded-For` só confiável atrás de proxy próprio
   (hoje o código aceita o header, o que permite burlar o rate limit se a API
   estiver exposta direto).

## Limitações conhecidas

- **O histórico vive na memória do processo.** Se reiniciar o servidor, ou se
  rodar com mais de um worker (`uvicorn --workers N`), a conversa "pula" entre
  requisições — cada worker tem a sua própria memória. Para produção isso
  precisa ir para Redis ou Postgres. Os testes cobrem o TTL e o descarte, não a
  persistência.
- **O modelo pode errar.** Legislação muda e é fácil de confundir: a RDC 216/2004
  trata de parâmetros de qualidade microbiológica de água para consumo humano, e
  a RDC 275/1994 trata de higiene de manipuladores de alimentos — são assuntos
  diferentes dos que o nome costuma sugerir. As respostas são material de
  estudo, não autorização. Confira sempre na fonte oficial.
- **O RAG não garante que a citação exista.** O prompt pede que o modelo
  referencie o bloco usado (`[1]`, `[2]`), mas ele pode citar um número que não
  corresponden ao que realmente usou. Os badges mostram o que foi *recuperado*,
  não o que foi *usado*. Verifique abrindo o documento.
- **A busca é só vetorial.** Sem BM25/híbrido nem reranking: pergunta com
  vocabulário muito diferente do documento (por exemplo, "merenda" contra
  "alimentos") perde recall. Se o recall importar, o próximo passo é busca
  híbrida e um cross-encoder rerankeando os 5 resultados.
- **Um trecho por página.** Não há divisão por artigo nem sobreposição, então
  um artigo que começa no fim de uma página e termina no começo da seguinte fica
  partido em dois. A busca acha as duas metades, mas o modelo recebe cada uma
  separada.
- **Sem cache de embedding.** Pergunta repetida gera nova chamada e novo custo,
  tanto no chat quanto na ingestão.
- **`RAG_MATCH_THRESHOLD` é chato de ajustar.** Não há métrica no repositório
  (recall@k, groundedness); a calibragem é no olho.

## Estrutura

```
app/
├── main.py             # rotas, tradução de erros da OpenAI
├── config.py           # settings via pydantic-settings
├── openai_client.py    # cliente criado sob demanda
├── conversation.py     # histórico em memória com TTL
├── deps.py             # X-API-Key, rate limit, IP do cliente
├── prompts.py          # system prompt (com e sem contexto recuperado)
├── rag.py              # embedding, busca pgvector, contexto e fontes
└── schemas.py          # contratos de entrada e saída
migrations/001_rag.sql  # extensão, tabela, índice HNSW e match_documents
scripts/ingest_pdf.py   # PDF -> trechos -> embeddings -> banco
tests/                  # 39 testes, sem necessidade de API key nem Supabase
```

## Licença

Ver o arquivo `LICENSE` no repositório.
