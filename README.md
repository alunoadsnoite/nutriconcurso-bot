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

//resposta
{ "reply": "...", "has_history": true }
```

`user_id` é opcional. Sem ele a chamada é isolada; com ele a API guarda o
histórico da conversa e o reenvia ao modelo nas chamadas seguintes.

### `DELETE /api/v1/chat?user_id=ana`

Limpa o histórico de um usuário. Usado pelo botão "Nova conversa" do app.

### `GET /health`

```json
{ "status": "ok", "configured": true, "model": "gpt-4o-mini" }
```

`configured: false` significa que `OPENAI_API_KEY` não está definida — o
servidor sobe normalmente, mas o chat responde `503`.

## Códigos de resposta

| Código | Quando |
| --- | --- |
| `200` | Resposta do modelo |
| `400` | Mensagem vazia |
| `401` | `X-API-Key` ausente ou incorreta (quando `API_KEY` está definida) |
| `422` | Payload inválido |
| `429` | Rate limit local, ou limite de uso da OpenAI |
| `502` | A OpenAI recusou a requisição ou devolveu resposta vazia |
| `503` | `OPENAI_API_KEY` não configurada no servidor |
| `504` | Timeout da OpenAI |

## Segurança

**`X-API-Key` não é segurança real no app mobile.** A chave viaja dentro do
binário do APK, então qualquer pessoa com o arquivo consegue extraí-la. Ela
protege contra chamada acidental e contra bot ingênuo — nada além disso.

O que a API já faz para reduzir dano:

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
- **O modelo pode errar.** Legislation muda (a RDC 216/2004 foi alterada pela
  RDC 216/2021, por exemplo). As respostas são material de estudo, não
  autorização. O system prompt pede citar a fonte, mas a verificação é do
  usuário.
- Sem cache: perguntas repetidas geram novas chamadas e novo custo.

## Estrutura

```
app/
├── main.py             # rotas, tradução de erros da OpenAI
├── config.py           # settings via pydantic-settings
├── openai_client.py    # cliente criado sob demanda
├── conversation.py     # histórico em memória com TTL
├── deps.py             # X-API-Key, rate limit, IP do cliente
├── prompts.py          # system prompt
└── schemas.py          # contratos de entrada e saída
tests/test_api.py       # 18 testes, sem necessidade de API key
```

## Licença

Ver o arquivo `LICENSE` no repositório.
