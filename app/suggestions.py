"""Geração das perguntas sugeridas que aparecem abaixo da resposta.

Chamada separada e curta, com saída JSON. Fica separada da resposta principal
por três motivos: a resposta principal não é contaminada por "quando você
quiser, pergunte..." no meio do texto; dá para desligar por configuração; e se
a geração falhar, a resposta principal já foi entregue e o app só não mostra
os chips.

O modelo recebe o histórico curto e a pergunta, não o contexto recuperado
inteiro — as perguntas sugeridas devem variar o assunto, não repetir o bloco
que acabou de ser lido.
"""

import json
import logging
import re

logger = logging.getLogger("nutriconcurso.suggestions")

SYSTEM = """\
Você sugere perguntas de continuação para um candidato a concurso de Nutrição.

Responda APENAS com um JSON no formato exato:
{"perguntas": ["...", "...", "..."]}

Regras:
- Cada pergunta é autocontida: quem lê só a pergunta entende do que se trata.
- Varie o ângulo: um aprofundamento no assunto respondido, um ponto de
  jurisprudência, um cálculo, uma questão comentada de banca.
- Não repita a pergunta que o candidato acabou de fazer.
- Não use numeração, aspas internas nem texto fora do JSON.
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _clean(text: str) -> str:
    text = text.strip().strip("`")
    return text.strip().strip("`").strip()


def _extract(text: str) -> str:
    """Pega o primeiro objeto JSON da resposta, ignorando cercas de código."""
    match = _JSON_BLOCK.search(text)
    return match.group(0) if match else text


def parse_suggestions(raw: str, *, limit: int) -> list[str]:
    """Converte a resposta do modelo em lista de perguntas.

    Reconhece tanto o formato pedido quanto um array solto, porque o modelo
    escapa com frequência. Devolve lista vazia em vez de explodir: perder os
    chips é aceitável, derrubar a requisição não.
    """
    if not raw or limit <= 0:
        return []

    candidates: list[str] = []

    text = _extract(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            raw_list = data.get("perguntas") or data.get("questions") or []
        elif isinstance(data, list):
            raw_list = data
        else:
            raw_list = []
        if isinstance(raw_list, list):
            candidates = [str(item) for item in raw_list]
    except ValueError:
        candidates = []

    if not candidates:
        # Fallback: linhas "- pergunta" ou aspas soltas.
        stripped = re.findall(r"[\"“](.{6,120}?)[\"”]", raw)
        candidates = stripped or [
            re.sub(r"^[-*\d.\s]+", "", line).strip()
            for line in raw.splitlines()
            if line.strip().startswith(("-", "*"))
        ]

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        question = _clean(candidate)
        if not (6 <= len(question) <= 120):
            continue
        if question.lower() in seen:
            continue
        seen.add(question.lower())
        out.append(question)
        if len(out) == limit:
            break

    return out


def generate(client, settings, *, question: str, reply: str, history: list[dict[str, str]]) -> list[str]:
    """Chama o modelo e devolve as perguntas. Nunca levanta exceção."""
    count = settings.suggestions_count
    if count <= 0:
        return []

    recent = history[-4:] if history else []
    transcript = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in recent)

    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Conversa:\n{transcript}\n\n"
                        f"Gere exatamente {count} perguntas de continuação."
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=settings.suggestions_max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - perder os chips não é erro fatal
        logger.warning("Falha ao gerar sugestões (%s): %s", type(exc).__name__, exc)
        return []

    raw = completion.choices[0].message.content or ""
    return parse_suggestions(raw, limit=count)