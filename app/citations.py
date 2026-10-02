"""Validação das citações que o modelo faz sobre o contexto recuperado.

Por que um marcador só nosso: `[1]` é sintaxe de inciso em lei ("inciso [1] do
art. 5º da RDC 275"), e legislação é justamente o conteúdo deste app. Se o
modelo citasse com `[1]`, não daria para distinguir "fonte 1" de "inciso I", e
qualquer correção automática apagaria texto legal. Por isso o prompt pede
`[[fonte:N]]` e a validação só toca nesse formato.

O que é corrigido aqui:
- `[[fonte:9]]` quando só existem 3 blocos: removido, porque aponta para um
  documento que não veio na busca;
- `[[fonte:0]]`, `[[fonte:abc]]` e `[[fonte:1` truncado: removidos;
- repetição do mesmo número: mantida só a primeira, para o texto não encher de
  marcador.

O que NÃO é corrigido: o modelo dizer "[[fonte:2]]" apontando para o bloco 2
quando na verdade raciocinou sobre o 1. Isso exige o trecho e a resposta lado a
lado; o que dá para garantir aqui é que todo marcador aponta para algo que
existe.

Duas decisões de projeto que custaram caro e valem registrar:

**Uma passagem só.** Tratar marcador válido e malformado em regexes separados
fazia o passe "remove os quebrados" comer também o início dos válidos, deixando
`1]]` pendurado no texto.

**Nada de reescrever espaçamento.** Uma versão anterior colapsava o espaço
duplo deixado pela remoção e colava pontuação. Isso é impossível no stream: o
espaço antes do marcador já foi enviado para a tela, e texto exibido não se
recupera. Como o histórico é gravado com o texto que o aluno viu, limpar um
caminho e não o outro faria a resposta da tela e a do histórico divergirem — e a
pergunta seguinte do modelo seria montada sobre o texto errado. Perder a limpeza
de espaço é cosmético e raro (só quando o modelo emite marcador malformado);
divergir entre os dois caminhos é um bug silencioso. O que este módulo garante,
então, é só o que importa: **todo marcador que sobrou aponta para um bloco que
existe**.
"""

import re

# Qualquer coisa que comece por `[[fonte`: válido, malformado ou truncado.
# Os dois `\]?` opcionais cobrem `]]`, `]` e o caso sem fechamento nenhum.
_MARKER = re.compile(r"\[\[fonte([^\[\]]{0,40})\]?\]?", re.IGNORECASE)
# Só o interior bem formado `: 12`.
_INNER = re.compile(r"^\s*:\s*(\d{1,4})\s*$")


def _pode_ser_marcador(pending: str) -> bool:
    """O texto em aberto ainda pode virar `[[fonte:N]]`?

    Aceita tanto `[[fonte` já escrito quanto um prefixo dele (`[[f`, `[[fon`),
    porque o marcador pode vir partido entre dois pedaços do stream.
    """
    minusculo = pending.lower()
    return minusculo.startswith("[[fonte") or "[[fonte".startswith(minusculo)


_ALVO = "[[fonte"


def _tamanho_do_prefixo_pendente(texto: str) -> int:
    """Quantos caracteres do fim de `texto` ainda podem virar um marcador.

    Sem isto o stream vaza marcador quando o modelo emite caractere a
    caractere: o primeiro `[` já foi para a tela quando o segundo chega, e
    nunca mais há `[[` para reconhecer. A resposta é olhar para trás, segurar
    o que ainda é prefixo de `[[fonte`, e soltar assim que o próximo pedaço
    mostrar que era texto comum.

    O custo é de poucos caracteres de atraso, invisível para quem lê, e o
    resultado continua idêntico ao do caminho sem stream.
    """
    for tamanho in range(min(len(_ALVO), len(texto)), 0, -1):
        if texto[-tamanho:].lower() == _ALVO[:tamanho]:
            return tamanho
    return 0


def validate_citations(reply: str, *, block_count: int) -> tuple[str, list[int]]:
    """Devolve (texto limpo, números de bloco efetivamente citados).

    `block_count` é quantos blocos de contexto havia. Citação fora dessa faixa
    é removida do texto em vez de virar remissão impossível de conferir.
    """
    if "[[" not in reply.lower():
        return reply, []

    used: list[int] = []
    out: list[str] = []
    last = 0

    for match in _MARKER.finditer(reply):
        out.append(reply[last : match.start()])
        last = match.end()

        inner = _INNER.match(match.group(1))
        # Marcador truncado no fim da resposta (`[[fonte:1`, sem `]]`) não
        # conta. Reparar acrescentando o `]]` que o modelo não escreveu
        # fabricaria uma citação que ele não fez.
        if not inner or not match.group(0).endswith("]]"):
            continue

        number = int(inner.group(1))
        if 1 <= number <= block_count and number not in used:
            used.append(number)
            out.append(f"[[fonte:{number}]]")
        # Fora da faixa ou repetido: nada é escrito, o marcador simplesmente
        # não entra no texto final.

    out.append(reply[last:])
    return "".join(out), used


def strip_all_citations(reply: str) -> str:
    """Usado quando não havia contexto nenhum."""
    return _MARKER.sub("", reply)


class StreamCitationFilter:
    """Aplica a mesma regra de `validate_citations` pedaço a pedaço.

    Existe por um motivo concreto: no modo stream o texto chega no app antes
    de a resposta terminar, e a validação só roda no fim. Sem isto, um
    `[[fonte:99]]` já estaria na tela do aluno quando se descobre que o bloco 99
    nem existe — e não há como recolher texto já exibido.

    A chave para resolver isso é que `block_count` é conhecido ANTES do stream:
    a validade de um marcador não depende de qual trecho vier depois, só de o
    número existir no contexto. Então dá para decidir marcador a marcador, no
    momento em que ele fecha.

    O que fica retido é só o pedaço em aberto a partir de `[[`, porque
    `[[fon` ainda pode virar `[[fonte:1]]`. Texto normal passa direto, sem
    atraso perceptível.
    """

    # Se o modelo nunca fechar o marcador, o buffer cresceria sem parar. Muito
    # acima do tamanho de um marcador real, o texto pendente é liberado como
    # texto comum: melhor exibir um resíduo estranho do que perder o fim da
    # resposta ou segurar memória.
    MAX_HOLD = 200

    def __init__(self, *, block_count: int) -> None:
        self._block_count = block_count
        self._pending = ""
        self._used: list[int] = []
        self.text = ""

    @property
    def used(self) -> list[int]:
        """Números de bloco citados, na ordem em que apareceram."""
        return list(self._used)

    def push(self, chunk: str) -> str:
        """Recebe um pedaço e devolve o que já pode ser exibido."""
        if not chunk:
            return ""

        self._pending += chunk
        safe: list[str] = []
        rest = self._pending
        self._pending = ""

        while rest:
            start = rest.find("[[")
            if start < 0:
                safe.append(rest)
                rest = ""
                break

            safe.append(rest[:start])
            tail = rest[start:]
            closing = tail.find("]]")

            if closing < 0:
                if _pode_ser_marcador(tail):
                    # Ainda pode virar `[[fonte:1]]` no próximo pedaço.
                    self._pending = tail
                else:
                    # `[[` que não é nosso marcador, como em `[[nota]]`. Não dá
                    # para segurar indefinidamente esperando um formato que não
                    # vai aparecer, e a rota sem stream também preservaria.
                    safe.append(tail)
                rest = ""
                break

            candidate = tail[: closing + 2]
            marker = _MARKER.fullmatch(candidate)

            if marker is None:
                # Fechou, mas não tem o nosso formato, como em `[[nota]]`.
                # `validate_citations` também preservaria; é texto normal.
                safe.append(candidate)
            else:
                number = _INNER.match(marker.group(1))
                if number:
                    value = int(number.group(1))
                    if 1 <= value <= self._block_count and value not in self._used:
                        self._used.append(value)
                        safe.append(f"[[fonte:{value}]]")
                # Formato nosso mas malformado, repetido ou fora da faixa:
                # sai sem nada no lugar.

            rest = tail[closing + 2 :]

        if len(self._pending) > self.MAX_HOLD:
            safe.append(self._pending)
            self._pending = ""

        out = "".join(safe)

        # Um único ponto de guarda, cobrindo todos os caminhos: o que sobra
        # no fim e ainda pode virar marcador fica para o próximo pedaço.
        retido = _tamanho_do_prefixo_pendente(out)
        if retido:
            self._pending = out[-retido:] + self._pending
            out = out[:-retido]

        self.text += out
        return out

    def flush(self) -> str:
        """Fecha o fluxo.

        O que sobrou pendente sai como texto normal, a não ser que seja um
        marcador truncado do nosso formato — aí é descartado, porque
        `validate_citations` também o removeria e os dois caminhos não podem
        divergir.
        """
        pendente = self._pending
        self._pending = ""

        if _MARKER.match(pendente):
            return ""

        self.text += pendente
        return pendente