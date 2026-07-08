"""
Agente com roteamento híbrido: o código decide qual ferramenta usar,
o modelo apenas gera a resposta com base no contexto fornecido.
Uso: python agente.py
"""

import re
import sys
from typing import Generator

import chromadb
import ollama

from ferramentas import Ferramentas

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "normas"
GEN_MODEL = "llama3.1:8b"
NUM_CTX = 16384
HISTORICO_MAX = 6  # pares user+assistant mantidos no histórico
LIMIAR_SIMILARIDADE = 0.50

SYSTEM_PROMPT = (
    "Você é um assistente técnico especializado em normas e laudos de engenharia, "
    "trabalhando para profissionais do Sistema CONFEA/CREA da SEDUH.\n\n"
    "COMPORTAMENTO:\n"
    "- Para perguntas técnicas sobre normas, laudos e engenharia: responda com base "
    "nos trechos fornecidos e cite sempre o arquivo de origem. Nunca invente dados.\n"
    "- Para saudações e perguntas sobre o que você faz: seja natural e direto, "
    "apresentando suas capacidades de forma amigável.\n"
    "- Para perguntas fora do escopo técnico (filosofia, opinião, entretenimento etc): "
    "responda de forma simpática e breve, com leveza, e convide o usuário a fazer "
    "uma pergunta técnica. Exemplo: 'Essa está além da minha área de especialização! "
    "Mas se tiver dúvidas sobre normas ABNT, laudos ou ART, pode perguntar.'\n"
    "- Nunca seja rude ou roboticamente frio. Mantenha um tom profissional mas humano.\n"
    "- ART significa 'Anotação de Responsabilidade Técnica'.\n"
    "- Se a informação técnica não estiver nos documentos disponíveis, diga claramente "
    "que não encontrou, sem inventar."
)

_SYSTEM_REWRITE = (
    "Sua única tarefa é reescrever a pergunta do usuário em uma única frase curta e direta, "
    "incorporando termos do histórico para torná-la autocontida.\n\n"
    "REGRAS ESTRITAS:\n"
    "- Retorne APENAS a pergunta reescrita, nada mais\n"
    "- Máximo de 20 palavras\n"
    "- Não responda a pergunta\n"
    "- Não liste normas, não dê exemplos, não explique\n"
    "- Se não precisar de contexto do histórico, repita a pergunta original\n\n"
    "Exemplo:\n"
    "Histórico: usuário perguntou sobre ART\n"
    "Pergunta: 'e quando ela precisa ser registrada?'\n"
    "Reescrita: 'Quando a ART precisa ser registrada?'\n\n"
    "Exemplo:\n"
    "Histórico: usuário analisou laudo predial com infiltrações\n"
    "Pergunta: 'quais normas são relevantes para os problemas encontrados?'\n"
    "Reescrita: 'Quais normas técnicas tratam de infiltração e impermeabilização predial?'"
)

_TERMOS_CONVERSACIONAIS = {
    "você lembra", "sua resposta", "você disse", "anteriormente",
    "primeira resposta", "última resposta", "o que falou",
    "o que disse", "lembra de", "disse antes", "falou antes",
}

_TERMOS_LER_DOCUMENTO = {
    "analise", "revise", "leia", "verifique esse", "esse pdf",
    "esse laudo", "esse documento",
}

_TERMOS_GERAL = {
    "olá", "oi", "bom dia", "boa tarde", "boa noite", "o que você",
    "o que voce", "pode me ajudar", "quem é você", "quem e voce",
    "como funciona", "what",
}

_REGEX_CAMINHO_PDF = re.compile(r"[^\s\"']+\.pdf", re.IGNORECASE)


def classificar_intencao(pergunta: str) -> str:
    texto = pergunta.lower()

    if any(termo in texto for termo in _TERMOS_CONVERSACIONAIS):
        return "conversacional"

    if any(termo in texto for termo in _TERMOS_GERAL):
        return "geral"

    if any(termo in texto for termo in _TERMOS_LER_DOCUMENTO) and _REGEX_CAMINHO_PDF.search(pergunta):
        return "ler_documento"

    return "buscar_norma"


def reescrever_pergunta(pergunta: str, historico: list[dict]) -> str:
    if not historico:
        return pergunta
    try:
        mensagens = (
            [{"role": "system", "content": _SYSTEM_REWRITE}]
            + historico[-4:]
            + [{"role": "user", "content": pergunta}]
        )
        resposta = ollama.chat(
            model=GEN_MODEL,
            messages=mensagens,
            options={"temperature": 0, "num_ctx": 4096},
        )
        resultado = resposta["message"]["content"].strip()

        # Pega só a primeira linha — evita que o modelo gere parágrafos
        primeira_linha = resultado.split('\n')[0].strip()
        if len(primeira_linha) > 10:
            resultado = primeira_linha

        return resultado
    except Exception:
        return pergunta


def _truncar_historico(historico: list[dict]) -> list[dict]:
    max_mensagens = HISTORICO_MAX * 2
    if len(historico) > max_mensagens:
        return historico[-max_mensagens:]
    return historico


def gerar_resposta_com_contexto(pergunta: str, contexto_ferramenta: str, historico: list[dict]) -> str:
    mensagem_user = f"Trechos das normas:\n\n{contexto_ferramenta}\n\nPergunta: {pergunta}"
    mensagens = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historico
        + [{"role": "user", "content": mensagem_user}]
    )
    resposta = ollama.chat(
        model=GEN_MODEL,
        messages=mensagens,
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    )
    return resposta["message"]["content"]


def gerar_resposta_com_contexto_stream(
    pergunta: str, contexto_ferramenta: str, historico: list[dict]
) -> Generator[str, None, None]:
    mensagem_user = f"Trechos das normas:\n\n{contexto_ferramenta}\n\nPergunta: {pergunta}"
    mensagens = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historico
        + [{"role": "user", "content": mensagem_user}]
    )
    stream = ollama.chat(
        model=GEN_MODEL,
        messages=mensagens,
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def _gerar_resposta_conversacional(pergunta: str, historico: list[dict]) -> str:
    mensagens = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historico
        + [{"role": "user", "content": pergunta}]
    )
    resposta = ollama.chat(
        model=GEN_MODEL,
        messages=mensagens,
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    )
    return resposta["message"]["content"]


def _gerar_resposta_conversacional_stream(
    pergunta: str, historico: list[dict]
) -> Generator[str, None, None]:
    mensagens = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historico
        + [{"role": "user", "content": pergunta}]
    )
    stream = ollama.chat(
        model=GEN_MODEL,
        messages=mensagens,
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def executar_tarefa(tarefa: str, ferramentas: Ferramentas, historico: list[dict]) -> dict:
    intencao = classificar_intencao(tarefa)
    if intencao == "buscar_norma":
        print("🔍 Buscando nas normas...")
    elif intencao == "ler_documento":
        print("📄 Lendo documento...")

    if intencao == "conversacional":
        return {"resposta": _gerar_resposta_conversacional(tarefa, historico), "fontes": []}

    if intencao == "geral":
        return {"resposta": _gerar_resposta_conversacional(tarefa, historico), "fontes": []}

    if intencao == "ler_documento":
        match = _REGEX_CAMINHO_PDF.search(tarefa)
        caminho = match.group(0) if match else ""
        resultado = ferramentas.ler_documento(caminho)
        if "erro" in resultado:
            return {
                "resposta": f"Não foi possível ler o documento: {resultado['erro']}",
                "fontes": [],
            }
        resposta = gerar_resposta_com_contexto(tarefa, resultado["texto"], historico)
        fontes = [{"arquivo": resultado["arquivo"], "trecho": "documento completo"}]
        return {"resposta": resposta, "fontes": fontes}

    # buscar_norma
    pergunta_rag = reescrever_pergunta(tarefa, historico)
    resultado = ferramentas.buscar_normas(pergunta_rag)
    if resultado["melhor_similaridade"] < LIMIAR_SIMILARIDADE:
        return {"resposta": "Não encontrei essa informação nos documentos disponíveis.", "fontes": []}
    resposta = gerar_resposta_com_contexto(tarefa, resultado["contexto"], historico)
    return {"resposta": resposta, "fontes": resultado["fontes"]}


def loop_agente(ferramentas: Ferramentas):
    print("Agente de Normas Técnicas — digite 'sair' para encerrar.\n")
    historico: list[dict] = []
    while True:
        try:
            tarefa = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not tarefa:
            continue
        if tarefa.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        resultado = executar_tarefa(tarefa, ferramentas, historico)
        resposta = resultado["resposta"]

        historico.append({"role": "user", "content": tarefa})
        historico.append({"role": "assistant", "content": resposta})
        historico = _truncar_historico(historico)

        print(f"Resposta:\n{resposta}\n")
        print("-" * 60)


def main():
    try:
        cliente = chromadb.PersistentClient(path=CHROMA_DIR)
        colecao = cliente.get_collection(COLLECTION_NAME)
    except Exception:
        print(
            "Erro: base vetorial não encontrada. "
            "Execute 'python ingestao.py' primeiro."
        )
        sys.exit(1)

    count = colecao.count()
    print(f"Base carregada: {count} chunks indexados.")

    ferramentas = Ferramentas(colecao)

    if len(sys.argv) > 1:
        tarefa = " ".join(sys.argv[1:])
        print(f"\nPergunta: {tarefa}\n")
        resultado = executar_tarefa(tarefa, ferramentas, [])
        print(f"Resposta:\n{resultado['resposta']}")
    else:
        loop_agente(ferramentas)


if __name__ == "__main__":
    main()