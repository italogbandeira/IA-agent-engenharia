"""
Consulta RAG sobre normas técnicas indexadas.
Uso: python consulta.py
     python consulta.py "qual a resistência mínima do concreto?"
"""

import sys

import chromadb
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "normas"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.1:8b"
TOP_K = 5  # chunks recuperados por consulta

SYSTEM_PROMPT = (
    "Você é um assistente de consulta a normas técnicas de engenharia.\n\n"
    "DEFINIÇÃO IMPORTANTE: ART significa 'Anotação de Responsabilidade Técnica', "
    "não 'Atestado'. Nunca use outra definição para essa sigla.\n\n"
    "REGRAS:\n"
    "1. Responda SOMENTE com base nos trechos fornecidos. Nunca use conhecimento próprio.\n"
    "2. Se a resposta não estiver nos trechos, diga: 'Não encontrei essa informação nos documentos disponíveis.'\n"
    "3. Cite sempre o nome do arquivo de origem ao responder.\n"
    "4. Não invente números, datas, nomes de órgãos ou artigos que não estejam nos trechos."
)

LIMIAR_SIMILARIDADE = 0.65
NUM_CTX = 8192  # janela de contexto do modelo — evita corte silencioso do Ollama


def recuperar_contexto(colecao, pergunta: str) -> tuple[str, list[dict]]:
    resposta_emb = ollama.embeddings(model=EMBED_MODEL, prompt=pergunta)
    embedding = resposta_emb["embedding"]

    resultados = colecao.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = resultados["documents"][0]
    metas = resultados["metadatas"][0]
    dists = resultados["distances"][0]

    fontes = []
    blocos_contexto = []
    for doc, meta, dist in zip(docs, metas, dists):
        similaridade = 1 - dist  # distância cosseno → similaridade
        fontes.append({
            "arquivo": meta["arquivo"],
            "trecho": meta["trecho"],
            "similaridade": round(similaridade, 4),
        })
        blocos_contexto.append(
            f"[Fonte: {meta['arquivo']} — trecho {meta['trecho']}]\n{doc}"
        )

    contexto = "\n\n---\n\n".join(blocos_contexto)
    return contexto, fontes


_HISTORICO_MAX_TROCAS = 6  # pares user+assistant mantidos no histórico

_TERMOS_CONVERSACIONAIS = {
    "você lembra", "sua resposta", "você disse", "anteriormente",
    "primeira resposta", "última resposta", "o que falou",
    "o que disse", "lembra de", "disse antes", "falou antes",
}


def detectar_pergunta_conversacional(pergunta: str) -> bool:
    texto = pergunta.lower()
    return any(termo in texto for termo in _TERMOS_CONVERSACIONAIS)


def gerar_resposta(historico: list[dict]) -> str:
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}] + historico
    resposta = ollama.chat(
        model=GEN_MODEL,
        messages=mensagens,
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    )
    return resposta["message"]["content"]


def _truncar_historico(historico: list[dict]) -> list[dict]:
    max_mensagens = _HISTORICO_MAX_TROCAS * 2
    if len(historico) > max_mensagens:
        print("  [histórico limitado às últimas 6 trocas]")
        return historico[-max_mensagens:]
    return historico


_SYSTEM_REWRITE = (
    "Reescreva a pergunta do usuário de forma autocontida usando o contexto do histórico. "
    "Use apenas termos que apareçam explicitamente no histórico ou na pergunta. "
    "Não adicione siglas, órgãos ou termos que não foram mencionados. "
    "Retorne APENAS a pergunta reescrita, sem explicações."
)


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
        return resposta["message"]["content"].strip()
    except Exception:
        return pergunta


def loop_consulta(colecao):
    print("RAG de Normas Técnicas — digite 'sair' para encerrar.\n")
    historico: list[dict] = []
    while True:
        try:
            pergunta = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        if detectar_pergunta_conversacional(pergunta):
            print("\nPergunta conversacional detectada — sem busca RAG.\n")
            historico.append({"role": "user", "content": pergunta})
            historico = _truncar_historico(historico)
            resposta = gerar_resposta(historico)
            historico.append({"role": "assistant", "content": resposta})
        else:
            pergunta_rag = reescrever_pergunta(pergunta, historico)
            if pergunta_rag != pergunta:
                print(f"  Pergunta reescrita: {pergunta_rag}")
            print("\nBuscando trechos relevantes...")
            contexto, fontes = recuperar_contexto(colecao, pergunta_rag)

            print("Fontes recuperadas:")
            for f in fontes:
                print(f"  • {f['arquivo']} (trecho {f['trecho']}, sim={f['similaridade']})")

            melhor_sim = max(f["similaridade"] for f in fontes)
            if melhor_sim < LIMIAR_SIMILARIDADE:
                resposta = "Não encontrei essa informação nos documentos disponíveis."
            else:
                print(f"  Tamanho do contexto: {len(contexto)} caracteres / ~{len(contexto)//4} tokens")
                print("\nGerando resposta...\n")
                mensagem_user = f"Trechos das normas:\n\n{contexto}\n\nPergunta: {pergunta}"
                historico.append({"role": "user", "content": mensagem_user})
                historico = _truncar_historico(historico)
                resposta = gerar_resposta(historico)
                historico.append({"role": "assistant", "content": resposta})
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

    # Modo de consulta única via argumento de linha de comando
    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
        print(f"\nPergunta: {pergunta}\n")
        contexto, fontes = recuperar_contexto(colecao, pergunta)
        print("Fontes recuperadas:")
        for f in fontes:
            print(f"  • {f['arquivo']} (trecho {f['trecho']}, sim={f['similaridade']})")
        melhor_sim = max(f["similaridade"] for f in fontes)
        if melhor_sim < LIMIAR_SIMILARIDADE:
            resposta = "Não encontrei essa informação nos documentos disponíveis."
        else:
            print(f"  Tamanho do contexto: {len(contexto)} caracteres / ~{len(contexto)//4} tokens")
            print("\nGerando resposta...\n")
            mensagem_user = f"Trechos das normas:\n\n{contexto}\n\nPergunta: {pergunta}"
            resposta = gerar_resposta([{"role": "user", "content": mensagem_user}])
        print(f"Resposta:\n{resposta}")
    else:
        loop_consulta(colecao)


if __name__ == "__main__":
    main()