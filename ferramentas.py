"""
Ferramentas disponíveis para o agente: busca em normas indexadas e leitura de PDFs.
"""

from pathlib import Path

import ollama
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from ingestao import limpar_texto

EMBED_MODEL = "nomic-embed-text"
TOP_K = 5  # chunks recuperados por consulta
TOP_K_HIBRIDO = 10  # candidatos recuperados por método antes da combinação
PESO_VETORIAL = 0.4
PESO_BM25 = 0.6
LIMIAR_SIMILARIDADE = 0.50
CONTEXTO_MAX_CARACTERES = 12000


DESCRICAO_FERRAMENTAS = """
Ferramentas disponíveis:

1. buscar_normas
   Descrição: busca trechos relevantes nas normas técnicas indexadas no banco vetorial.
   Parâmetros: {"query": "<texto da pergunta ou termo de busca>"}
   Retorno: contexto com os trechos encontrados, lista de fontes e a melhor similaridade.

2. ler_documento
   Descrição: extrai o texto completo de um arquivo PDF a partir do caminho informado.
   Parâmetros: {"caminho_pdf": "<caminho do arquivo PDF>"}
   Retorno: texto extraído, número de páginas e nome do arquivo (ou erro se não encontrado).
"""


class Ferramentas:
    def __init__(self, colecao):
        self.colecao = colecao

        todos = colecao.get(include=["documents", "metadatas"])
        self.todos_docs = todos["documents"]
        self.todos_metas = todos["metadatas"]

        corpus_tokenizado = [doc.lower().split() for doc in self.todos_docs]
        self.bm25 = BM25Okapi(corpus_tokenizado)

    def _buscar_bm25(self, query: str, n: int) -> list[dict]:
        tokens_query = query.lower().split()
        scores = self.bm25.get_scores(tokens_query)

        score_maximo = max(scores) if len(scores) > 0 and max(scores) > 0 else 1

        indices_ordenados = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n]

        return [
            {
                "doc": self.todos_docs[i],
                "meta": self.todos_metas[i],
                "score_bm25": scores[i] / score_maximo,
            }
            for i in indices_ordenados
        ]

    def buscar_normas(self, query: str, n_results: int = TOP_K) -> dict:
        resposta_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)
        embedding = resposta_emb["embedding"]

        resultados = self.colecao.query(
            query_embeddings=[embedding],
            n_results=TOP_K_HIBRIDO,
            include=["documents", "metadatas", "distances"],
        )

        docs_vetor = resultados["documents"][0]
        metas_vetor = resultados["metadatas"][0]
        dists_vetor = resultados["distances"][0]

        resultados_bm25 = self._buscar_bm25(query, TOP_K_HIBRIDO)

        combinados: dict[tuple, dict] = {}

        for doc, meta, dist in zip(docs_vetor, metas_vetor, dists_vetor):
            chave = (meta["arquivo"], meta["trecho"])
            combinados[chave] = {
                "doc": doc,
                "meta": meta,
                "score_vetorial": 1 - dist,  # distância cosseno → similaridade
                "score_bm25": 0.0,
            }

        for item in resultados_bm25:
            chave = (item["meta"]["arquivo"], item["meta"]["trecho"])
            if chave in combinados:
                combinados[chave]["score_bm25"] = item["score_bm25"]
            else:
                combinados[chave] = {
                    "doc": item["doc"],
                    "meta": item["meta"],
                    "score_vetorial": 0.0,
                    "score_bm25": item["score_bm25"],
                }

        for item in combinados.values():
            item["score_hibrido"] = (
                PESO_VETORIAL * item["score_vetorial"] + PESO_BM25 * item["score_bm25"]
            )

        melhores = sorted(
            combinados.values(), key=lambda item: item["score_hibrido"], reverse=True
        )[:n_results]

        fontes = []
        blocos_contexto = []
        for item in melhores:
            meta = item["meta"]
            score_hibrido = round(item["score_hibrido"], 4)
            fontes.append({
                "arquivo": meta["arquivo"],
                "trecho": meta["trecho"],
                "similaridade": score_hibrido,
                "score_hibrido": score_hibrido,
            })
            blocos_contexto.append(
                f"[Fonte: {meta['arquivo']} — trecho {meta['trecho']}]\n{item['doc']}"
            )

        melhor_similaridade = max((f["score_hibrido"] for f in fontes), default=0.0)
        relevante = melhor_similaridade >= LIMIAR_SIMILARIDADE

        contexto = "\n\n---\n\n".join(blocos_contexto)
        if len(contexto) > CONTEXTO_MAX_CARACTERES:
            contexto = contexto[:CONTEXTO_MAX_CARACTERES] + "... [truncado]"

        return {
            "contexto": contexto,
            "fontes": fontes,
            "melhor_similaridade": melhor_similaridade,
            "relevante": relevante,
        }

    def ler_documento(self, caminho_pdf: str) -> dict:
        caminho = Path(caminho_pdf)
        if not caminho.is_file():
            return {"erro": f"Arquivo não encontrado: {caminho_pdf}"}

        reader = PdfReader(str(caminho))
        partes = []
        for pagina in reader.pages:
            texto = pagina.extract_text()
            if texto:
                partes.append(limpar_texto(texto))

        return {
            "texto": "\n".join(partes),
            "paginas": len(reader.pages),
            "arquivo": caminho.name,
        }
