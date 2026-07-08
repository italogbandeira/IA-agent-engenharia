"""
Ingestão de PDFs de normas técnicas no ChromaDB.
Uso: python ingestao.py
"""

import os
import re
import sys
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

DOCUMENTOS_DIR = Path("documentos")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "normas"
CHUNK_SIZE = 300       # palavras por chunk
CHUNK_OVERLAP = 100    # palavras de sobreposição
CHUNK_MAX_PALAVRAS = 2000  # limite de segurança: chunk acima disso é redividido
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 8192  # janela de contexto — evita corte/erro silencioso do Ollama


_LIGATURAS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

_PALAVRAS_GRUDADAS = {
    "120dias": "120 dias",
    "parr": "partir",
    "ronas": "rotinas",
    "avidades": "atividades",
    "avidade": "atividade",
    "respecvo": "respectivo",
    "respecva": "respectiva",
    "administravas": "administrativas",
    "administrava": "administrativa",
    "corporavo": "corporativo",
    "corporava": "corporativa",
    "efevem": "efetivem",
    "efevado": "efetivado",
    "emidas": "emitidas",
    "emida": "emitida",
    "cerdão": "certidão",
    "hps://": "https://",
}


def limpar_texto(texto: str) -> str:
    # Remove bytes nulos que o pypdf insere no lugar de "ti" em certos PDFs
    texto = texto.replace('\x00', 'ti')

    for ligadura, substituicao in _LIGATURAS.items():
        texto = texto.replace(ligadura, substituicao)

    for grudada, substituicao in _PALAVRAS_GRUDADAS.items():
        texto = texto.replace(grudada, substituicao)

    texto = re.sub(r'(\d+)([a-záéíóúãõâêîôûç])', r'\1 \2', texto)
    texto = re.sub(r'\.([A-Z])', r'. \1', texto)
    texto = re.sub(r',([A-Za-z])', r', \1', texto)

    return texto


def extrair_texto_pdf(caminho: Path) -> str:
    reader = PdfReader(str(caminho))
    partes = []
    for pagina in reader.pages:
        texto = pagina.extract_text()
        if texto:
            partes.append(limpar_texto(texto))
    return "\n".join(partes)


def dividir_em_chunks(texto: str, tamanho: int, overlap: int) -> list[str]:
    palavras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palavras):
        fim = inicio + tamanho
        chunk = " ".join(palavras[inicio:fim])
        if chunk.strip():
            chunks.append(chunk)
        if fim >= len(palavras):
            break
        inicio = fim - overlap

    # Proteção: se algum chunk saiu gigante (texto mal extraído, sem
    # quebras de linha reais), redivide forçadamente em pedaços menores.
    chunks_seguros = []
    for chunk in chunks:
        palavras_chunk = chunk.split()
        if len(palavras_chunk) <= CHUNK_MAX_PALAVRAS:
            chunks_seguros.append(chunk)
        else:
            print(
                f"  Aviso: chunk com {len(palavras_chunk)} palavras excede o "
                f"limite de {CHUNK_MAX_PALAVRAS} — dividindo forçadamente."
            )
            for i in range(0, len(palavras_chunk), CHUNK_MAX_PALAVRAS):
                pedaco = " ".join(palavras_chunk[i:i + CHUNK_MAX_PALAVRAS])
                if pedaco.strip():
                    chunks_seguros.append(pedaco)

    return chunks_seguros


def gerar_embedding(texto: str) -> list[float]:
    resposta = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=texto,
        options={"num_ctx": NUM_CTX},
    )
    return resposta["embedding"]


def gerar_embeddings_chunk(chunk: str) -> list[tuple[str, list[float]]]:
    """
    Gera o embedding de um chunk. Se o Ollama recusar por o texto ser
    grande demais, divide o chunk pela metade recursivamente até funcionar.
    Retorna uma lista de (texto, embedding) — pode ser mais de um par
    se o chunk original precisou ser dividido.
    """
    try:
        return [(chunk, gerar_embedding(chunk))]
    except ollama.ResponseError as erro:
        palavras = chunk.split()
        if "context length" not in str(erro).lower() or len(palavras) <= 20:
            # Erro de outro tipo, ou chunk já pequeno demais — não adianta dividir
            raise
        meio = len(palavras) // 2
        metade1 = " ".join(palavras[:meio])
        metade2 = " ".join(palavras[meio:])
        print(
            f"\n  Aviso: chunk com {len(palavras)} palavras excedeu o limite do "
            f"modelo de embedding — dividindo ao meio e tentando novamente."
        )
        return gerar_embeddings_chunk(metade1) + gerar_embeddings_chunk(metade2)


def ingerir_pdfs():
    pdfs = list(DOCUMENTOS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em '{DOCUMENTOS_DIR}/'.")
        sys.exit(1)

    cliente = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recria a coleção para evitar duplicatas em reingestões
    try:
        cliente.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    colecao = cliente.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf in pdfs:
        print(f"\nProcessando: {pdf.name}")
        texto = extrair_texto_pdf(pdf)
        if not texto.strip():
            print(f"  Aviso: nenhum texto extraído de {pdf.name}.")
            continue

        chunks = dividir_em_chunks(texto, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {len(chunks)} chunks gerados.")

        documentos, embeddings, ids, metadados = [], [], [], []
        contador_id = 0

        for i, chunk in enumerate(chunks):
            print(f"  Embedding {i + 1}/{len(chunks)}...", end="\r")
            pares = gerar_embeddings_chunk(chunk)
            for texto_final, emb in pares:
                chunk_id = f"{pdf.stem}_chunk_{contador_id}"
                documentos.append(texto_final)
                embeddings.append(emb)
                ids.append(chunk_id)
                metadados.append({"arquivo": pdf.name, "trecho": contador_id})
                contador_id += 1

        colecao.add(
            documents=documentos,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadados,
        )
        total_chunks += len(documentos)
        print(f"  {len(documentos)} chunks salvos no ChromaDB.      ")

    print(f"\nIngestão concluída. Total de chunks: {total_chunks}")
    print(f"Base vetorial salva em '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    ingerir_pdfs()