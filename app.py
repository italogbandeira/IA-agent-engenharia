"""
Interface web (Streamlit) para o Agente de Normas Técnicas.
Uso: streamlit run app.py
"""

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import chromadb
import edge_tts
import ollama
import streamlit as st
import whisper

from agente import (
    LIMIAR_SIMILARIDADE,
    _REGEX_CAMINHO_PDF,
    _gerar_resposta_conversacional_stream,
    classificar_intencao,
    gerar_resposta_com_contexto_stream,
    reescrever_pergunta,
)
from ferramentas import EMBED_MODEL, Ferramentas
from ingestao import dividir_em_chunks, ingerir_pdfs

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "normas"
PASTA_DOCUMENTOS = Path("documentos")
MENSAGEM_ANALISE = "analise os documentos anexados"
ARQUIVO_FEEDBACK = Path("feedback.jsonl")
VOZ_TTS = "pt-BR-FranciscaNeural"
CAMINHO_AUDIO_RESPOSTA = "temp_audio_resposta.mp3"
CHUNK_SIZE_UPLOAD = 300
CHUNK_OVERLAP_UPLOAD = 100
ARQUIVO_HISTORICO = Path("historico_conversa.json")
HISTORICO_MAX_MENSAGENS = 50


def carregar_historico() -> list[dict]:
    try:
        if ARQUIVO_HISTORICO.exists():
            with open(ARQUIVO_HISTORICO, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def salvar_historico(historico: list[dict]):
    try:
        with open(ARQUIVO_HISTORICO, "w", encoding="utf-8") as f:
            json.dump(historico[-HISTORICO_MAX_MENSAGENS:], f, indent=2, ensure_ascii=False)
    except Exception:
        pass


@st.cache_resource
def carregar_ferramentas() -> Ferramentas:
    cliente = chromadb.PersistentClient(path=CHROMA_DIR)
    colecao = cliente.get_collection(COLLECTION_NAME)
    return Ferramentas(colecao)


@st.cache_resource
def carregar_whisper():
    return whisper.load_model("small")


_CORRECOES_TRANSCRICAO = {
    "IRP": "ART",
    "ERP": "ART",
    "ARP": "ART",
    "arte ": "ART ",
    "artes ": "ARTs ",
    "CRIAR": "CREA",
    " cria ": " CREA ",
    "COFEA": "CONFEA",
    "CONFIA": "CONFEA",
    "COMFEA": "CONFEA",
    "NBR ": "NBR ",
}


def corrigir_transcricao(texto: str) -> str:
    for errado, correto in _CORRECOES_TRANSCRICAO.items():
        texto = texto.replace(errado, correto)
    return texto


def transcrever_audio(caminho_audio: str) -> str:
    modelo = carregar_whisper()
    resultado = modelo.transcribe(
        caminho_audio,
        language="pt",
        initial_prompt=(
            "Conversa técnica sobre engenharia civil e normas brasileiras. "
            "Termos comuns: ART, ARTs, CREA, CONFEA, NBR, NBR 5674, laudo, "
            "vistoria predial, manutenção preventiva, manutenção corretiva, "
            "impermeabilização, infiltração, responsabilidade técnica, "
            "Anotação de Responsabilidade Técnica, acervo técnico, "
            "fiscalização, obra, serviço, profissional habilitado."
        ),
    )
    texto = resultado["text"].strip()
    texto = corrigir_transcricao(texto)
    return texto


async def _texto_para_audio_async(texto: str, caminho_saida: str):
    communicate = edge_tts.Communicate(texto, voice=VOZ_TTS)
    await communicate.save(caminho_saida)


def texto_para_audio(texto: str, caminho_saida: str):
    asyncio.run(_texto_para_audio_async(texto, caminho_saida))


st.set_page_config(page_title="Agente de Normas Técnicas")
st.title("Agente de Normas Técnicas")
st.caption("Consulte normas, analise laudos e tire dúvidas técnicas")

try:
    ferramentas = carregar_ferramentas()
except Exception:
    st.error(
        "Erro: base vetorial não encontrada. Execute 'python ingestao.py' primeiro."
    )
    st.stop()

if "historico" not in st.session_state:
    st.session_state.historico = carregar_historico()
if "arquivos_salvos" not in st.session_state:
    st.session_state.arquivos_salvos = []
if "disparar_analise" not in st.session_state:
    st.session_state.disparar_analise = False
if "feedback_enviado" not in st.session_state:
    st.session_state.feedback_enviado = False
if "tarefa_por_voz" not in st.session_state:
    st.session_state.tarefa_por_voz = None
if "ultimo_audio_hash" not in st.session_state:
    st.session_state.ultimo_audio_hash = None
if "processando" not in st.session_state:
    st.session_state.processando = False
if "mostrar_opcao_indexar" not in st.session_state:
    st.session_state.mostrar_opcao_indexar = False

with st.sidebar:
    if st.button("🗑️ Limpar conversa"):
        st.session_state.historico = []
        ARQUIVO_HISTORICO.unlink(missing_ok=True)
        st.rerun()

    st.divider()
    st.subheader("📄 Upload de documentos")
    uploads = st.file_uploader(
        "Envie um ou mais PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploads:
        st.write("**Arquivos anexados:**")
        for arq in uploads:
            tamanho_kb = len(arq.getbuffer()) / 1024
            st.write(f"- {arq.name} ({tamanho_kb:.1f} KB)")

        if st.button("Analisar documentos"):
            PASTA_DOCUMENTOS.mkdir(exist_ok=True)
            caminhos = []
            for arq in uploads:
                caminho = PASTA_DOCUMENTOS / f"temp_{arq.name}"
                with open(caminho, "wb") as f:
                    f.write(arq.getbuffer())
                caminhos.append(caminho)
            st.session_state.arquivos_salvos = caminhos
            st.session_state.disparar_analise = True
            st.rerun()

    if st.session_state.arquivos_salvos:
        if st.button("Limpar anexos"):
            for caminho in st.session_state.arquivos_salvos:
                Path(caminho).unlink(missing_ok=True)
            st.session_state.arquivos_salvos = []
            st.rerun()

    st.divider()
    st.subheader("🗄️ Base de conhecimento")

    total_chunks = ferramentas.colecao.count()
    st.caption(f"📚 {total_chunks} chunks indexados")

    if st.button("🔄 Reindexar normas"):
        try:
            with st.spinner("Reindexando documentos da pasta /documentos..."):
                ingerir_pdfs()
            st.cache_resource.clear()
            st.success("✅ Base atualizada com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao reindexar: {e}")

    with st.expander("📋 Documentos indexados"):
        metadados_existentes = ferramentas.colecao.get(include=["metadatas"])["metadatas"]
        arquivos_unicos = sorted({meta["arquivo"] for meta in metadados_existentes})
        if arquivos_unicos:
            for arquivo in arquivos_unicos:
                st.markdown(f"• {arquivo}")
        else:
            st.write("Nenhum documento indexado.")

    st.divider()
    st.subheader("🎤 Entrada por voz")

    if hasattr(st, "audio_input"):
        audio_valor = st.audio_input("Grave sua pergunta")
    else:
        audio_valor = st.file_uploader(
            "Envie um áudio (fallback)", type=["wav", "mp3", "ogg"], key="audio_fallback"
        )

    if audio_valor is not None:
        audio_bytes = audio_valor.getvalue()
        hash_audio = hashlib.sha256(audio_bytes).hexdigest()

        if hash_audio != st.session_state.ultimo_audio_hash:
            st.session_state.ultimo_audio_hash = hash_audio
            nome_original = getattr(audio_valor, "name", "audio.wav")
            sufixo = Path(nome_original).suffix or ".wav"

            tmp = tempfile.NamedTemporaryFile(suffix=sufixo, delete=False)
            tmp.write(audio_bytes)
            tmp.close()  # Fecha antes de passar o caminho pro Whisper
            caminho_tmp = tmp.name

            try:
                texto_transcrito = transcrever_audio(caminho_tmp)
                st.info(f"Transcrição: {texto_transcrito}")
                if texto_transcrito:
                    st.session_state.tarefa_por_voz = texto_transcrito
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao transcrever o áudio: {e}")
            finally:
                os.unlink(caminho_tmp)  # Deleta manualmente após o uso

    st.divider()
    st.subheader("🔊 Configurações de áudio")
    ler_respostas_em_voz = st.toggle("🔊 Ler respostas em voz alta", value=True)


def processar_pergunta(tarefa: str, ferramentas: Ferramentas, historico: list[dict]):
    intencao = classificar_intencao(tarefa)

    if intencao in ("conversacional", "geral"):
        return _gerar_resposta_conversacional_stream(tarefa, historico), []

    if intencao == "ler_documento":
        match = _REGEX_CAMINHO_PDF.search(tarefa)
        caminho = match.group(0) if match else ""
        resultado = ferramentas.ler_documento(caminho)
        if "erro" in resultado:
            mensagem = f"Não foi possível ler o documento: {resultado['erro']}"
            return (chunk for chunk in [mensagem]), []
        fontes = [{"arquivo": resultado["arquivo"], "trecho": "documento completo"}]
        return gerar_resposta_com_contexto_stream(tarefa, resultado["texto"], historico), fontes

    # buscar_norma
    pergunta_rag = reescrever_pergunta(tarefa, historico)
    resultado = ferramentas.buscar_normas(pergunta_rag)
    if resultado["melhor_similaridade"] < LIMIAR_SIMILARIDADE:
        mensagem = "Não encontrei essa informação nos documentos disponíveis."
        return (chunk for chunk in [mensagem]), []
    return (
        gerar_resposta_com_contexto_stream(tarefa, resultado["contexto"], historico),
        resultado["fontes"],
    )


def analisar_documentos_anexados(
    tarefa: str, ferramentas: Ferramentas, historico: list[dict], caminhos: list[Path]
):
    blocos = []
    fontes = []
    for caminho in caminhos:
        resultado = ferramentas.ler_documento(str(caminho))
        if "erro" in resultado:
            blocos.append(f"[Erro ao ler {caminho.name}: {resultado['erro']}]")
        else:
            blocos.append(f"[Documento: {resultado['arquivo']}]\n{resultado['texto']}")
            fontes.append({"arquivo": resultado["arquivo"], "trecho": "documento completo"})
    contexto = "\n\n---\n\n".join(blocos)
    return gerar_resposta_com_contexto_stream(tarefa, contexto, historico), fontes


def indexar_documentos_anexados(caminhos: list[Path], ferramentas: Ferramentas) -> bool:
    cliente = chromadb.PersistentClient(path=CHROMA_DIR)
    colecao = cliente.get_collection(COLLECTION_NAME)

    barra = st.progress(0.0, text="Indexando documentos...")
    algum_indexado = False

    try:
        for indice_arquivo, caminho in enumerate(caminhos):
            resultado = ferramentas.ler_documento(str(caminho))
            if "erro" in resultado:
                continue

            nome_arquivo = resultado["arquivo"]

            existentes = colecao.get(where={"arquivo": nome_arquivo})
            if existentes["ids"]:
                continue  # já indexado anteriormente — evita duplicatas

            chunks = dividir_em_chunks(
                resultado["texto"], CHUNK_SIZE_UPLOAD, CHUNK_OVERLAP_UPLOAD
            )

            documentos, embeddings, ids, metadados = [], [], [], []
            for i, chunk in enumerate(chunks):
                resposta_emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                documentos.append(chunk)
                embeddings.append(resposta_emb["embedding"])
                ids.append(f"upload_{Path(caminho).stem}_chunk_{i}")
                metadados.append({"arquivo": nome_arquivo, "trecho": i})

            if documentos:
                colecao.add(
                    documents=documentos,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadados,
                )
                algum_indexado = True

            barra.progress((indice_arquivo + 1) / len(caminhos))
    finally:
        barra.empty()

    return algum_indexado


def exibir_fontes(fontes: list[dict]):
    if not fontes:
        return
    with st.expander("📎 Fontes utilizadas"):
        for fonte in fontes:
            linha = f"• {fonte['arquivo']} — trecho {fonte['trecho']}"
            if "similaridade" in fonte:
                linha += f" (similaridade: {fonte['similaridade']})"
            st.markdown(linha)


def salvar_feedback(pergunta: str, resposta: str, fontes: list[dict], avaliacao: str):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "pergunta": pergunta,
        "resposta": resposta,
        "fontes": [fonte["arquivo"] for fonte in fontes],
        "avaliacao": avaliacao,
    }
    with open(ARQUIVO_FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def exibir_feedback(indice: int, pergunta: str, resposta: str, fontes: list[dict]):
    if st.session_state.feedback_enviado or st.session_state.processando:
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍 Útil", key=f"feedback_util_{indice}"):
            salvar_feedback(pergunta, resposta, fontes, "positiva")
            st.session_state.feedback_enviado = True
            st.toast("Obrigado pelo feedback!")
            st.rerun()
    with col2:
        if st.button("👎 Não útil", key=f"feedback_nao_util_{indice}"):
            salvar_feedback(pergunta, resposta, fontes, "negativa")
            st.session_state.feedback_enviado = True
            st.toast("Anotado. Vamos melhorar!")
            st.rerun()


for indice, mensagem in enumerate(st.session_state.historico):
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])
        if mensagem["role"] == "assistant":
            exibir_fontes(mensagem.get("fontes", []))
            if indice == len(st.session_state.historico) - 1:
                pergunta_anterior = (
                    st.session_state.historico[indice - 1]["content"] if indice > 0 else ""
                )
                exibir_feedback(
                    indice, pergunta_anterior, mensagem["content"], mensagem.get("fontes", [])
                )

tarefa = None
if st.session_state.disparar_analise:
    tarefa = MENSAGEM_ANALISE
    st.session_state.disparar_analise = False

if st.session_state.tarefa_por_voz:
    tarefa = st.session_state.tarefa_por_voz
    st.session_state.tarefa_por_voz = None

tarefa = st.chat_input("Digite sua pergunta...") or tarefa

if tarefa:
    st.session_state.processando = True
    st.session_state.feedback_enviado = False

    with st.chat_message("user"):
        st.markdown(tarefa)

    with st.chat_message("assistant"):
        if tarefa == MENSAGEM_ANALISE and st.session_state.arquivos_salvos:
            stream_generator, fontes = analisar_documentos_anexados(
                tarefa,
                ferramentas,
                st.session_state.historico,
                st.session_state.arquivos_salvos,
            )
        else:
            stream_generator, fontes = processar_pergunta(
                tarefa, ferramentas, st.session_state.historico
            )
        resposta = st.write_stream(stream_generator)
        if not isinstance(resposta, str):
            resposta = "".join(resposta)
        exibir_fontes(fontes)

        if tarefa == MENSAGEM_ANALISE and st.session_state.arquivos_salvos:
            st.session_state.mostrar_opcao_indexar = True

        if ler_respostas_em_voz:
            try:
                texto_para_audio(resposta, CAMINHO_AUDIO_RESPOSTA)
                with open(CAMINHO_AUDIO_RESPOSTA, "rb") as f:
                    st.audio(f.read(), format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"Erro ao gerar áudio da resposta: {e}")

        st.session_state.historico.append({"role": "user", "content": tarefa})
        st.session_state.historico.append(
            {"role": "assistant", "content": resposta, "fontes": fontes}
        )
        salvar_historico(st.session_state.historico)
        st.session_state.processando = False
        indice_nova_resposta = len(st.session_state.historico) - 1
        exibir_feedback(indice_nova_resposta, tarefa, resposta, fontes)

if st.session_state.mostrar_opcao_indexar and st.session_state.arquivos_salvos:
    st.info(
        "💾 Deseja adicionar esse documento à base de conhecimento para consultas futuras?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Sim, indexar"):
            algum_indexado = indexar_documentos_anexados(
                st.session_state.arquivos_salvos, ferramentas
            )
            st.session_state.mostrar_opcao_indexar = False
            if algum_indexado:
                st.cache_resource.clear()
                st.success(
                    "✅ Documento indexado com sucesso! Agora faz parte da base de conhecimento."
                )
            else:
                st.info("Esse documento já estava indexado na base de conhecimento.")
            st.rerun()
    with col2:
        if st.button("❌ Não"):
            st.session_state.mostrar_opcao_indexar = False
            st.toast("Documento não indexado.")
            st.rerun()
