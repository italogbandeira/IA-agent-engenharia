"""
Interface web (Streamlit) para o Agente de Normas Técnicas.
Uso: streamlit run app.py
"""

import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import edge_tts
import ollama
import streamlit as st
import whisper

from agente import (
    LIMIAR_SIMILARIDADE,
    SYSTEM_PROMPT,
    _REGEX_CAMINHO_PDF,
    _gerar_resposta_conversacional_stream,
    classificar_intencao,
    comparar_laudos_stream,
    gerar_apontamentos,
    gerar_resposta_com_contexto_stream,
    reescrever_pergunta,
)
from auth import (
    atualizar_perfil,
    cadastrar_usuario,
    gerar_contexto_perfil,
    get_nome,
    verificar_login,
)
from ferramentas import EMBED_MODEL, Ferramentas
from ingestao import dividir_em_chunks, ingerir_pdfs
from relatorio import gerar_relatorio_docx

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "normas"
PASTA_DOCUMENTOS = Path("documentos")
MENSAGEM_ANALISE = "analise os documentos anexados"
VOZ_TTS = "pt-BR-FranciscaNeural"
CAMINHO_AUDIO_RESPOSTA = "temp_audio_resposta.mp3"
CHUNK_SIZE_UPLOAD = 300
CHUNK_OVERLAP_UPLOAD = 100
HISTORICO_MAX_MENSAGENS = 50


def arquivo_historico(username: str) -> Path:
    return Path(f"historico_{username}.json")


def arquivo_feedback(username: str) -> Path:
    return Path(f"feedback_{username}.jsonl")


def carregar_historico(username: str) -> list[dict]:
    try:
        caminho = arquivo_historico(username)
        if caminho.exists():
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def salvar_historico(username: str, historico: list[dict]):
    try:
        with open(arquivo_historico(username), "w", encoding="utf-8") as f:
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

if "usuario_logado" not in st.session_state:
    st.session_state.usuario_logado = None

if not st.session_state.usuario_logado:
    st.title("🔐 Agente de Normas Técnicas")
    st.subheader("Faça login para continuar")

    with st.form("form_login"):
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")

        if submitted:
            if verificar_login(usuario, senha):
                st.session_state.usuario_logado = usuario
                atualizar_perfil(usuario, {"nova_sessao": True})
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos.")
    st.stop()

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
    st.session_state.historico = carregar_historico(st.session_state.usuario_logado)
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
if "mostrar_opcao_relatorio" not in st.session_state:
    st.session_state.mostrar_opcao_relatorio = False
if "contexto_ultimo_laudo" not in st.session_state:
    st.session_state.contexto_ultimo_laudo = ""
if "arquivos_ultimo_laudo" not in st.session_state:
    st.session_state.arquivos_ultimo_laudo = []
if "relatorio_bytes" not in st.session_state:
    st.session_state.relatorio_bytes = None
if "ultima_comparacao" not in st.session_state:
    st.session_state.ultima_comparacao = None

with st.sidebar:
    nome = get_nome(st.session_state.usuario_logado)
    st.caption(f"👤 {nome}")
    if st.button("Sair", key="logout"):
        st.session_state.usuario_logado = None
        st.session_state.historico = []
        st.rerun()
    st.divider()

    if st.button("🗑️ Limpar conversa"):
        st.session_state.historico = []
        arquivo_historico(st.session_state.usuario_logado).unlink(missing_ok=True)
        st.session_state.mostrar_opcao_relatorio = False
        st.session_state.contexto_ultimo_laudo = ""
        st.session_state.arquivos_ultimo_laudo = []
        st.session_state.relatorio_bytes = None
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


def processar_pergunta(
    tarefa: str, ferramentas: Ferramentas, historico: list[dict], system_prompt: str = SYSTEM_PROMPT
):
    intencao = classificar_intencao(tarefa)

    if intencao in ("conversacional", "geral"):
        return _gerar_resposta_conversacional_stream(tarefa, historico, system_prompt), []

    if intencao == "ler_documento":
        match = _REGEX_CAMINHO_PDF.search(tarefa)
        caminho = match.group(0) if match else ""
        resultado = ferramentas.ler_documento(caminho)
        if "erro" in resultado:
            mensagem = f"Não foi possível ler o documento: {resultado['erro']}"
            return (chunk for chunk in [mensagem]), []
        fontes = [{"arquivo": resultado["arquivo"], "trecho": "documento completo"}]
        return (
            gerar_resposta_com_contexto_stream(tarefa, resultado["texto"], historico, system_prompt),
            fontes,
        )

    # buscar_norma
    pergunta_rag = reescrever_pergunta(tarefa, historico)
    resultado = ferramentas.buscar_normas(pergunta_rag)
    if resultado["melhor_similaridade"] < LIMIAR_SIMILARIDADE:
        mensagem = "Não encontrei essa informação nos documentos disponíveis."
        return (chunk for chunk in [mensagem]), []
    return (
        gerar_resposta_com_contexto_stream(tarefa, resultado["contexto"], historico, system_prompt),
        resultado["fontes"],
    )


def analisar_documentos_anexados(
    tarefa: str,
    ferramentas: Ferramentas,
    historico: list[dict],
    caminhos: list[Path],
    system_prompt: str = SYSTEM_PROMPT,
):
    blocos = []
    fontes = []
    nomes_arquivos = []
    for caminho in caminhos:
        resultado = ferramentas.ler_documento(str(caminho))
        if "erro" in resultado:
            blocos.append(f"[Erro ao ler {caminho.name}: {resultado['erro']}]")
        else:
            blocos.append(f"[Documento: {resultado['arquivo']}]\n{resultado['texto']}")
            fontes.append({"arquivo": resultado["arquivo"], "trecho": "documento completo"})
            nomes_arquivos.append(resultado["arquivo"])
    contexto = "\n\n---\n\n".join(blocos)
    stream = gerar_resposta_com_contexto_stream(tarefa, contexto, historico, system_prompt)
    return stream, fontes, contexto, nomes_arquivos


def _salvar_temp_comparacao(arquivo_anterior, arquivo_atual) -> tuple[Path, Path]:
    PASTA_DOCUMENTOS.mkdir(exist_ok=True)
    sufixo = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_anterior = PASTA_DOCUMENTOS / f"comparacao_{sufixo}_anterior_{arquivo_anterior.name}"
    caminho_atual = PASTA_DOCUMENTOS / f"comparacao_{sufixo}_atual_{arquivo_atual.name}"
    with open(caminho_anterior, "wb") as f:
        f.write(arquivo_anterior.getbuffer())
    with open(caminho_atual, "wb") as f:
        f.write(arquivo_atual.getbuffer())
    return caminho_anterior, caminho_atual


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


def calcular_tamanho_pasta_mb(pasta: str) -> float:
    total_bytes = 0
    for raiz, _, arquivos in os.walk(pasta):
        for nome in arquivos:
            caminho = os.path.join(raiz, nome)
            total_bytes += os.path.getsize(caminho)
    return total_bytes / (1024 * 1024)


def contar_feedbacks(username: str) -> int:
    caminho = arquivo_feedback(username)
    if not caminho.exists():
        return 0
    with open(caminho, "r", encoding="utf-8") as f:
        return sum(1 for linha in f if linha.strip())


def carregar_feedbacks(username: str) -> list[dict]:
    caminho = arquivo_feedback(username)
    if not caminho.exists():
        return []
    registros = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return registros


def exibir_fontes(fontes: list[dict]):
    if not fontes:
        return
    with st.expander("📎 Fontes utilizadas"):
        for fonte in fontes:
            linha = f"• {fonte['arquivo']} — trecho {fonte['trecho']}"
            if "similaridade" in fonte:
                linha += f" (similaridade: {fonte['similaridade']})"
            st.markdown(linha)


def salvar_feedback(username: str, pergunta: str, resposta: str, fontes: list[dict], avaliacao: str):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "pergunta": pergunta,
        "resposta": resposta,
        "fontes": [fonte["arquivo"] for fonte in fontes],
        "avaliacao": avaliacao,
    }
    with open(arquivo_feedback(username), "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def exibir_feedback(indice: int, pergunta: str, resposta: str, fontes: list[dict]):
    if st.session_state.feedback_enviado or st.session_state.processando:
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍 Útil", key=f"feedback_util_{indice}"):
            salvar_feedback(st.session_state.usuario_logado, pergunta, resposta, fontes, "positiva")
            atualizar_perfil(st.session_state.usuario_logado, {"feedback": "positivo"})
            st.session_state.feedback_enviado = True
            st.toast("Obrigado pelo feedback!")
            st.rerun()
    with col2:
        if st.button("👎 Não útil", key=f"feedback_nao_util_{indice}"):
            salvar_feedback(st.session_state.usuario_logado, pergunta, resposta, fontes, "negativa")
            atualizar_perfil(st.session_state.usuario_logado, {"feedback": "negativo"})
            st.session_state.feedback_enviado = True
            st.toast("Anotado. Vamos melhorar!")
            st.rerun()


tab_chat, tab_admin, tab_comparar = st.tabs(["💬 Chat", "⚙️ Administração", "🔍 Comparar Laudos"])

with tab_chat:
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
            contexto_perfil = gerar_contexto_perfil(st.session_state.usuario_logado)
            system_prompt_personalizado = SYSTEM_PROMPT + (
                f"\n\nContexto do usuário: {contexto_perfil}" if contexto_perfil else ""
            )

            if tarefa == MENSAGEM_ANALISE and st.session_state.arquivos_salvos:
                stream_generator, fontes, contexto_laudo, nomes_laudo = analisar_documentos_anexados(
                    tarefa,
                    ferramentas,
                    st.session_state.historico,
                    st.session_state.arquivos_salvos,
                    system_prompt_personalizado,
                )
            else:
                stream_generator, fontes = processar_pergunta(
                    tarefa, ferramentas, st.session_state.historico, system_prompt_personalizado
                )
            resposta = st.write_stream(stream_generator)
            if not isinstance(resposta, str):
                resposta = "".join(resposta)
            exibir_fontes(fontes)

            atualizar_perfil(st.session_state.usuario_logado, {"tema": classificar_intencao(tarefa)})

            if tarefa == MENSAGEM_ANALISE and st.session_state.arquivos_salvos:
                st.session_state.mostrar_opcao_indexar = True
                st.session_state.mostrar_opcao_relatorio = True
                st.session_state.contexto_ultimo_laudo = contexto_laudo
                st.session_state.arquivos_ultimo_laudo = nomes_laudo
                st.session_state.relatorio_bytes = None
                for nome_doc in nomes_laudo:
                    atualizar_perfil(
                        st.session_state.usuario_logado, {"documento_analisado": nome_doc}
                    )

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
            salvar_historico(st.session_state.usuario_logado, st.session_state.historico)
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

    if st.session_state.mostrar_opcao_relatorio and st.session_state.contexto_ultimo_laudo:
        st.divider()
        st.subheader("📝 Relatório de apontamentos")

        if st.button("📝 Gerar relatório de apontamentos (.docx)"):
            with st.spinner("Extraindo apontamentos e montando o relatório..."):
                try:
                    dados_apontamentos = gerar_apontamentos(st.session_state.contexto_ultimo_laudo)
                    st.session_state.relatorio_bytes = gerar_relatorio_docx(
                        st.session_state.arquivos_ultimo_laudo, dados_apontamentos
                    )
                    qtd_apontamentos = len(dados_apontamentos.get("apontamentos", []))
                    st.success(f"✅ Relatório gerado com {qtd_apontamentos} apontamento(s).")
                except Exception as e:
                    st.session_state.relatorio_bytes = None
                    st.error(f"Erro ao gerar relatório: {e}")

        if st.session_state.relatorio_bytes:
            nome_relatorio = f"relatorio_apontamentos_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"
            st.download_button(
                "⬇️ Baixar relatório",
                data=st.session_state.relatorio_bytes,
                file_name=nome_relatorio,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

with tab_admin:
    st.subheader("📚 Documentos na base")

    metadados_admin = ferramentas.colecao.get(include=["metadatas"])["metadatas"]
    contagem_por_arquivo = Counter(meta["arquivo"] for meta in metadados_admin)
    total_chunks_admin = sum(contagem_por_arquivo.values())

    if not contagem_por_arquivo:
        st.write("Nenhum documento indexado.")
    else:
        for arquivo, qtd_chunks in sorted(contagem_por_arquivo.items()):
            percentual = (qtd_chunks / total_chunks_admin * 100) if total_chunks_admin else 0
            col_nome, col_chunks, col_pct, col_acao = st.columns([3, 1, 1, 1])
            col_nome.write(arquivo)
            col_chunks.write(str(qtd_chunks))
            col_pct.write(f"{percentual:.1f}%")
            if col_acao.button("🗑️ Remover", key=f"remover_{arquivo}"):
                ferramentas.colecao.delete(where={"arquivo": arquivo})
                st.cache_resource.clear()
                st.success(f"{arquivo} removido da base.")
                st.rerun()

    st.divider()
    st.subheader("📊 Estatísticas")

    tamanho_mb = calcular_tamanho_pasta_mb(CHROMA_DIR)
    total_feedbacks = contar_feedbacks(st.session_state.usuario_logado)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de chunks", total_chunks_admin)
    col2.metric("Documentos únicos", len(contagem_por_arquivo))
    col3.metric("Tamanho da base", f"{tamanho_mb:.2f} MB")
    col4.metric("Feedbacks coletados", total_feedbacks)

    st.divider()
    st.subheader("📝 Feedbacks recentes")

    feedbacks = carregar_feedbacks(st.session_state.usuario_logado)
    if not feedbacks:
        st.write("Nenhum feedback coletado ainda.")
    else:
        for registro in feedbacks[-10:][::-1]:
            emoji_avaliacao = "👍" if registro.get("avaliacao") == "positiva" else "👎"
            resposta_truncada = registro.get("resposta", "")[:100]
            with st.container(border=True):
                st.markdown(f"**Pergunta:** {registro.get('pergunta', '')}")
                st.markdown(f"**Resposta:** {resposta_truncada}...")
                st.markdown(
                    f"{emoji_avaliacao} {registro.get('avaliacao', '')} — "
                    f"{registro.get('timestamp', '')}"
                )

        caminho_feedback_atual = arquivo_feedback(st.session_state.usuario_logado)
        if caminho_feedback_atual.exists():
            with open(caminho_feedback_atual, "rb") as f:
                st.download_button(
                    "⬇️ Exportar feedbacks",
                    data=f.read(),
                    file_name=caminho_feedback_atual.name,
                    mime="application/jsonl",
                )

    st.divider()
    st.subheader("👥 Usuários")
    with st.form("form_novo_usuario"):
        novo_usuario = st.text_input("Nome de usuário")
        novo_nome = st.text_input("Nome completo")
        nova_senha = st.text_input("Senha", type="password")
        if st.form_submit_button("Cadastrar"):
            if cadastrar_usuario(novo_usuario, nova_senha, novo_nome):
                st.success(f"Usuário {novo_nome} cadastrado com sucesso!")
            else:
                st.error("Usuário já existe.")

with tab_comparar:
    st.subheader("🔍 Comparação de laudos")
    st.caption(
        "Envie dois laudos do mesmo imóvel ou obra para identificar novos apontamentos, "
        "apontamentos resolvidos, evolução de patologias e mudanças nas recomendações."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        laudo_anterior = st.file_uploader(
            "Laudo anterior (mais antigo)", type=["pdf"], key="upload_laudo_anterior"
        )
    with col_b:
        laudo_atual = st.file_uploader(
            "Laudo atual (mais recente)", type=["pdf"], key="upload_laudo_atual"
        )

    st.caption(
        "⏱️ A comparação processa os dois laudos com uma janela de contexto ampla e pode "
        "levar vários minutos (o processamento roda localmente, sem GPU)."
    )

    if st.button("🔍 Comparar laudos", disabled=not (laudo_anterior and laudo_atual)):
        caminho_anterior, caminho_atual = _salvar_temp_comparacao(laudo_anterior, laudo_atual)
        try:
            resultado_1 = ferramentas.ler_documento(str(caminho_anterior))
            resultado_2 = ferramentas.ler_documento(str(caminho_atual))
            if "erro" in resultado_1 or "erro" in resultado_2:
                st.error("Não foi possível ler um dos laudos enviados.")
            else:
                st.markdown(f"**Comparando:** {resultado_1['arquivo']} → {resultado_2['arquivo']}")
                texto_resultado = st.write_stream(
                    comparar_laudos_stream(
                        resultado_1["texto"],
                        resultado_2["texto"],
                        resultado_1["arquivo"],
                        resultado_2["arquivo"],
                    )
                )
                st.session_state.ultima_comparacao = {
                    "nome_anterior": resultado_1["arquivo"],
                    "nome_atual": resultado_2["arquivo"],
                    "texto": texto_resultado,
                }
        finally:
            caminho_anterior.unlink(missing_ok=True)
            caminho_atual.unlink(missing_ok=True)
    elif st.session_state.ultima_comparacao:
        c = st.session_state.ultima_comparacao
        st.markdown(f"**Última comparação:** {c['nome_anterior']} → {c['nome_atual']}")
        st.markdown(c["texto"])