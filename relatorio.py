"""
Geração de relatório Word (.docx) de análise de conformidade a partir
da resposta do agente sobre um laudo analisado.
"""

import re
from datetime import datetime

import ollama
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from agente import GEN_MODEL
from ferramentas import Ferramentas

LIMIAR_SIMILARIDADE_APONTAMENTO = 0.60
MAX_PROBLEMAS = 8


def gerar_relatorio_docx(
    nome_documento: str,
    resumo_laudo: str,
    apontamentos: list[dict],
    usuario: str,
    caminho_saida: str
) -> str:
    """
    Gera relatório Word com:
    - Cabeçalho institucional (SEDUH)
    - Resumo do laudo analisado
    - Tabela de apontamentos de conformidade
    - Rodapé com data e responsável

    apontamentos é uma lista de dicts:
    [{"item": "ART", "status": "Conforme", "norma": "Res. 1137", "observacao": "..."}]

    Retorna o caminho do arquivo gerado.
    """
    doc = Document()

    # Margens
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    # Cabeçalho
    cabecalho = doc.add_heading("SEDUH — Secretaria de Desenvolvimento Urbano e Habitação", 0)
    cabecalho.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitulo = doc.add_paragraph("Relatório de Análise de Conformidade Técnica")
    subtitulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitulo.runs[0].bold = True
    subtitulo.runs[0].font.size = Pt(13)

    doc.add_paragraph()

    # Metadados
    meta = doc.add_paragraph()
    meta.add_run("Documento analisado: ").bold = True
    meta.add_run(nome_documento)

    meta2 = doc.add_paragraph()
    meta2.add_run("Data da análise: ").bold = True
    meta2.add_run(datetime.now().strftime("%d/%m/%Y %H:%M"))

    meta3 = doc.add_paragraph()
    meta3.add_run("Responsável: ").bold = True
    meta3.add_run(usuario)

    doc.add_paragraph()

    # Resumo do laudo
    doc.add_heading("1. Resumo do Laudo Analisado", level=1)
    doc.add_paragraph(resumo_laudo)

    doc.add_paragraph()

    # Apontamentos de conformidade
    doc.add_heading("2. Apontamentos de Conformidade com Normas", level=1)

    if apontamentos:
        tabela = doc.add_table(rows=1, cols=4)
        tabela.style = "Table Grid"

        # Cabeçalho da tabela
        headers = ["Item", "Status", "Norma de Referência", "Observação"]
        for i, header in enumerate(headers):
            cell = tabela.rows[0].cells[i]
            cell.text = header
            cell.paragraphs[0].runs[0].bold = True

        # Linhas de apontamento
        for ap in apontamentos:
            row = tabela.add_row()
            row.cells[0].text = ap.get("item", "")
            row.cells[1].text = ap.get("status", "")
            row.cells[2].text = ap.get("norma", "")
            row.cells[3].text = ap.get("observacao", "")

            # Colore status
            status = ap.get("status", "").lower()
            run = row.cells[1].paragraphs[0].runs
            if run:
                if "conforme" in status and "não" not in status:
                    run[0].font.color.rgb = RGBColor(0, 128, 0)   # verde
                elif "não conforme" in status:
                    run[0].font.color.rgb = RGBColor(200, 0, 0)   # vermelho
                else:
                    run[0].font.color.rgb = RGBColor(180, 120, 0) # amarelo
    else:
        doc.add_paragraph("Nenhum apontamento identificado.")

    doc.add_paragraph()

    # Rodapé
    doc.add_paragraph("_" * 60)
    rodape = doc.add_paragraph()
    rodape.add_run("Nota: ").bold = True
    rodape.add_run(
        "Este relatório foi gerado com auxílio de sistema de IA e deve ser "
        "revisado e assinado pelo profissional responsável (ART obrigatória)."
    )
    rodape.runs[1].font.size = Pt(9)

    doc.save(caminho_saida)
    return caminho_saida


def extrair_apontamentos_da_resposta(resposta: str) -> list[dict]:
    """
    Tenta extrair apontamentos estruturados do texto de resposta do agente.
    Retorna lista de dicts com item, status, norma, observacao.
    """
    apontamentos = []
    linhas = resposta.split("\n")

    for linha in linhas:
        linha = linha.strip().lstrip("*•-").strip()
        if not linha:
            continue

        # Detecta padrões de conformidade
        status = "A verificar"
        if any(t in linha.lower() for t in ["conforme", "atende", "adequado", "correto"]):
            if any(t in linha.lower() for t in ["não", "nao", "in"]):
                status = "Não conforme"
            else:
                status = "Conforme"
        elif any(t in linha.lower() for t in ["não", "nao", "ausente", "falta", "irregular"]):
            status = "Não conforme"
        elif any(t in linha.lower() for t in ["verificar", "analisar", "pendente"]):
            status = "Pendente"

        # Detecta norma mencionada
        norma = ""
        for padrao in [r"NBR[\s\d]+", r"Resolução[\s\w\.]+", r"Art\.?\s*\d+", r"CONFEA\s*\d+"]:
            match = re.search(padrao, linha, re.IGNORECASE)
            if match:
                norma = match.group(0).strip()
                break

        if len(linha) > 10:
            apontamentos.append({
                "item": linha[:60] + ("..." if len(linha) > 60 else ""),
                "status": status,
                "norma": norma,
                "observacao": linha[60:120] if len(linha) > 60 else ""
            })

    return apontamentos[:15]  # máximo 15 apontamentos


def extrair_problemas(resposta_analise: str) -> list[str]:
    """
    Pede ao modelo uma lista objetiva dos problemas técnicos citados na
    análise, um por linha, pra depois buscar a norma aplicável de cada um.
    """
    prompt = (
        "Da análise abaixo, extraia uma lista com os problemas técnicos encontrados. "
        "Retorne apenas os problemas, um por linha, sem numeração, sem explicação "
        f"adicional.\n\nAnálise:\n{resposta_analise}"
    )
    try:
        resposta = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        linhas = resposta["message"]["content"].strip().split("\n")
        problemas = [linha.strip().lstrip("*•-").strip() for linha in linhas if linha.strip()]
        return problemas[:MAX_PROBLEMAS]
    except Exception:
        return []


def _extrair_trecho_contexto(contexto: str) -> str:
    primeiro_bloco = contexto.split("\n\n---\n\n")[0]
    linhas = primeiro_bloco.split("\n", 1)
    if len(linhas) > 1 and linhas[0].startswith("[Fonte:"):
        return linhas[1].strip()
    return primeiro_bloco.strip()


def buscar_norma_para_problema(problema: str, ferramentas: Ferramentas) -> dict:
    """Busca na base vetorial a norma mais relevante para um problema identificado."""
    resultado = ferramentas.buscar_normas(problema)

    if resultado["melhor_similaridade"] >= LIMIAR_SIMILARIDADE_APONTAMENTO and resultado["fontes"]:
        fonte = resultado["fontes"][0]
        trecho = _extrair_trecho_contexto(resultado["contexto"])
        return {
            "item": problema[:60],
            "status": "A verificar",
            "norma": fonte["arquivo"],
            "observacao": trecho[:100],
        }

    return {
        "item": problema[:60],
        "status": "A verificar",
        "norma": "Sem norma indexada",
        "observacao": "",
    }
