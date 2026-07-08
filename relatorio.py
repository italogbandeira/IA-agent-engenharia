"""
Geração de relatório de apontamentos técnicos (.docx) a partir da
análise de um ou mais laudos.
"""

from datetime import datetime
from io import BytesIO

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import RGBColor

_COR_GRAVIDADE = {
    "alta": RGBColor(0xC0, 0x00, 0x00),
    "média": RGBColor(0xB8, 0x86, 0x00),
    "media": RGBColor(0xB8, 0x86, 0x00),
    "baixa": RGBColor(0x2E, 0x7D, 0x32),
}


def gerar_relatorio_docx(nomes_arquivos: list[str], dados: dict) -> bytes:
    """Monta um relatório de apontamentos em .docx e retorna os bytes do arquivo."""
    documento = Document()

    titulo = documento.add_heading("Relatório de Apontamentos Técnicos", level=0)
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER

    documento.add_paragraph(
        f"Documento(s) analisado(s): {', '.join(nomes_arquivos) if nomes_arquivos else 'não identificado'}"
    )
    documento.add_paragraph(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    nota = documento.add_paragraph()
    nota_run = nota.add_run(
        "Gerado automaticamente pelo Agente de Normas Técnicas (SEDUH). "
        "Revisão de um profissional habilitado é obrigatória antes de assinatura e arquivamento."
    )
    nota_run.italic = True

    documento.add_heading("Resumo", level=1)
    documento.add_paragraph(dados.get("resumo") or "Sem resumo disponível.")

    documento.add_heading("Apontamentos", level=1)
    apontamentos = dados.get("apontamentos", [])

    if not apontamentos:
        documento.add_paragraph("Nenhum apontamento identificado no texto analisado.")
    else:
        for indice, item in enumerate(apontamentos, start=1):
            documento.add_heading(f"{indice}. {item.get('problema', 'Apontamento')}", level=2)

            gravidade = (item.get("gravidade") or "não classificada").strip().lower()
            p_gravidade = documento.add_paragraph()
            p_gravidade.add_run("Gravidade: ").bold = True
            run_valor = p_gravidade.add_run(gravidade.capitalize())
            run_valor.bold = True
            cor = _COR_GRAVIDADE.get(gravidade)
            if cor:
                run_valor.font.color.rgb = cor

            p_norma = documento.add_paragraph()
            p_norma.add_run("Norma violada: ").bold = True
            p_norma.add_run(item.get("norma_violada") or "Não identificada")

            p_rec = documento.add_paragraph()
            p_rec.add_run("Recomendação: ").bold = True
            p_rec.add_run(item.get("recomendacao") or "—")

    documento.add_paragraph()
    rodape = documento.add_paragraph()
    rodape_run = rodape.add_run("Assinatura do responsável técnico: _________________________________")
    rodape_run.italic = True

    buffer = BytesIO()
    documento.save(buffer)
    return buffer.getvalue()
