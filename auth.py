"""
Login com perfil de usuário persistente: credenciais locais (hash bcrypt)
e um perfil por usuário usado para personalizar o system prompt do agente.
"""

import json
from datetime import datetime
from pathlib import Path

import bcrypt

USUARIOS_FILE = Path("usuarios/usuarios.json")
PERFIS_DIR = Path("usuarios/perfis")
PERFIL_MAX_BYTES = 2048  # 2KB máximo por perfil


def hash_senha(senha: str) -> str:
    return bcrypt.hashpw(senha.encode(), bcrypt.gensalt()).decode()


def _verificar_hash(senha: str, hash_armazenado: str) -> bool:
    return bcrypt.checkpw(senha.encode(), hash_armazenado.encode())


def carregar_usuarios() -> dict:
    if not USUARIOS_FILE.exists():
        # cria admin padrão na primeira vez
        USUARIOS_FILE.parent.mkdir(exist_ok=True)
        usuarios = {"admin": {"senha": hash_senha("admin123"), "nome": "Administrador"}}
        USUARIOS_FILE.write_text(json.dumps(usuarios, indent=2), encoding="utf-8")
    return json.loads(USUARIOS_FILE.read_text(encoding="utf-8"))


def verificar_login(usuario: str, senha: str) -> bool:
    usuarios = carregar_usuarios()
    return usuario in usuarios and _verificar_hash(senha, usuarios[usuario]["senha"])


def cadastrar_usuario(usuario: str, senha: str, nome: str) -> bool:
    usuarios = carregar_usuarios()
    if usuario in usuarios:
        return False
    usuarios[usuario] = {"senha": hash_senha(senha), "nome": nome}
    USUARIOS_FILE.write_text(json.dumps(usuarios, indent=2), encoding="utf-8")
    PERFIS_DIR.mkdir(exist_ok=True)
    return True


def get_nome(usuario: str) -> str:
    usuarios = carregar_usuarios()
    return usuarios.get(usuario, {}).get("nome", usuario)


def carregar_perfil(usuario: str) -> dict:
    caminho = PERFIS_DIR / f"{usuario}.json"
    if not caminho.exists():
        return {
            "temas": {"ultima_semana": {}, "ultimo_mes": {}, "total": {}},
            "documentos_analisados": [],
            "total_sessoes": 0,
            "total_perguntas": 0,
            "feedbacks": {"positivo": 0, "negativo": 0},
            "ultima_sessao": None,
        }
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_perfil(usuario: str, perfil: dict):
    PERFIS_DIR.mkdir(exist_ok=True)
    caminho = PERFIS_DIR / f"{usuario}.json"
    # compacta se passar de 2KB
    texto = json.dumps(perfil, indent=2, ensure_ascii=False)
    if len(texto.encode()) > PERFIL_MAX_BYTES:
        perfil = compactar_perfil(perfil)
        texto = json.dumps(perfil, indent=2, ensure_ascii=False)
    caminho.write_text(texto, encoding="utf-8")


def compactar_perfil(perfil: dict) -> dict:
    # zera ultima_semana, move ultimo_mes pra total, descarta detalhes
    for tema, count in perfil["temas"]["ultimo_mes"].items():
        perfil["temas"]["total"][tema] = perfil["temas"]["total"].get(tema, 0) + count
    perfil["temas"]["ultima_semana"] = {}
    perfil["temas"]["ultimo_mes"] = {}
    # mantém só os 5 documentos mais recentes
    perfil["documentos_analisados"] = perfil["documentos_analisados"][-5:]
    return perfil


def atualizar_perfil(usuario: str, evento: dict):
    """
    Atualiza silenciosamente o perfil com um evento.
    evento pode ter: tema, documento_analisado, feedback, nova_sessao
    """
    perfil = carregar_perfil(usuario)
    agora = datetime.now().isoformat()

    if "tema" in evento and evento["tema"] not in ("conversacional", "geral"):
        tema = evento["tema"]
        for periodo in ["ultima_semana", "ultimo_mes", "total"]:
            perfil["temas"][periodo][tema] = perfil["temas"][periodo].get(tema, 0) + 1

    if "documento_analisado" in evento:
        doc = evento["documento_analisado"]
        if doc not in perfil["documentos_analisados"]:
            perfil["documentos_analisados"].append(doc)

    if "feedback" in evento:
        perfil["feedbacks"][evento["feedback"]] = perfil["feedbacks"].get(evento["feedback"], 0) + 1

    if "nova_sessao" in evento:
        perfil["total_sessoes"] += 1
        perfil["ultima_sessao"] = agora

    perfil["total_perguntas"] = perfil.get("total_perguntas", 0) + 1
    salvar_perfil(usuario, perfil)


def gerar_contexto_perfil(usuario: str) -> str:
    """Gera texto de contexto pro SYSTEM_PROMPT baseado no perfil."""
    perfil = carregar_perfil(usuario)
    partes = []

    temas_recentes = perfil["temas"]["ultimo_mes"]
    if temas_recentes:
        top = max(temas_recentes, key=temas_recentes.get)
        partes.append(f"Este profissional consulta frequentemente sobre {top}.")

    if perfil["total_sessoes"] > 10:
        partes.append("É um usuário experiente do sistema.")
    elif perfil["total_sessoes"] > 0:
        partes.append("Ainda está conhecendo o sistema.")

    if perfil["documentos_analisados"]:
        ultimo = perfil["documentos_analisados"][-1]
        partes.append(f"Último documento analisado: {ultimo}.")

    return " ".join(partes)
