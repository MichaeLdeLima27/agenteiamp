import json
import os
import re
import pathlib
from pathlib import Path
from typing import TypedDict, Optional, List, Dict
from dotenv import load_dotenv

# --- Passos de Configuração ---
# 1. Garanta que você instalou todas as bibliotecas no terminal:
#    pip install --upgrade langchain_community faiss-cpu langchain-text-splitters pymupdf langgraph google-generativeai python-dotenv

# 2. Crie um arquivo .env na mesma pasta deste script com a linha:
#    GOOGLE_API_KEY="SUA_CHAVE_AQUI"

# 3. Crie uma pasta chamada 'pdfs' na mesma pasta deste script e coloque seus arquivos PDF lá.
# -----------------------------

# Carrega as variáveis do arquivo .env
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY não foi configurada como variável de ambiente.")

# Importações após a instalação
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm_triagem = ChatGoogleGenerativeAI(model="gemini-1.5-flashs", google_api_key=google_api_key)

# Carregar documentos PDF do diretório 'pdfs'
docs = []
pdf_dir = Path("./pdfs/")
pdf_dir.mkdir(exist_ok=True)

for n in pdf_dir.glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Carregado com sucesso arquivo {n.name}")
    except Exception as e:
        print(f"Erro ao carregar arquivo {n.name}: {e}")

if not docs:
    print("Nenhum arquivo PDF encontrado. Verifique se há arquivos na pasta 'pdfs'.")

print(f"Total de documentos carregados: {len(docs)}")

# Dividir documentos em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# Gerar embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

# Criar e configurar o vectorstore (FAISS) e o retriever
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                     search_kwargs={"score_threshold":0.3, "k": 4})

# Prompt para o agente RAG
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
      "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
      "Responda SOMENTE com base no contexto fornecido. "
      "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Funções auxiliares para formatação de citações
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

# Definindo a lógica da triagem (CORRIGIDO)
def triagem(pergunta: str) -> Dict:
    template = """
    Você é um agente de IA para triagem de tickets de Service Desk. Classifique a seguinte pergunta do usuário de acordo com as seguintes categorias e retorne um JSON.
    As categorias são:
    - 'AUTO_RESOLVER': Para perguntas que podem ser respondidas por um assistente de IA com base em uma base de dados.
    - 'ABRIR_CHAMADO': Para pedidos de exceções, liberações ou problemas técnicos que exigem intervenção humana.
    - 'PEDIR_INFO': Para perguntas que não são claras ou falta informação crucial para serem classificadas.

    Exemplo de formato de resposta para 'AUTO_RESOLVER':
    {{"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"}}

    Exemplo de formato de resposta para 'ABRIR_CHAMADO':
    {{"decisao": "ABRIR_CHAMADO", "urgencia": "MEDIA", "prioridade_negocio": "NORMAL"}}

    Exemplo de formato de resposta para 'PEDIR_INFO':
    {{"decisao": "PEDIR_INFO", "campos_faltantes": ["contexto", "tipo de acesso"]}}

    A urgência pode ser 'BAIXA', 'MEDIA', 'ALTA'. Considere 'ALTA' se o usuário usar palavras como "urgente", "crítico" ou mencionar interrupção no trabalho.

    Pergunta do usuário: {pergunta}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_triagem
    response = chain.invoke({"pergunta": pergunta})

    # Extrair e limpar o JSON
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        print("Erro ao decodificar JSON. Retornando padrão.")
        return {"decisao": "PEDIR_INFO", "campos_faltantes": ["informacao_adicional"]}

    return {"decisao": "PEDIR_INFO", "campos_faltantes": ["informacao_adicional"]}

# Definição do State e dos Nodes para LangGraph
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])
    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if faltantes:
        detalhe = ",".join(faltantes)
    else:
        detalhe = "Tema e contexto específico"
    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir_chamado...")
    triagem = state["triagem"]
    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]
    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")
    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"
    state_da_pergunta = (state["pergunta"] or "").lower()
    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"
    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"

# Definindo o grafo
workflow = StateGraph(AgentState)
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})
workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

# Loop interativo para testar o agente no terminal
while True:
    pergunta_usuario = input("\nDigite sua pergunta (ou 'sair' para fechar): ")
    if pergunta_usuario.lower() == 'sair':
        break

    resposta_final = grafo.invoke({"pergunta": pergunta_usuario})
    triag = resposta_final.get("triagem", {})
    print(f"PERGUNTA: {pergunta_usuario}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")

    if resposta_final.get("citacoes"):
        print("CITAÇÕES:")
        for citacao in resposta_final.get("citacoes"):
            print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
            print(f"   Trecho: {citacao['trecho']}")
    print("------------------------------------")
