import os
import streamlit as st
from pinecone import Pinecone
from groq import Groq
from typing import List, Set
import re
import unicodedata
import nltk
nltk.download("punkt_tab")
nltk.download("punkt")


os.environ['PINECONE_API_KEY'] = ''
os.environ['GROQ_API_KEY'] = ''


def read_and_chunk_sentences(
    file_path: str,
    chunk_size: int = 40,
    overlap: int = 10
) -> List[str]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text, language='spanish')
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

def upload_documents(file_path: str, index_name: str, namespace: str):
    chunks = read_and_chunk_sentences(file_path, chunk_size=5, overlap=2)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "_id": f"{namespace}_chunk_{i + 1}",
            "chunk_text": chunk,
            "category": namespace
        })
    print(documents)

    index_pc = pc.Index(index_name)

    index_pc.upsert_records(namespace=namespace, records=documents)
    print(f"Subidos {len(documents)} chunks al índice '{index_name}' en el namespace '{namespace}'.")


files_and_namespaces = [
    ("CV LARA ROSENBERG.txt", "cv-lara-ns"),
    ("CV VICTORIA TERAN.txt", "cv-victoria-ns"),
    ("CV CLAUDIO BARRIL.txt", "cv-claudio-ns")
]


for file, namespace in files_and_namespaces:
    index_name = f"{namespace}-index"


    if not pc.list_indexes() or index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
        print(f"Índice '{index_name}' creado.")


    #upload_documents(file, index_name, namespace)


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in texto if unicodedata.category(c) != 'Mn')

person_patterns = {
    "cv-lara-ns-index": re.compile(r"\blara(\s+rosenberg)?\b|\brosenberg\b", re.IGNORECASE),
    "cv-victoria-ns-index": re.compile(r"\bvictoria(\s+teran)?\b|\bteran\b", re.IGNORECASE),
    "cv-claudio-ns-index": re.compile(r"\bclaudio(\s+barril)?\b|\bbarril\b", re.IGNORECASE)
}

def identificar_personas_mencionadas(texto: str) -> Set[str]:
    texto_normalizado = normalizar(texto)
    indices_decididos = set()

    for idx, pat in person_patterns.items():
        if pat.search(texto_normalizado):
            indices_decididos.add(idx)

    return indices_decididos

def search_similar(texto, top_k=10, indices=None):
    contexto = ""

    if not indices:
        indices = ["cv-lara-ns-index"]


    for indice in indices:
        namespace = indice.replace("-index", "")
        results = pc.Index(indice).search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    'text': texto
                }
            }
        )

        contexto += "\n".join([hit['fields']['chunk_text'] for hit in results['result']['hits']])

    return contexto

class Agent:
    def __init__(self, client, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = client
        self.model = model
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def __call__(self, message):
        self.add_message("user", message)
        result = self.execute(message)
        self.add_message("assistant", result)
        return result

    def execute(self, message):
        indices_decididos = identificar_personas_mencionadas(message)

        if not indices_decididos:
            indices_decididos = ["cv-lara-ns-index"]

        contexto = search_similar(message, indices=indices_decididos)

        prompt = f"""
Sos un asistente conversacional. Respondé de forma natural y conversacional.

Responde basandote en el contexto. Si no tenes informacion, aclaralo.
Consulta previa (historial): {self.messages}

Consulta: {message}

Contexto:
{contexto}
"""
        self.add_message("user", prompt.strip())

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7
        )
        return completion.choices[0].message.content

groq_client = Groq(api_key=GROQ_API_KEY)
agente = Agent(client=groq_client)

# Configuración de Streamlit
st.title("Consulta los CVs con IA")
st.write("Este es un asistente conversacional que responde preguntas basadas en los currículums cargados.")

consulta_usuario = st.text_input("Haz una pregunta:")

if consulta_usuario:
    respuesta = agente(consulta_usuario)

    st.write("Respuesta del asistente:")
    st.write(respuesta)