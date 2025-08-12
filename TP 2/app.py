import os
import streamlit as st
from pinecone import Pinecone
from groq import Groq
from typing import List
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

index_name = "cvlrf-index"

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

index_pc = pc.Index(index_name)

chunks = read_and_chunk_sentences("CV LARA ROSENBERG.txt", chunk_size=5, overlap=2)

documents = []
for i, chunk in enumerate(chunks):
    documents.append({
        "_id": f"cv_chunk_{i + 1}",
        "chunk_text": chunk,
        "category": "cv"
    })


namespace = "cvlrf-namespace"

#index_pc.upsert_records(namespace=namespace, records=documents)
#print(f"Subidos {len(documents)} chunks al índice.")

def search_similar(text, top_k=10, namespace="cvlrf-namespace", debug = False):
    results = index_pc.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "inputs": {
                'text': text
            }
        }
    )

    data = []
    for hit in results['result']['hits']:
        tmp = f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}"
        if debug:
            print(tmp)
        data.append(tmp)

    return data

client = Groq(api_key=GROQ_API_KEY)

class ChatSession:
    def __init__(self, client, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = client
        self.model = model
        self.messages = []

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def chat(self, msg, temperature=1, max_completion_tokens=1024, top_p=1, stream=True, stop=None):
        context_text = ""
        context = search_similar(msg, top_k=10, debug=False)
        context_text = "\n\n".join(context)
        prompt = f"""
Sos un asistente conversacional. Respondé de forma natural y conversacional.

Sin embargo, si la pregunta se refiere a Lara Rosenberg o su currículum vitae, usá exclusivamente la información del siguiente contexto para responder. Si no encontrás la respuesta en el contexto, decilo claramente. Si la pregunta no tiene que ver con Lara, respondé normalmente sin usar el contexto.
Consulta previa (historial): {self.messages}

Consulta: {msg}

Contexto:
{context_text}
"""
        self.add_user_message(prompt.strip())

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stream=stream,
            stop=stop
        )

        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            response += delta
        self.add_assistant_message(response)
        return response

# Configuramos Streamlit
st.set_page_config(page_title="CV Chat", layout="wide")
st.title("📄 Consulta el CV de Lara Rosenberg con IA")

# Iniciamos sesión de chat
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession(client)

# Botón para reiniciar la conversación
if st.button("🔄 Nueva conversación"):
    st.session_state.chat_session = ChatSession(client)
    st.rerun()

# Formulario de consulta
with st.form("formulario_consulta"):
    consulta = st.text_input("Escribí tu consulta:")
    enviar = st.form_submit_button("Enviar")

if enviar and consulta:
    with st.spinner("Consultando..."):
        respuesta = st.session_state.chat_session.chat(consulta)
        st.markdown("### 💬 Respuesta del modelo:")
        st.markdown(respuesta)

