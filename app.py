import os  # Para manejo de variables de entorno
import hashlib  # Para generar hash único del archivo PDF
from pathlib import Path  # Para verificar si existen archivos locales (hashes.txt)
import fitz  # Librería PyMuPDF: lectura y extracción de texto desde archivos PDF
import streamlit as st  # Para crear la interfaz web interactiva
from dotenv import load_dotenv  # Para cargar las variables de entorno desde el archivo .env

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Modelo Gemini y embeddings de Google
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Para dividir el texto del PDF en fragmentos pequeños
from langchain_pinecone import PineconeVectorStore  # Conexión con el índice de vectores en Pinecone
from langchain.chains import ConversationalRetrievalChain  # Cadena que permite preguntas con recuperación de contexto
from langchain.chains.conversation.memory import ConversationBufferMemory  # Memoria para mantener el historial de conversación
from langchain.callbacks import get_openai_callback  # Para medir tokens utilizados (funciona con modelos compatibles)
from pinecone import Pinecone, ServerlessSpec  # Cliente de Pinecone y configuración de región para índices vectoriales


# === Configuración inicial ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "asistente"

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Inicializar modelo y embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

# === Funciones ===
def hash_archivo(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def ya_existente(file_hash: str) -> bool:
    if not Path("hashes.txt").exists():
        return False
    with open("hashes.txt", "r") as f:
        return file_hash in f.read()

def guardar_hash(file_hash: str):
    with open("hashes.txt", "a") as f:
        f.write(file_hash + "\n")

def leer_pdf_bytes(content: bytes) -> str:
    texto = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            texto += page.get_text()
    return texto

def fragmentar(texto: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    return splitter.create_documents([texto])

def vectorizar_documento(texto: str):
    documentos = fragmentar(texto)
    PineconeVectorStore.from_documents(documentos, index_name=INDEX_NAME, embedding=embedding)

def nueva_conversacion():
    memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memoria
    )

# === Interfaz Streamlit ===
st.set_page_config(page_title="PDF Gemini QA", layout="centered")
st.title("📄🔍 Pregunta a tu PDF con Gemini + Pinecone")

# Subida de archivo PDF (opcional)
uploaded_file = st.file_uploader("Sube un archivo PDF (opcional)", type="pdf")

if uploaded_file:
    content = uploaded_file.read()
    file_hash = hash_archivo(content)

    if ya_existente(file_hash):
        st.info("✅ Este archivo ya fue vectorizado anteriormente.")
    else:
        with st.spinner("📚 Procesando y vectorizando el PDF..."):
            texto = leer_pdf_bytes(content)
            if texto.strip():
                vectorizar_documento(texto)
                guardar_hash(file_hash)
                st.success("✅ Documento vectorizado y almacenado correctamente.")
            else:
                st.error("⚠️ No se pudo extraer texto del PDF.")

# Crear cadena de conversación si no existe
if "chain" not in st.session_state:
    st.session_state.chain = nueva_conversacion()

# Entrada de pregunta
pregunta = st.text_input("📝 Escribe tu pregunta:")

if st.button("💬 Preguntar") and pregunta.strip():
    with st.spinner("🧠 Pensando..."):
        try:
            with get_openai_callback() as cb:
                respuesta = st.session_state.chain.run(pregunta)
                st.success(respuesta)
                st.info(f"🔢 Tokens estimados utilizados: {cb.total_tokens}")
        except Exception as e:
            st.error(f"❌ Ocurrió un error: {e}")
