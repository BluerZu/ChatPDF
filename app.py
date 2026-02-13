import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# Imports para documento word y html
import docx
from bs4 import BeautifulSoup
import io

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat PDF con Gemini")

# Carga variables de entorno desde .env
# Aquí se espera GOOGLE_API_KEY=xxxx
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelo de embeddings local
# Se puede cambiar por otros modelos de sentence-transformers
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Se Inicializa el Cliente de ChromaDB
client = chromadb.Client()

# ============================================================
# SESSION STATE
# ============================================================
# session_state nos permite "recordar" cosas entre reruns.
if "collection" not in st.session_state:
    st.session_state.collection = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

# ============================================================
# INICIALIZACIÓN DEL HISTORIAL DE CHAT
# ============================================================
# Verificamos si la variable 'chat_history' ya existe en la sesión.
# st.session_state es la "memoria" que no se borra al recargar la app.
if "chat_history" not in st.session_state:
    # Si no existe, la creamos como una lista vacía [].
    # Aquí iremos guardando diccionarios con las preguntas y respuestas.
    st.session_state.chat_history = []

# ============================================================
# FUNCIONES
# ============================================================
def hash_pdf(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()

def get_files_text(uploaded_file):
    text = ""
    # --- ERROR COMÚN: Si tienes "for file in uploaded_file:" AQUÍ, BÓRRALO ---
    
    # Obtenemos la extensión directo del archivo único
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
                
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
                
        elif file_extension == 'txt':
            text += uploaded_file.getvalue().decode("utf-8")
            
        elif file_extension == 'html':
            soup = BeautifulSoup(uploaded_file, 'html.parser')
            text += soup.get_text()
            
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        
    return text


def chunk_text(text):
    """
    Divide un texto largo en fragmentos (chunks) con solapamiento.

    chunk_size:
        - Número máximo de caracteres por fragmento
        - Valores típicos: 400–800
        - Más grande = más contexto, pero embeddings más caros

    overlap:
        - Número de caracteres que se repiten entre chunks consecutivos
        - Evita que una idea quede cortada entre fragmentos
        - Regla común: 10–20% del chunk_size

    Devuelve:
        Lista de diccionarios, cada uno representando un chunk con:
        - id           -> identificador único
        - content      -> texto del fragmento
        - start_index  -> posición donde comienza en el texto original
        - size         -> longitud real del chunk
    """
    chunk_size = 500 
    overlap = 100
    chunks = []          # Aquí guardaremos todos los fragmentos
    start = 0            # Puntero que indica desde dónde empezamos a cortar
    chunk_id = 0         # Contador para asignar IDs únicos

    # El while se ejecuta mientras NO hayamos llegado al final del texto
    while start < len(text):

        # 1️⃣ Cortamos el texto desde 'start' hasta 'start + chunk_size'
        #    Python corta automáticamente si se pasa del largo del texto
        chunk_text = text[start:start + chunk_size]

        # 2️⃣ Guardamos el chunk junto con metadata útil
        chunks.append({
            "id": f"chunk_{chunk_id}",   # Identificador único del fragmento
            "content": chunk_text,       # Texto real del fragmento
            "start_index": start,        # Posición en el texto original
            "size": len(chunk_text)      # Tamaño real del fragmento
        })

        # 3️⃣ Incrementamos el ID para el próximo chunk
        chunk_id += 1

        # 4️⃣ Avanzamos el puntero 'start'
        #    No avanzamos chunk_size completo,
        #    sino (chunk_size - overlap) para que haya solapamiento
        #
        #    Ejemplo:
        #    chunk_size = 500
        #    overlap    = 100
        #    start avanza 400 caracteres
        #
        #    Los últimos 100 caracteres del chunk actual
        #    aparecerán también al inicio del siguiente
        start += chunk_size - overlap

    # 5️⃣ Cuando start >= len(text), el while termina
    #    y devolvemos todos los fragmentos creados
    return chunks



def create_chroma_collection(chunks):
    """
    Crea una colección nueva en ChromaDB a partir de los chunks generados.

    Cada chunk se almacena junto con:
    - su embedding (vector numérico)
    - su texto original
    - metadata útil
    """

    # ------------------------------
    # 1️⃣ Borrado defensivo
    # ------------------------------
    # Si ya existe una colección con el mismo nombre ("pdf_rag"),
    try:
        client.delete_collection("pdf_rag")
    except:
        # Si la colección no existe, Chroma lanza error.
        # Lo ignoramos porque es un caso esperado.
        pass

    # ------------------------------
    # 2️⃣ Crear colección nueva
    # ------------------------------
    # Aquí Chroma crea:
    # - una tabla de documentos
    # - un índice vectorial
    # - espacio para metadatos
    collection = client.create_collection(name="pdf_rag")

    # ------------------------------
    # 3️⃣ Separar texto de metadata
    # ------------------------------
    # Extraemos SOLO el contenido textual de cada chunk.
    # Esto es lo que se convertirá en embeddings.
    texts = [c["content"] for c in chunks]

    # ------------------------------
    # 4️⃣ Generar embeddings
    # ------------------------------
    # El modelo de SentenceTransformers convierte cada texto
    # en un vector numérico.
    #
    # Cada vector representa el significado del chunk.
    embeddings = EMBEDDING_MODEL.encode(texts)

    # ------------------------------
    # 5️⃣ Insertar datos en Chroma
    # ------------------------------
    collection.add(
        # Texto original del chunk
        documents=texts,

        # Vectores que permiten búsqueda semántica
        embeddings=embeddings.tolist(),

        # IDs únicos
        # Sirven para identificar cada chunk internamente
        ids=[c["id"] for c in chunks],

        # Metadata asociada a cada chunk
        metadatas=[
            {
                "chunk_index": i,         # Orden del chunk
                "start_index": c["start_index"],  # Posición en el texto original
                "chunk_size": c["size"]   # Tamaño real del fragmento
            }
            for i, c in enumerate(chunks)
        ]
    )

    # ------------------------------
    # 6️⃣ Devolver colección lista
    # ------------------------------
    # La colección ya puede:
    # - recibir queries (preguntas)
    # - devolver chunks relevantes
    return collection



def retrieve_context(collection, query, k=4):
    """
    Recupera los k chunks más similares a la pregunta.
    Devuelve tanto el texto como la metadata asociada.
    """
    query_embedding = EMBEDDING_MODEL.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    return results


def ask_gemini(context, question):
    """
    Llama a Gemini usando el contexto recuperado.
    El prompt fuerza comportamiento RAG (no inventar).
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    prompt = f"""
Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, di: "No se encuentra en el documento".

Contexto:
{context}

Pregunta:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# ============================================================
# INTERFAZ
# ============================================================

st.title("📄 Chat con PDF + ChromaDB + Gemini")

uploaded_file = st.file_uploader("Sube un PDF", type=["pdf", "txt", "docx", "html"])

# 🔄 Detectar cambio de PDF y resetear estado
if uploaded_file:
    current_hash = hash_pdf(uploaded_file)

    if st.session_state.pdf_hash != current_hash:
        st.session_state.pdf_hash = current_hash
        st.session_state.pdf_processed = False
        st.session_state.collection = None

# ------------------------------
# BOTÓN PROCESAR PDF
# ------------------------------
if uploaded_file and not st.session_state.pdf_processed:
    if st.button("📥 Procesar "):
        with st.spinner("Procesando ..."):
            text = get_files_text(uploaded_file)
            chunks = chunk_text(text)
            st.session_state.collection = create_chroma_collection(chunks)
            st.session_state.pdf_processed = True

        st.success(f" procesado ✅ ({len(chunks)} fragmentos)")

# ============================================================
# SECCIÓN DE PREGUNTAS (ÁREA PRINCIPAL)
# ============================================================
if st.session_state.pdf_processed and st.session_state.collection:
    st.divider()
    st.subheader("❓ Pregunta al documento")

    question = st.text_input("Escribe tu pregunta")

    if st.button("🤖 Preguntar") and question:
        with st.spinner("Buscando respuesta..."):
            # 1. Buscamos en ChromaDB
            results = retrieve_context(st.session_state.collection, question)
            context_text = "\n\n".join(results["documents"][0])
            
            # 2. Obtenemos respuesta de Gemini
            answer = ask_gemini(context_text, question)

            # 3. MOSTRAR RESPUESTA EN EL CENTRO
            st.subheader("🤖 Respuesta")
            st.write(answer)

            # 4. MOSTRAR CONTEXTO (Limpio y ordenado)
            with st.expander("📚 Ver detalles del contexto"):
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    st.markdown(f"**Fragmento #{meta['chunk_index']}**")
                    st.text(doc)
                    st.divider()

            # 5. GUARDADO INTELIGENTE
            # Verificamos si es una pregunta repetida para no llenar el historial de duplicados
            should_save = True
            if len(st.session_state.chat_history) > 0:
                if question == st.session_state.chat_history[-1]["q"]:
                    should_save = False
            
            if should_save:
                # Guardamos en la lista
                st.session_state.chat_history.append({"q": question, "a": answer})
                
                # 🔥 EL TRUCO FINAL: Forzamos la recarga de la página
                # Esto hace que la barra lateral se actualice AL INSTANTE.
                st.rerun()

# ============================================================
# BARRA LATERAL (HISTORIAL)
# ============================================================
with st.sidebar:
    st.header("⏳ Historial de Sesión")
    
    if st.session_state.chat_history:
        # Mostramos el historial (lo más nuevo arriba)
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**🧑 Tú:** {chat['q']}")
            st.info(f"**🤖 Bot:** {chat['a']}")
            st.divider()
    else:
        st.write("El historial está vacío.")
