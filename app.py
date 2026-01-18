import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from docx import Document
from pptx import Presentation

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat basado en Documentos con Gemini")

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

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ============================================================
# FUNCIONES
# ============================================================
def hash_file(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()

def extract_text_from_file(file):
    """
    Extrae texto de un archivo según su tipo.
    Para PDFs, incluye el número de página como marcador.
    """
    filename = file.name.lower()
    if filename.endswith('.pdf'):
        reader = PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                text += f"\n[Página {i+1}]\n{content}"
        return text
    elif filename.endswith('.docx'):
        doc = Document(file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    elif filename.endswith('.pptx'):
        prs = Presentation(file)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + '\n'
        return text
    elif filename.endswith('.txt'):
        return str(file.read(), 'utf-8')
    else:
        raise ValueError("Tipo de archivo no soportado. Usa PDF, DOCX, PPTX o TXT.")


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
    chunk_size = 800
    overlap = 160
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



def retrieve_context(collection, query, k=10):
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

st.title("📄 Chat con Documentos + ChromaDB + Gemini")

uploaded_file = st.file_uploader("Sube un documento", type=["pdf", "docx", "txt", "pptx"])

# 🔄 Detectar cambio de archivo y resetear estado
if uploaded_file:
    current_hash = hash_file(uploaded_file)

    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        st.session_state.file_processed = False
        st.session_state.collection = None

# ------------------------------
# BOTÓN PROCESAR DOCUMENTO
# ------------------------------
if uploaded_file and not st.session_state.file_processed:
    if st.button("📥 Procesar Documento"):
        with st.spinner("Procesando documento..."):
            text = extract_text_from_file(uploaded_file)
            chunks = chunk_text(text)
            st.session_state.collection = create_chroma_collection(chunks)
            st.session_state.file_processed = True

        st.success(f"Documento procesado ✅ ({len(chunks)} fragmentos)")

# ------------------------------
# SECCIÓN DE PREGUNTAS
# ------------------------------
if st.session_state.file_processed and st.session_state.collection:
    st.divider()
    st.subheader("❓ Pregunta al documento")

    question = st.text_input("Escribe tu pregunta")

    if st.button("🤖 Preguntar") and question:
        with st.spinner("Buscando respuesta..."):
            results = retrieve_context(st.session_state.collection, question)

            # Filtrar chunks por umbral de similitud (distancia coseno < 0.3)
            threshold = 0.3
            distances = results["distances"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            filtered_docs = [doc for doc, dist in zip(documents, distances) if dist < threshold]
            filtered_metas = [meta for meta, dist in zip(metadatas, distances) if dist < threshold]
            filtered_distances = [dist for dist in distances if dist < threshold]

            # Asegurar al menos 2 chunks
            if len(filtered_docs) < 2:
                # Tomar los top 2 más relevantes
                sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
                top2_docs = [documents[i] for i in sorted_indices[:2]]
                top2_metas = [metadatas[i] for i in sorted_indices[:2]]
                top2_distances = [distances[i] for i in sorted_indices[:2]]
                filtered_docs = top2_docs
                filtered_metas = top2_metas
                filtered_distances = top2_distances

            # Unimos los documentos filtrados para Gemini
            context_text = "\n\n".join(filtered_docs)

            answer = ask_gemini(context_text, question)

        st.subheader("🤖 Respuesta")
        st.write(answer)

        # ------------------------------
        # DETALLE DEL CONTEXTO USADO
        # ------------------------------
        with st.expander(f"📚 Contexto usado ({len(filtered_docs)} chunks filtrados)"):
            for i, (doc, meta) in enumerate(zip(filtered_docs, filtered_metas)):
                st.markdown(f"""
                **Chunk #{meta['chunk_index']}**
                - 📍 Inicio en texto: `{meta['start_index']}`
                - 📏 Tamaño: `{meta['chunk_size']}` caracteres
                - 📊 Similitud: `{1 - filtered_distances[i]:.3f}`

                ```text
                {doc}
                """)