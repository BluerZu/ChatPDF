import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# 1. CONFIGURACIÓN Y MODELOS
# ============================================================
st.set_page_config(page_title="Multi-Doc AI Assistant", layout="wide")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDING_MODEL = load_embedding_model()
CHROMA_CLIENT = chromadb.Client()

# ============================================================
# 2. EXTRACTORES DE TEXTO (PDF, DOCX, PPTX, TXT)
# ============================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content: text += f"\n[Página {i+1}]\n{content}"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_pptx(file):
    prs = Presentation(file)
    text = []
    for i, slide in enumerate(prs.slides):
        slide_txt = f"\n[Diapositiva {i+1}]\n"
        for shape in slide.shapes:
            if hasattr(shape, "text"): slide_txt += shape.text + " "
        text.append(slide_txt)
    return "\n".join(text)

def extract_text_from_txt(file):
    try:
        return file.getvalue().decode("utf-8")
    except:
        return file.getvalue().decode("latin-1")

def get_document_text(uploaded_file):
    ext = uploaded_file.name.lower()
    if ext.endswith(".pdf"): return extract_text_from_pdf(uploaded_file)
    if ext.endswith(".docx"): return extract_text_from_docx(uploaded_file)
    if ext.endswith(".pptx"): return extract_text_from_pptx(uploaded_file)
    if ext.endswith(".txt"): return extract_text_from_txt(uploaded_file)
    return None

# ============================================================
# 3. PROCESAMIENTO RAG MEJORADO (CON METADATOS)
# ============================================================

def chunk_text(text, size=600, overlap=120):
    """
    Ahora devuelve una lista de diccionarios con contenido y posición original.
    """
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + size
        chunk_content = text[start:end]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "content": chunk_content,
            "start_index": start,
            "size": len(chunk_content)
        })
        start += size - overlap
        chunk_id += 1
    return chunks

def create_collection(chunks):
    try: CHROMA_CLIENT.delete_collection("multi_doc_rag")
    except: pass
    
    collection = CHROMA_CLIENT.create_collection(name="multi_doc_rag")
    
    # Preparamos los datos para ChromaDB
    texts = [c["content"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"start_index": c["start_index"], "chunk_index": i} for i, c in enumerate(chunks)]
    
    embeddings = EMBEDDING_MODEL.encode(texts).tolist()
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return collection

# ============================================================
# 4. INTERFAZ DE USUARIO (STREAMLIT)
# ============================================================

st.title("🤖 Asistente Documental Inteligente")

if "processed" not in st.session_state:
    st.session_state.update({"processed": False, "collection": None, "file_hash": None})

with st.sidebar:
    st.header("Carga de Documento")
    uploaded_file = st.file_uploader("Sube tu archivo", type=["pdf", "docx", "pptx", "txt"])
    
    if uploaded_file:
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.file_hash != file_hash:
            st.session_state.update({"processed": False, "file_hash": file_hash})

    if uploaded_file and not st.session_state.processed:
        if st.button("Procesar Archivo"):
            with st.spinner("Analizando y fragmentando contenido..."):
                text = get_document_text(uploaded_file)
                if text:
                    chunks = chunk_text(text)
                    st.session_state.collection = create_collection(chunks)
                    st.session_state.processed = True
                    st.success(f"Archivo procesado: {len(chunks)} fragmentos generados.")

if st.session_state.processed:
    st.divider()
    query = st.text_input("Haz una pregunta sobre el documento:")
    
    if query:
        with st.spinner("Buscando en la base de datos vectorial..."):
            # 1. Recuperar contexto y metadatos
            query_emb = EMBEDDING_MODEL.encode([query]).tolist()
            results = st.session_state.collection.query(query_embeddings=query_emb, n_results=4)
            
            context = "\n\n".join(results["documents"][0])
            
            # 2. Llamada a Gemini
            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
                prompt = f"""
                Eres un asistente experto. Responde la pregunta basándote únicamente en el contexto proporcionado.
                Si la respuesta no está en el contexto, indica que no se encuentra en el documento.

                Contexto:
                {context}

                Pregunta:
                {query}
                """
                response = model.generate_content(prompt)
                
                st.subheader(" Respuesta:")
                st.write(response.text)
                
                # 3. MOSTRAR CHUNKS (Igual al código original)
                with st.expander(" Fuentes y Contexto Utilizado"):
                    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                        st.markdown(f"**Fragmento #{meta['chunk_index']}**")
                        st.caption(f"📍 Posición inicial en texto: {meta['start_index']}")
                        st.info(doc)
                        
            except Exception as e:
                st.error(f"Error en la comunicación con la IA: {e}")