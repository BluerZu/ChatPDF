# PDF Gemini QA con Streamlit y Pinecone

Este proyecto es una aplicación de preguntas y respuestas basada en contenidos de archivos PDF, utilizando Google Gemini (via LangChain) para el modelo de lenguaje y Pinecone como almacén vectorial. La interfaz se implementa con Streamlit, permitiendo una experiencia simple y rápida.

---

## 📝 Requisitos

- **Python 3.10** o superior
- Una cuenta de **Google Generative AI** con su clave API
- Una cuenta de **Pinecone** con su clave API y un índice configurado
- Conexión a Internet para llamar a las APIs

---

## 📦 Instalación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/BluerZu/ChatPDF
   cd ChatPDF
   ```

2. **Crear y activar un entorno virtual** (recomendado)

   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En macOS/Linux:
   source .venv/bin/activate
   ```

3. **Crear el archivo** `.env` en la raíz del proyecto con las siguientes variables (reemplaza `<apikey>` por tus claves reales):

   ```env
   GOOGLE_API_KEY=<api-key>
   PINECONE_API_KEY=<api-key>
   ```

   - `GOOGLE_API_KEY`: tu clave de acceso para Google Generative AI.
   - `PINECONE_API_KEY`: tu clave de Pinecone.
   

4. **Instalar dependencias**

   Ejecutar el siguiente comando para instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Uso

1. **Ejecutar la aplicación**

   ```bash
   streamlit run app.py
   ```

2. **En el navegador**, ve a la URL indicada (por defecto `http://localhost:8501`).

3. **Subir un PDF** (opcional):

   - Si deseas vectorizar un nuevo documento, súbelo y espera a que se procese.
   - Si no subes uno, la aplicación usará los documentos ya cargados en Pinecone.

4. **Escribir tu pregunta** en el campo de texto y presionar "💬 Preguntar".

   - La respuesta se mostrará en pantalla.
   - También verás un indicador de los tokens estimados utilizados.

---

## 📂 Estructura del proyecto

```
proyecto/
├── .env               # Variables de entorno
├── app.py             # Código principal de la app Streamlit
├── requirements.txt   # Dependencias del proyecto
└── hashes.txt         # Hashes de archivos procesados (se genera automáticamente)
```

---

## 🛠️ Personalizaciones

- Cambia el modelo `gemini-2.5-flash` por otra versión si lo deseas.
- Ajusta `chunk_size` y `chunk_overlap` en `RecursiveCharacterTextSplitter` según la longitud de tus PDFs.
- Agrega más callbacks o métricas siguiendo la documentación de LangChain.

---

## 🤝 Contribuciones

Si deseas contribuir, haz un fork del proyecto y envía un pull request con tus mejoras.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

