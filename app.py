import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG PDF Chat")
st.caption("Upload a PDF, build a FAISS knowledge base, and chat with it using an open-weight model via Groq.")

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# Cache the embedding model so it is loaded once per Streamlit session/app worker.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(uploaded_file):
    """Extract text from all pages of the uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping character-based chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_faiss_index(chunks, model):
    """Embed chunks and create a cosine-similarity FAISS index."""
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(question, chunks, index, model, top_k=TOP_K):
    """Retrieve the most relevant chunks for a question."""
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[idx], float(score)))

    return results


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks, client):
    """Ask Groq to answer only from retrieved PDF context."""
    context = "\n\n".join(
        f"--- Source chunk {i + 1} ---\n{chunk}"
        for i, (chunk, _) in enumerate(retrieved_chunks)
    )

    system_prompt = """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded document."

Do not invent facts. Keep the answer clear and concise.
When possible, mention the page number shown in the retrieved context.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} characters")
    st.write(f"**Chunk overlap:** {CHUNK_OVERLAP} characters")
    st.write(f"**Retrieved chunks:** {TOP_K}")

    if st.button("🗑️ Clear document"):
        for key in ["chunks", "index", "embeddings", "file_name", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()


# -----------------------------
# Upload and indexing
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned/image-only PDFs require OCR before their text can be retrieved.",
)

if uploaded_file:
    if st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Extracting text from PDF..."):
            text = extract_pdf_text(uploaded_file)

        if not text.strip():
            st.error(
                "No selectable text was found. This may be a scanned PDF. "
                "Add OCR support if you need to process scanned documents."
            )
            st.stop()

        with st.spinner("Creating chunks and FAISS embeddings..."):
            chunks = chunk_text(text)
            model = load_embedding_model()
            index, embeddings = build_faiss_index(chunks, model)

        st.session_state["file_name"] = uploaded_file.name
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index
        st.session_state["embeddings"] = embeddings
        st.session_state["messages"] = []

        st.success(
            f"Indexed **{len(chunks)} chunks** from **{uploaded_file.name}**."
        )

if "chunks" in st.session_state:
    st.info(
        f"📄 **{st.session_state['file_name']}** • "
        f"{len(st.session_state['chunks'])} chunks indexed"
    )

# -----------------------------
# Chat
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about your uploaded PDF..."
)

if question:
    client = get_groq_client()

    if client is None:
        st.error(
            "GROQ_API_KEY is not configured. Add it to Streamlit Cloud Secrets "
            "or set it as an environment variable locally."
        )
        st.stop()

    if "chunks" not in st.session_state:
        st.warning("Please upload and index a PDF first.")
        st.stop()

    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    model = load_embedding_model()

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            retrieved = retrieve_chunks(
                question,
                st.session_state["chunks"],
                st.session_state["index"],
                model,
            )

            answer = generate_answer(question, retrieved, client)

        st.markdown(answer)

        with st.expander("🔎 Retrieved context"):
            for i, (chunk, score) in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {i} — similarity: {score:.3f}**")
                st.write(chunk)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer}
    )
else:
    if "chunks" not in st.session_state:
        st.markdown(
            """
            ### How it works

            1. Upload a PDF.
            2. Text is extracted with `pypdf`.
            3. Text is split into overlapping chunks.
            4. Chunks are embedded with an open-source Sentence Transformer.
            5. Embeddings are stored in a FAISS vector index.
            6. Relevant chunks are retrieved for each question.
            7. An open-weight model on Groq generates the answer from the retrieved context.
            """
        )
