import os
import html
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI / CSS
# ============================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    [data-testid="stAppViewContainer"] {
        background: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6e9ef;
        min-width: 270px;
        max-width: 270px;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 24px 18px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 28px;
        padding: 0 8px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        background: #111827;
        color: white;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    .brand-title {
        font-size: 18px;
        font-weight: 700;
        color: #172033;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #8a93a5;
        margin-top: 3px;
    }

    .side-heading {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #9aa2b1;
        letter-spacing: .7px;
        margin: 25px 8px 12px;
    }

    .doc-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 11px 9px;
        border-radius: 9px;
        background: #f7f8fa;
        margin-bottom: 7px;
    }

    .doc-icon {
        width: 34px;
        height: 34px;
        background: #fff0f0;
        color: #e44b4b;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
    }

    .doc-name {
        font-size: 12px;
        font-weight: 600;
        color: #172033;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .doc-meta {
        font-size: 10px;
        color: #9aa2b1;
        margin-top: 3px;
    }

    .system-card {
        background: #f7f8fa;
        border-radius: 12px;
        padding: 14px;
        margin-top: 28px;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 600;
        color: #172033;
    }

    .green-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
    }

    .system-text {
        color: #7d8696;
        margin-top: 7px;
        font-size: 11px;
        line-height: 1.4;
    }

    /* ---------- Top bar ---------- */
    .topbar {
        height: 70px;
        background: #ffffff;
        border-bottom: 1px solid #e6e9ef;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 35px;
        margin: 0 -1rem 0 -1rem;
    }

    .topbar-title {
        font-size: 15px;
        font-weight: 700;
        color: #172033;
    }

    .model-pill {
        background: #f2f4f7;
        border-radius: 20px;
        padding: 8px 13px;
        font-size: 11px;
        color: #667085;
    }

    /* ---------- Welcome ---------- */
    .welcome {
        text-align: center;
        padding: 42px 15px 28px;
    }

    .welcome-icon {
        width: 65px;
        height: 65px;
        margin: 0 auto 18px;
        border-radius: 18px;
        background: #111827;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }

    .welcome h1 {
        font-size: 30px;
        color: #172033;
        margin: 0 0 10px 0;
    }

    .welcome p {
        color: #7b8494;
        font-size: 14px;
        max-width: 560px;
        line-height: 1.6;
        margin: auto;
    }

    /* ---------- Upload ---------- */
    .upload-label {
        border: 1.5px dashed #cbd1dc;
        background: #ffffff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 0 auto 22px;
        max-width: 850px;
    }

    .upload-icon {
        width: 45px;
        height: 45px;
        margin: auto;
        border-radius: 12px;
        background: #f0f2f5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .upload-title {
        font-size: 14px;
        font-weight: 700;
        color: #172033;
        margin-top: 12px;
    }

    .upload-subtitle {
        font-size: 11px;
        color: #939baa;
        margin-top: 6px;
    }

    /* Style Streamlit uploader */
    [data-testid="stFileUploader"] {
        max-width: 850px;
        margin: 0 auto;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff;
        border: 1.5px dashed #cbd1dc;
        border-radius: 15px;
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #7b8494;
    }

    /* ---------- Suggestion cards ---------- */
    .suggestion-card {
        background: #ffffff;
        border: 1px solid #e7eaf0;
        border-radius: 12px;
        padding: 14px;
        min-height: 80px;
    }

    .suggestion-title {
        font-size: 12px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .suggestion-text {
        color: #8992a2;
        font-size: 11px;
        line-height: 1.4;
    }

    /* ---------- Chat ---------- */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
    }

    [data-testid="stChatMessage"] {
        border-radius: 13px;
        margin-bottom: 12px;
    }

    /* ---------- Info ---------- */
    .document-info {
        max-width: 850px;
        margin: 0 auto 15px;
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 11px 14px;
        color: #667085;
        font-size: 12px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 800px) {
        [data-testid="stSidebar"] {
            min-width: 0;
            max-width: 100%;
        }

        .topbar {
            padding: 0 15px;
        }

        .welcome h1 {
            font-size: 24px;
        }

        .welcome {
            padding-top: 25px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "chunks": None,
    "index": None,
    "embeddings": None,
    "file_name": None,
    "file_size": None,
    "messages": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# MODELS
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# ============================================================
# FUNCTIONS
# ============================================================
def extract_pdf_text(uploaded_file):
    """Extract selectable text from every PDF page."""
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
        raise ValueError("Chunk overlap must be smaller than chunk size.")

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
    """Create normalized embeddings and a FAISS inner-product index."""
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
    """Retrieve the most relevant document chunks."""
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
    """Load Groq API key from environment or Streamlit secrets."""
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
    """Generate an answer using only retrieved document context."""
    context = "\n\n".join(
        f"--- Source chunk {i + 1} ---\n{chunk}"
        for i, (chunk, _) in enumerate(retrieved_chunks)
    )

    system_prompt = """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded document."

Do not invent facts.

Keep the answer clear, useful and concise.

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
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def clear_document():
    """Clear the active document and conversation."""
    for key in [
        "chunks",
        "index",
        "embeddings",
        "file_name",
        "file_size",
        "messages",
    ]:
        st.session_state[key] = None if key != "messages" else []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">📚</div>
        <div>
            <div class="brand-title">RAG Chat</div>
            <div class="brand-subtitle">AI Document Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "＋  New Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="side-heading">Documents</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.file_name:

        size_mb = (
            st.session_state.file_size / (1024 * 1024)
            if st.session_state.file_size
            else 0
        )

        safe_name = html.escape(st.session_state.file_name)

        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-icon">📄</div>
            <div style="min-width:0;">
                <div class="doc-name">{safe_name}</div>
                <div class="doc-meta">
                    {size_mb:.1f} MB · {len(st.session_state.chunks or [])} chunks
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🗑️ Clear document",
            use_container_width=True,
        ):
            clear_document()
            st.rerun()

    else:
        st.markdown("""
        <div class="doc-card">
            <div class="doc-icon">📄</div>
            <div>
                <div class="doc-name">No document uploaded</div>
                <div class="doc-meta">PDF files supported</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="side-heading">Settings</div>

    <div style="font-size:11px;color:#7d8696;line-height:1.8;padding:0 8px;">
        <b>Embedding</b><br>
        {EMBEDDING_MODEL}<br><br>

        <b>LLM</b><br>
        {GROQ_MODEL}<br><br>

        <b>Retrieval</b><br>
        Top {TOP_K} chunks
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="system-card">
        <div class="system-status">
            <span class="green-dot"></span>
            System Ready
        </div>

        <div class="system-text">
            FAISS vector database connected
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TOP BAR
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-title">
        Document Q&A
    </div>

    <div class="model-pill">
        ⚡ Groq · Open-weight LLM
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================
main = st.container()

with main:

    # --------------------------------------------------------
    # Welcome section
    # --------------------------------------------------------
    if not st.session_state.messages:

        st.markdown("""
        <div class="welcome">

            <div class="welcome-icon">
                ✨
            </div>

            <h1>Chat with your documents</h1>

            <p>
                Upload a PDF and ask questions about its content.
                The AI retrieves relevant information from your document
                before generating an answer.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # PDF upload
    # --------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload a text-based PDF. Scanned/image-only PDFs require OCR.",
        label_visibility="collapsed",
    )

    # Upload instructions shown above the actual uploader
    if not uploaded_file:
        st.markdown("""
        <div class="upload-label">
            <div class="upload-icon">⬆</div>
            <div class="upload-title">
                Click the box above to upload your PDF
            </div>
            <div class="upload-subtitle">
                Text-based PDF documents are supported
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Process uploaded PDF
    # --------------------------------------------------------
    if uploaded_file:

        # Avoid rebuilding the vector database every Streamlit rerun.
        if st.session_state.file_name != uploaded_file.name:

            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_text(uploaded_file)

            if not text.strip():
                st.error(
                    "No selectable text was found. This may be a scanned "
                    "or image-only PDF. OCR support is required for scanned PDFs."
                )
                st.stop()

            with st.spinner("Creating chunks and FAISS embeddings..."):

                chunks = chunk_text(text)

                if not chunks:
                    st.error("No usable text chunks were created.")
                    st.stop()

                model = load_embedding_model()

                index, embeddings = build_faiss_index(
                    chunks,
                    model,
                )

            st.session_state.file_name = uploaded_file.name
            st.session_state.file_size = uploaded_file.size
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.session_state.messages = []

            st.success(
                f"Document indexed successfully — {len(chunks)} chunks created."
            )

            st.rerun()

    # --------------------------------------------------------
    # Active document information
    # --------------------------------------------------------
    if st.session_state.chunks:

        safe_name = html.escape(st.session_state.file_name)

        st.markdown(f"""
        <div class="document-info">
            📄 <b>{safe_name}</b>
            &nbsp;•&nbsp;
            {len(st.session_state.chunks)} chunks indexed
            &nbsp;•&nbsp;
            FAISS ready
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------
    if (
        st.session_state.chunks
        and not st.session_state.messages
    ):

        st.markdown(
            '<div style="max-width:850px;margin:0 auto 10px;'
            'font-size:12px;font-weight:700;color:#667085;">'
            'Try asking</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div class="suggestion-card">
                <div class="suggestion-title">
                    📋 Summarize
                </div>
                <div class="suggestion-text">
                    Give me a summary of this document
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="suggestion-card">
                <div class="suggestion-title">
                    🔎 Find key topics
                </div>
                <div class="suggestion-text">
                    What are the main topics discussed?
                </div>
            </div>
            """, unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("""
            <div class="suggestion-card">
                <div class="suggestion-title">
                    💡 Key findings
                </div>
                <div class="suggestion-text">
                    Identify the most important findings
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="suggestion-card">
                <div class="suggestion-title">
                    🎓 Explain
                </div>
                <div class="suggestion-text">
                    Explain the document in simple terms
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Chat history
    # --------------------------------------------------------
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"],
            avatar="👤" if message["role"] == "user" else "✨",
        ):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("🔎 Retrieved context"):
                    for i, (chunk, score) in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Chunk {i} — similarity: {score:.3f}**"
                        )
                        st.write(chunk)

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        client = get_groq_client()

        if client is None:
            st.error(
                "GROQ_API_KEY is not configured. Add it to Streamlit "
                "Cloud Secrets or set it as an environment variable locally."
            )
            st.stop()

        if not st.session_state.chunks:
            st.warning(
                "Please upload and index a PDF before asking a question."
            )
            st.stop()

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Retrieve + generate
        model = load_embedding_model()

        with st.chat_message("assistant", avatar="✨"):

            with st.spinner(
                "Searching the document and generating an answer..."
            ):

                retrieved = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.index,
                    model,
                )

                answer = generate_answer(
                    question,
                    retrieved,
                    client,
                )

            st.markdown(answer)

            with st.expander("🔎 Retrieved context"):
                for i, (chunk, score) in enumerate(
                    retrieved,
                    start=1,
                ):
                    st.markdown(
                        f"**Chunk {i} — similarity: {score:.3f}**"
                    )
                    st.write(chunk)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved,
            }
        )
