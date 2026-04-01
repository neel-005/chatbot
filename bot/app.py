import os
import time
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📘",
    layout="centered"
)

st.markdown("""
<h1 style="text-align:center;">PDF Q&A Chatbot</h1>
<p style="text-align:center; color:gray;">
Ask questions about your uploaded PDF documents.
</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "new-bot-fixed"
EMBEDDING_DIM = 384

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("❌ Missing API keys. Please set PINECONE_API_KEY and HUGGINGFACE_API_KEY in Streamlit Secrets.")
    st.stop()

# --------------------------------------------------
# CUSTOM EMBEDDING CLASS (free, API-based)
# --------------------------------------------------
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_url = "https://router.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en-v1.5"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(3):
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=60
            )

            if not response.ok:
                error_detail = f"Status {response.status_code}: {response.text}"

                if response.status_code == 503:
                    st.warning(f"⏳ Embedding model loading, retrying... (attempt {attempt+1}/3)")
                    time.sleep(10)
                    continue

                if response.status_code == 401:
                    st.error("❌ Invalid HuggingFace API key. Check your HUGGINGFACE_API_KEY secret.")
                    st.stop()

                if response.status_code == 403:
                    st.error("❌ Access denied to embedding model.")
                    st.stop()

                st.error(f"❌ Embedding API error: {error_detail}")
                st.stop()

            result = response.json()
            embeddings = []
            for item in result:
                if isinstance(item[0], list):
                    # 3D response: mean pool over token dimension
                    vec = [sum(col) / len(col) for col in zip(*item)]
                else:
                    vec = item
                embeddings.append(vec)
            return embeddings

        st.error("❌ Embedding model failed to load after 3 retries. Please try again in a minute.")
        st.stop()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# --------------------------------------------------
# PINECONE INIT
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
existing_namespaces = index.describe_index_stats().get("namespaces", {})

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Document Control")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if not uploaded_pdf:
    st.info("📎 Upload a PDF from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# SESSION / NAMESPACE
# --------------------------------------------------
pdf_namespace = uploaded_pdf.name.replace(" ", "_").lower()

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = pdf_namespace

if st.session_state.active_pdf != pdf_namespace:
    st.session_state.active_pdf = pdf_namespace
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.cache_resource.clear()

# --------------------------------------------------
# VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(pdf_bytes, namespace):
    embeddings = HFInferenceEmbeddings(api_key=HUGGINGFACE_API_KEY)

    if namespace in existing_namespaces:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

# Pass bytes so the cached function gets a hashable-friendly input
pdf_bytes = uploaded_pdf.read()
vectorstore = load_vectorstore(pdf_bytes, pdf_namespace)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

# --------------------------------------------------
# LLM — Zephyr-7B (free on HuggingFace)
# --------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def call_llm(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "return_full_text": False
        }
    }
    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=60
            )

            if not response.ok:
                if response.status_code == 503:
                    st.warning(f"⏳ LLM loading, retrying... (attempt {attempt+1}/3)")
                    time.sleep(15)
                    continue
                if response.status_code == 401:
                    st.error("❌ Invalid HuggingFace API key.")
                    st.stop()
                st.error(f"❌ LLM API error — Status {response.status_code}: {response.text}")
                st.stop()

            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()

            return "I cannot find this information in the document."

        except requests.exceptions.Timeout:
            st.warning(f"⏳ LLM request timed out, retrying... (attempt {attempt+1}/3)")
            time.sleep(10)

    return "❌ LLM failed to respond after 3 retries. Please try again."

# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """You are a precise document assistant.

ABSOLUTE RULES:
1. Answer ONLY from the provided context chunks below
2. If the answer exists in ANY chunk, provide it
3. Quote exact text from the document when possible
4. NEVER use external knowledge or make assumptions
5. If you cannot find the answer in the context, say exactly: 'I cannot find this information in the document.'
6. Be comprehensive - check ALL chunks for relevant information
7. Combine information from multiple chunks if needed

OUTPUT FORMAT:
- Give a clear, direct answer
- Use the document's exact wording when available
- Keep it concise but complete
- Provide page number of the chunk that is in the answer
- Do NOT mention chunks, or context in your answer"""

# --------------------------------------------------
# ANSWER FUNCTION
# --------------------------------------------------
def answer_question(question: str, retriever) -> str:
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the document."

    context_parts = []
    page_set = set()

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        page_set.add(page_num)
        context_parts.append(doc.page_content.strip())

    context = "\n---\n".join(context_parts)

    # Zephyr instruct format
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\nContext:\n{context}\n\nQuestion: {question}</s>\n"
        f"<|assistant|>"
    )

    answer = call_llm(prompt)

    if answer.startswith("Answer:"):
        answer = answer[7:].strip()

    not_found_phrases = [
        "cannot find", "not found", "no information",
        "not mentioned", "not available"
    ]
    if any(p in answer.lower() for p in not_found_phrases):
        return "I cannot find this information in the document."

    pages = sorted(page_set)
    source_info = f"\n\n📄 **Source:** Page(s) {', '.join(map(str, pages[:3]))}"

    return answer + source_info

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about the PDF...")

if query and st.session_state.pending_question is None:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_question = query
    st.rerun()

if st.session_state.pending_question:
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document..."):
            answer = answer_question(
                st.session_state.pending_question,
                retriever
            )
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.pending_question = None
    st.rerun()
