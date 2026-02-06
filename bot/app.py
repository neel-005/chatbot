#hello
import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Question Answering Bot")
st.caption("Answers are based only on the uploaded document")

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "new-bot-gte"
EMBEDDING_DIM = 1024   # gte-large dimension

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys in environment.")
    st.stop()

# --------------------------------------------------
# PINECONE INIT (NO AUTO CREATION)
# --------------------------------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception:
    st.error(
        f"Pinecone index '{INDEX_NAME}' not found.\n\n"
        "Create it manually in Pinecone dashboard with:\n"
        "- Dimension: 1024\n"
        "- Metric: cosine\n"
        "- Cloud: AWS"
    )
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if not uploaded_pdf:
    st.info("Upload a PDF to begin.")
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
# VECTORSTORE (GTE-LARGE)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(uploaded_pdf, namespace):
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,   # reduced overlap (important)
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    # Create empty vectorstore first
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )

    # Batch upload to prevent memory crash
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)

    return vectorstore


# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        temperature=0.0,
        max_new_tokens=500,
        repetition_penalty=1.15,
        top_p=0.95,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise HR policy document assistant.\n\n"
            "RULES:\n"
            "1. Answer ONLY from the provided context.\n"
            "2. Use exact wording when possible.\n"
            "3. NEVER use outside knowledge.\n"
            "4. If not found, respond EXACTLY: "
            "'I cannot find this information in the document.'\n"
            "5. Do NOT mention pages inside the answer.\n"
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]
)

# --------------------------------------------------
# ANSWER FUNCTION
# --------------------------------------------------
def answer_question(question):
    docs = retriever.invoke(question)

    if not docs:
        return "I cannot find this information in the document."

    context_parts = []
    page_numbers = []

    for doc in docs:
        page_num = int(doc.metadata.get("page", 0)) + 1
        page_numbers.append(page_num)
        context_parts.append(doc.page_content.strip())

    context = "\n\n".join(context_parts)

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    answer = response.content.strip()

    # Remove accidental inline citations
    answer = re.sub(r"\[Page \d+\]", "", answer).strip()

    not_found_phrases = [
        "cannot find",
        "not found",
        "not mentioned",
        "not available",
        "not specified"
    ]

    if any(phrase in answer.lower() for phrase in not_found_phrases):
        return "I cannot find this information in the document."

    # Clean citation output (top 2 pages only)
    unique_pages = sorted(set(page_numbers))[:2]

    source_info = f"\n\n📄 Source: Page(s) {', '.join(map(str, unique_pages))}"

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

query = st.chat_input("Ask a question about your document")

if query and st.session_state.pending_question is None:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_question = query
    st.rerun()

if st.session_state.pending_question:
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            answer = answer_question(st.session_state.pending_question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.pending_question = None
    st.rerun()
