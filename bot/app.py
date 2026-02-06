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
from pinecone import Pinecone, ServerlessSpec

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

INDEX_NAME = "new-bot"
EMBEDDING_DIM = 768

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys.")
    st.stop()

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
# VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(uploaded_pdf, namespace):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if namespace in existing_namespaces:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

vectorstore = load_vectorstore(uploaded_pdf, pdf_namespace)

# Slightly tighter retrieval for better citation precision
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 15}
)

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
# PROMPT WITH STRICT CITATION RULES
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise HR policy document assistant.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Answer ONLY from the provided context.\n"
            "2. If answer exists, provide it clearly.\n"
            "3. Use exact wording from the document when possible.\n"
            "4. NEVER use outside knowledge.\n"
            "5. If not found, respond EXACTLY: "
            "'I cannot find this information in the document.'\n"
            "6. After every factual statement, cite like this: [Page X]\n"
            "7. Do NOT invent page numbers.\n"
            "8. Do NOT mention chunks or context.\n\n"
            "OUTPUT:\n"
            "- Clear answer\n"
            "- Inline page citations only\n"
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]
)

# --------------------------------------------------
# ANSWER FUNCTION WITH TRUE CITATION EXTRACTION
# --------------------------------------------------
def answer_question(question):
    docs = retriever.invoke(question)

    if not docs:
        return "I cannot find this information in the document."

    # Attach page numbers directly in context
    context_parts = []

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        content = doc.page_content.strip()
        context_parts.append(f"[Page {page_num}]\n{content}")

    context = "\n\n".join(context_parts)

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    answer = response.content.strip()

    if answer.startswith("Answer:"):
        answer = answer[7:].strip()

    # Standard not found handling
    not_found_phrases = [
        "cannot find",
        "not found",
        "not mentioned",
        "not available",
        "not specified"
    ]

    if any(phrase in answer.lower() for phrase in not_found_phrases):
        return "I cannot find this information in the document."

    # Extract ONLY cited pages
    cited_pages = set(re.findall(r"\[Page (\d+)\]", answer))

    if cited_pages:
        pages = sorted(map(int, cited_pages))
        source_info = f"\n\n📄 **Source:** Page(s) {', '.join(map(str, pages))}"
    else:
        source_info = ""

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
