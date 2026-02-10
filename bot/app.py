import os
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
# 🌈 PAGE CONFIG (PREVIOUS UI STYLE)
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
 
INDEX_NAME = "new-bot-gte"
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
# 📂 SIDEBAR (PREVIOUS UI STYLE)
# --------------------------------------------------
with st.sidebar:
    st.markdown("Document Control")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
 
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
 
if not uploaded_pdf:
    st.info("Upload a PDF from the sidebar to begin")
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
        model_name="Alibaba-NLP/gte-base-en-v1.5",
        model_kwargs={"trust_remote_code": True}
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
# PROMPT (UNCHANGED)
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise HR policy document assistant.\n\n"
            "ABSOLUTE RULES - FOLLOW EXACTLY:\n"
            "1. Answer ONLY from the provided context chunks below\n"
            "2. If the answer exists in ANY chunk, provide it\n"
            "3. Quote exact text from the document when possible\n"
            "4. NEVER use external knowledge or make assumptions\n"
            "5. If you cannot find the answer in the context, respond EXACTLY: "
            "'I cannot find this information in the document.'\n"
            "6. Be comprehensive - check ALL chunks for relevant information\n"
            "7. Combine information from multiple chunks if needed\n\n"
            "OUTPUT FORMAT:\n"
            "- Give a clear, direct answer\n"
            "- Use document's exact wording when available\n"
            "- Keep it concise but complete\n"
            "- Do NOT mention chunks, pages, or context in your answer"
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]
)
 
 
# --------------------------------------------------
# ANSWER FUNCTION (UNCHANGED)
# --------------------------------------------------
def answer_question(question, retriever):
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
 
    response = llm.invoke(
        prompt.format(context=context, question=question)
    )
 
    answer = response.content.strip()
 
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
        with st.spinner("Searching document..."):
            answer = answer_question(st.session_state.pending_question)
 
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.pending_question = None
    st.rerun()
