# -----------------------------------
# IMPORTS
# -----------------------------------
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

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="PDF Q&A Bot",
    layout="wide"
)

st.title("PDF Question Answering Bot")
st.caption("Answers are based only on the uploaded document")

# -----------------------------------
# LOAD ENV
# -----------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "new-bot-gte-base"

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys.")
    st.stop()

# -----------------------------------
# PINECONE CONNECT
# -----------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Clear chat"):
        st.session_state.clear()
        st.rerun()

if not uploaded_pdf:
    st.stop()

pdf_namespace = uploaded_pdf.name.replace(" ", "_").lower()

# -----------------------------------
# VECTORSTORE
# -----------------------------------
@st.cache_resource(show_spinner=True)
def load_vectorstore(uploaded_pdf, namespace):

    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-base"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # increased for better semantic grouping
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )

    # embed in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i:i+batch_size])

    return vectorstore


vectorstore = load_vectorstore(uploaded_pdf, pdf_namespace)

# 🔥 Use similarity search instead of MMR
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.0,
        max_new_tokens=400,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
)

# -----------------------------------
# PROMPT (LESS RESTRICTIVE)
# -----------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions based only on the provided document context. "
            "Use the information clearly present in the context. "
            "If the answer is not present, say: "
            "'I cannot find this information in the document.'"
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]
)

# -----------------------------------
# ANSWER FUNCTION
# -----------------------------------
def answer_question(question):

    docs = retriever.invoke(question)

    if not docs:
        return "I cannot find this information in the document."

    context = "\n\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    answer = response.content.strip()

    if "cannot find" in answer.lower():
        return "I cannot find this information in the document."

    return answer

# -----------------------------------
# CHAT UI
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = answer_question(query)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
