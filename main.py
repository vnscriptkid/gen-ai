import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

# 1. App Setup
st.set_page_config(page_title="Chat with Your PDF", layout="wide")
st.title("📄 Chat with Your PDF using RAG")

# 2. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. File Upload
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# 4. Load and Embed PDF
@st.cache_resource
def load_pdf_and_create_index(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(texts, embeddings)
        return vectorstore.as_retriever()
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

retriever = None
if pdf_file is not None:
    retriever = load_pdf_and_create_index(pdf_file)

# 5. Chat Interface
prompt = st.chat_input("Ask your PDF something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if retriever:
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.invoke(prompt)["result"]
    else:
        response = "Please upload a PDF first."

    st.session_state.messages.append({"role": "assistant", "content": response})

# 6. Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 