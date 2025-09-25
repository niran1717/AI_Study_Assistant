# app.py

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from file_ingest import get_vector_store, ingest_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Session State Initialization ---
if "kb_is_ready" not in st.session_state:
    st.session_state.kb_is_ready = os.path.exists(CHROMA_DB_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RAG Setup (Load the persisted database) ---
@st.cache_resource
def load_rag_chain():
    """Loads the ChromaDB vector store and sets up the RAG chain."""
    
    with st.spinner("⏳ Loading RAG resources..."):
        # This will either load the existing DB or create a new one
        vector_store = get_vector_store()
        
        try:
            groq_llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL
            )
        except Exception as e:
            st.error(f"Groq API Error: {e}. Please ensure your `GROQ_API_KEY` is set correctly.")
            st.stop()

        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ RAG chain loaded.")
        return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Personalized Study Assistant", page_icon="📚")
st.title("📚 Personalized Study Assistant")

# Sidebar for file upload
with st.sidebar:
    st.header("Manage Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload a new PDF to your knowledge base", 
        type="pdf",
        accept_multiple_files=True,
    )
    
if uploaded_file and st.button("Add to Knowledge Base"):
    current_vector_store = load_rag_chain().retriever.vectorstore

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            ingest_file(current_vector_store, uploaded_file)
            st.session_state.kb_is_ready = True
            st.success(f"Successfully processed {uploaded_file.name}.")
            load_rag_chain.clear()
            st.rerun()
        except Exception as e:
            # Change this line
            st.error(f"Failed to process file: {e}")
            # to this, to see the full error details
            print("--- An error occurred during file processing ---")
            print(e)
            print("---------------------------------------------")
            st.sidebar.error(f"Failed to process file. Check your terminal for details.")

# Main content
if st.session_state.kb_is_ready:
    qa_chain = load_rag_chain()
    # The rest of your existing chat UI logic goes here
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Searching and generating response..."):
            result = qa_chain({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Show Sources"):
                for doc in sources:
                    st.markdown(f"**Source:** `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`")
                    st.code(doc.page_content[:200] + "...")
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("The knowledge base is empty. Please upload a PDF to begin.")
    st.chat_input("Please upload a document first...", disabled=True)