# app.py

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from file_ingest import create_new_vector_store, ingest_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Session State Initialization ---
# The kb_is_ready flag will manage the UI flow
if "kb_is_ready" not in st.session_state:
    st.session_state.kb_is_ready = False 

# The actual database object is stored in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_new_vector_store()
    
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- RAG Setup (Load the persisted database) ---
@st.cache_resource
def load_rag_chain():
    """Sets up the RAG chain using the vector store from session state."""
    
    with st.spinner("⏳ Loading RAG resources..."):
        # Retrieve the IN-MEMORY vector store from session state
        vector_store = st.session_state.vector_store
        
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
    st.header("Manage Knowledge Base (Resets per Session)")
    uploaded_file = st.file_uploader(
        "Upload a new PDF to begin your session", 
        type="pdf",
        accept_multiple_files=False,
    )
    
    if uploaded_file and st.button("Start Session"):
        # --- FIX: Ensure we are dealing with a single file object ---
        file_to_process = uploaded_file[0] if isinstance(uploaded_file, list) else uploaded_file
        
        # Get the existing (empty or populated) vector store from session state
        current_vector_store = st.session_state.vector_store

        with st.spinner(f"Processing {file_to_process.name}..."):
            try:
                ingest_file(current_vector_store, file_to_process)
                st.session_state.kb_is_ready = True
                st.success(f"Successfully processed {file_to_process.name}. Chat is ready!")
                # No need to clear cache, just rerun to update UI
                st.rerun() 
            except Exception as e:
                st.error(f"Failed to process file: {e}")
                print(f"File Processing Error: {e}") # Print full error to console

# Main content
if st.session_state.kb_is_ready:
    qa_chain = load_rag_chain()
    
    # Display chat messages from history
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
    st.info("Your session is currently empty. Please upload a PDF to begin your personalized study session.")
    st.chat_input("Please upload a document first...", disabled=True)