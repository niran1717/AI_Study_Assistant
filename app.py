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
if "kb_is_ready" not in st.session_state:
    st.session_state.kb_is_ready = False 
    
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = [] # NEW: To store names of files for display

if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_new_vector_store()
    
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- RAG Setup (same as before) ---
@st.cache_resource
def load_rag_chain():
    # ... (same function body) ...
    with st.spinner("⏳ Loading RAG resources..."):
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
    
    # CHANGE 1: Use accept_multiple_files=True to allow continuous uploads
    uploaded_files = st.file_uploader(
        "Upload new PDFs to add to your session", 
        type="pdf",
        accept_multiple_files=True, # CHANGED
    )
    
    if uploaded_files and st.button("Add Files to Session"):
        if not uploaded_files:
            st.error("Please select at least one file.")
        else:
            current_vector_store = st.session_state.vector_store
            
            # Use a progress bar for multiple files
            progress_text = "Processing uploaded files..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, file in enumerate(uploaded_files):
                my_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {file.name}...")
                try:
                    ingest_file(current_vector_store, file)
                    st.session_state.kb_is_ready = True
                    if file.name not in st.session_state.uploaded_file_names:
                        st.session_state.uploaded_file_names.append(file.name)
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")
                    print(f"File Processing Error: {e}")
            
            my_bar.progress(1.0, text="All files processed!")
            st.success("All files successfully added. Chat is ready!")
            st.rerun() 

    st.subheader("Current Documents")
    if st.session_state.uploaded_file_names:
        # CHANGE 2: Display all file names using a list
        st.markdown(
            "| **File Name** |\n| :--- |\n" +
            "\n".join([f"| {name} |" for name in st.session_state.uploaded_file_names])
        )
    else:
        st.info("No documents uploaded yet.")


# Main content logic (same as before)
if st.session_state.kb_is_ready:
    qa_chain = load_rag_chain()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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