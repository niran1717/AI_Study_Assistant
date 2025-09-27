# ingestion_logic.py

import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# We remove CHROMA_DB_PATH since the database will be IN-MEMORY
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
TEMP_UPLOADS_DIR = "temp_uploads" # Still need this for file processing


def create_new_vector_store():
    """Creates and returns a brand new, empty, in-memory ChromaDB instance."""
    print("Creating new IN-MEMORY vector store for the session.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # By omitting persist_directory, the database is held only in memory
    vector_store = Chroma(
        embedding_function=embeddings
    )
    return vector_store


def ingest_file(vector_store, uploaded_file):
    """Processes an uploaded file and adds its content to the given vector store."""
    print(f"Processing uploaded file: {uploaded_file.name}")
    
    # Save the uploaded file to a temporary location (required for Unstructured)
    temp_file_path = os.path.join(TEMP_UPLOADS_DIR, uploaded_file.name)
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the document
    loader = UnstructuredPDFLoader(temp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        os.remove(temp_file_path)
        print("No content extracted from the file.")
        return 0
    
    # Add chunks directly to the provided vector store (in-memory)
    vector_store.add_documents(chunks)
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    print(f"Successfully added {len(chunks)} chunks from {uploaded_file.name}.")
    return len(chunks)