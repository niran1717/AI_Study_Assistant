# ingestion_logic.py

import os
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

STARTUP_DOCS_PATH = "resources"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"



def get_vector_store():
    """Initializes the vector store, either by loading an existing one or creating a new one."""
    print("Initializing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_PATH):
        # Load the existing database
        print("Existing database found. Loading from disk.")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    else:
        # Create a new database from startup documents if none exists
        print("No database found. Creating a new one from startup documents.")
        if not os.path.exists(STARTUP_DOCS_PATH):
            raise FileNotFoundError(f"Startup documents not found at '{STARTUP_DOCS_PATH}'.")
            
        loader = DirectoryLoader(STARTUP_DOCS_PATH, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print("Successfully created database from startup documents.")
        
    return vector_store


def ingest_file(vector_store, uploaded_file):
    """Processes an uploaded file and adds its content to the given vector store."""
    print(f"Processing uploaded file: {uploaded_file.name}")
    
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
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

    # Add chunks directly to the provided vector store
    vector_store.add_documents(chunks)
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    print(f"Successfully added {len(chunks)} chunks from {uploaded_file.name}.")
    return len(chunks)