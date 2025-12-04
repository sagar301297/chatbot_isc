# load_vectorstore.py
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

# --- IMPORT SETTINGS ---
from config import PERSIST_DIR, EMBED_MODEL, COLLECTION_NAME

UPLOADED_DIR = "./uploaded_pdfs"
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True) # Ensure the DB folder exists

def load_vectorstore(uploaded_files):
    file_paths = []

    # Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOADED_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Load documents
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # Initialize embeddings (using the config)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Check if the store already exists
    # We will initialize the vectorstore object first
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # Add the new texts. LangChain's Chroma handles
    # checking if the store is new or existing.
    vectorstore.add_documents(texts)
    
    print(f"Added {len(texts)} new document chunks to the store.")
    
    return vectorstore