import sys, pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
import os

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger

from config import PERSIST_DIR, EMBED_MODEL, COLLECTION_NAME

# Import these at startup
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

client = chromadb.PersistentClient(path=PERSIST_DIR)
os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI(title='chatbot')

# Initialize embeddings and vectorstore at startup
embeddings = None
vectorstore = None
chain = None

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstore, chain
    logger.info("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    logger.info("Embeddings model loaded")
    
    logger.info("Initializing vectorstore...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    logger.info("Vectorstore initialized")
    
    # Initialize chain (will be recreated after uploads)
    chain = get_llm_chain(vectorstore)
    logger.info("LLM chain ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Exception")
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/")
def health_check():
    return {"status": "healthy", "vectorstore_initialized": vectorstore is not None}

@app.post("/upload_pdfs")
async def uploaded_pdfs(files: List[UploadFile] = File(...)):
    global vectorstore, chain
    try:
        logger.info(f"Received {len(files)} files")
        load_vectorstore(files)
        
        # Reinitialize vectorstore to pick up new documents
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        chain = get_llm_chain(vectorstore)
        
        logger.info("Documents added to chroma")
        return {"message": "Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during file upload")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global chain
    try:
        logger.info(f"User query: {question}")
        
        if chain is None:
            logger.error("Chain not initialized")
            return JSONResponse(
                status_code=500, 
                content={"error": "System not ready. Please upload documents first."}
            )
        
        result = query_chain(chain, question)
        logger.info("Query successful")
        return result
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset")
def reset_chat():
    global vectorstore, chain
    try:
        logger.info(f"Resetting collection: {COLLECTION_NAME}")
        
        # Try to delete the collection if it exists
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} deleted.")
        except Exception as e:
            logger.warning(f"Collection might not exist: {e}")
        
        # Create a new, empty collection with the same name
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection {COLLECTION_NAME} created.")
        
        # Reinitialize vectorstore and chain
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        chain = get_llm_chain(vectorstore)
        
        # Optional: Clear uploaded PDFs folder
        import shutil
        UPLOADED_DIR = "./uploaded_pdfs"
        if os.path.exists(UPLOADED_DIR):
            shutil.rmtree(UPLOADED_DIR)
            os.makedirs(UPLOADED_DIR, exist_ok=True)
            logger.info("Cleared uploaded PDFs folder")
        
        return {"message": "Chat context cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error resetting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting chat: {str(e)}")