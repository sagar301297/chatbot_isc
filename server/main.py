import sys, pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
import os

from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger

from config import PERSIST_DIR, EMBED_MODEL, COLLECTION_NAME

client = chromadb.PersistentClient(path=PERSIST_DIR)
os.makedirs(PERSIST_DIR, exist_ok=True) # Ensure the folder exists

app=FastAPI(title='chatbot')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Exception")
        return JSONResponse(status_code=500,content={"error":str(exc)})

@app.post("/upload_pdfs")
async def uploaded_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"received {len(files)} files")
        load_vectorstore(files)
        logger.info("documenrts added to chroma")
        return {"messafe":"files proceed and vectorstore updated"}
    except Exception as e:
        logger.exception("error during file upload")
        return JSONResponse(status_code=500,content={"error":str(e)})

@app.post("/ask")
async def ask_question(question:str=Form(...)):
    try:
        logger.info(f"user query:{question}")
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from modules.load_vectorstore import PERSIST_DIR

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL
        )

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME 
        )

        chain=get_llm_chain(vectorstore)
        result=query_chain(chain,question)
        logger.info("query successfulyl")
        return result
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return JSONResponse(status_code=500,content={"error":str(e)})


@app.post("/reset")
def reset_chat():
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