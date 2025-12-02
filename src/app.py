from csv import DictReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
import logging
from src.gemini_rag_service import GeminiRAGService
from google import genai
from google.genai import types
from src.config import settings
import os

rag_service = GeminiRAGService()
app = FastAPI()
client = genai.Client(api_key=settings.GEMINI_API_KEY)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/rag/upload-files")
async def upload_file(
    file: UploadFile = File(...),
    store_name: str = "store"
):
    file_path = f"./data/{file.filename}"

    try:
        contents = await file.read()

        os.makedirs("./data", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)

        store_id = rag_service.upload_file(
            file_path=file_path, 
            file_name=file.filename or "unknown_filename")

        os.remove(file_path)

        return {
            "status": "success",
            "file_name": file.filename,
            "store_id": store_id
        }

    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
class ChatRequest(BaseModel):
    query: str
    store_id: str

@app.post("/rag/chat")
async def chat_with_store(request: ChatRequest):
    response = rag_service.response_document(request.query, request.store_id)
    return response