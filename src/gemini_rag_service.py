import os
from typing import List, Dict, Union, Any, Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError
import logging
from src.config import settings 
import logging
import time
import re
from src.prompt import FILE_SEARCH_INSTRUCTION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiRAGService:
    """
    Service to interact with Gemini API for RAG tasks
    """
    def __init__(self):
        self.client =genai.Client(api_key=settings.GEMINI_API_KEY)

    def create_store(self):
        file_search_store = self.client.file_search_stores.create(config={'display_name': 'your-fileSearchStore-name'})

        if not file_search_store.name:
            raise ValueError("file_search_store.name is None")
        
        return file_search_store.name

    def upload_file(self, 
                    file_path: str, 
                    file_name: str):
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        safe_file_name = re.sub(r'[^a-z0-9\-]', '-', file_name.lower()).strip('-')
        file_resource_name = f"files/{safe_file_name}" 
        
        try:
            self.client.files.upload(
                file=file_path,
                config={'name': safe_file_name, 'display_name': file_name}
            )
            logging.info(f"Uploaded new file: {file_name}")

        except Exception as e:
            if "ALREADY_EXISTS" in str(e) or "409" in str(e):
                logging.info(f"File {file_name} already exists in Google Cloud. Switching to retrieval mode.")
            else:
                raise e
        existing_stores = self.client.file_search_stores.list()
        
        target_store = None
        for store in existing_stores:
            if store.display_name == file_name:
                target_store = store
                break
        
        if target_store:
            logging.info(f"Found existing store for {file_name}: {target_store.name}")
            return target_store.name 
        
        else:
            logging.info(f"Creating new store for {file_name}")
            new_store = self.client.file_search_stores.create(config={'display_name': file_name})
            
            operation = self.client.file_search_stores.import_file(
                file_search_store_name=new_store.name if new_store.name else "",
                file_name=file_resource_name, 
            )
            
            while not operation.done:
                time.sleep(2)
                operation = self.client.operations.get(operation)
            
            logging.info(f"Created store {new_store.name} and imported file.")
            return new_store.name
    
    def response_document(self, question: str, store_name) -> Tuple[Optional[str], Optional[List[Any]]]:
        message = FILE_SEARCH_INSTRUCTION + f'Use the following question to provide an answer based on the retrieved documents: "{question}"'

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=message,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ]
            )
        )
        response_text = response.text
        citations = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                citations = metadata.grounding_chunks

        return response_text, citations