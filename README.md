# File Search Endpoint (Gemini RAG)

This project is a high-performance **Retrieval-Augmented Generation (RAG)** API built with **FastAPI** and **Google Gemini (GenAI SDK)**. It enables users to upload PDF documents and perform semantic searches or chat with the content using Google's advanced Gemini 2.5 Flash model.

## Features

- **Document Upload:** Upload PDF files securely to Google's GenAI File Store.
- **Smart Duplicate Handling:** Automatically detects if a file already exists to prevent duplicate uploads, reusing the existing storage for efficiency.
- **Context-Aware Chat:** Ask questions about your documents and get precise answers based *only* on the provided context.
- **Citations:** Returns specific source citations (grounding chunks) used to generate the answer.
- **FastAPI Powered:** Async architecture for high performance.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NguyenHoangXuanSon/markdown_table_parser.git](https://github.com/NguyenHoangXuanSon/markdown_table_parser.git)
   cd markdown_table_parser
2. **Install dependencies: Using uv to sync dependencies from pyproject.toml / uv.lock:**
    ```bash
    uv sync
3. **Configuration.**
Create a .env file in the root directory of the project.
   ```bash
GEMINI_API_KEY=your_actual_api_key_here
4. **Running the Application**
   ```bash 
uv run uvicorn src.app:app --reload





