# GeminiRagAssistant - Document Query Application

This project is a Python Flask application that allows users to upload documents (PDF, DOCX, TXT) and query them using Retrieval-Augmented Generation (RAG) techniques. It supports two RAG modes: Self-RAG and Agentic RAG, leveraging Gemini models for response generation and context analysis.

## Features

- **Document Upload**: Supports uploading documents in PDF, DOCX, and TXT formats.
- **Text Extraction**: Extracts text content from uploaded documents.
- **Embeddings**: Creates and saves embeddings for document chunks to enable semantic search.
- **Querying**: Allows users to query the uploaded documents.
- **Self-RAG**: Implements Self-Reflective Retrieval-Augmented Generation to filter and evaluate context relevance, improving response quality and identifying missing information.
- **Agentic RAG**: Implements Agentic Retrieval-Augmented Generation, utilizing a RAG agent to refine context retrieval, analyze relevance, and generate more informed responses.
- **Session Management**: Uses sessions to manage user-specific document embeddings.

## Project Structure

- `app.py`: The main Flask application file. Defines routes for:
    - `/`:  Homepage, renders the `index.html` template.
    - `/upload`: Handles document uploads, processes documents, creates embeddings, and saves them.
    - `/query`: Handles user queries, retrieves relevant context using either Self-RAG or Agentic RAG, generates responses using Gemini, and returns responses along with source information and RAG metrics.
- `main.py`: Entry point to run the Flask application.
- `pyproject.toml`: Project configuration file, including dependencies (though currently may not list all).
- `.git/`: Git repository directory (version control).
- `static/`: Directory for static files (e.g., CSS, JavaScript, images for the frontend).
- `templates/`: Directory for HTML templates (currently contains `index.html`).
- `utils/`: Utility modules:
    - `document_processor.py`: Handles document processing tasks:
        - `process_document(file_path)`:  Dispatches document processing based on file type (PDF, DOCX, TXT).
        - `process_pdf(file_path)`: Extracts text from PDF files using `PyPDF2`.
        - `process_docx(file_path)`: Extracts text from DOCX files using `docx2txt`.
        - `process_txt(file_path)`: Extracts text from TXT files, handling UTF-8 and latin-1 encodings.
        - `chunk_text(text, chunk_size=1000, overlap=200)`: Splits text into overlapping chunks for embedding.
    - `embedding.py`: Handles embedding creation and storage:
        - `create_embeddings(text)`: Creates embeddings for text chunks.
        - `save_embeddings(user_id, document_chunks, document_embeddings)`: Saves embeddings associated with a user ID.
        - `load_embeddings(user_id)`: Loads embeddings for a given user ID.
    - `retrieval.py`: Implements context retrieval:
        - `retrieve_context(query, document_chunks, document_embeddings)`: Retrieves relevant document chunks based on a query using embedding similarity.
    - `gemini_integration.py`: Integrates with Gemini models for response generation and Self-RAG analysis:
        - `generate_response(query, context_chunks)`: Generates responses using Gemini based on provided context.
        - `self_rag_filter_context(model, query, context_chunks)`: Filters context chunks using Self-RAG relevance evaluation.
        - `self_rag_analysis(model, query, filtered_context)`: Analyzes if the filtered context is sufficient for answering the query.
    - `agentic_rag.py`: Implements Agentic RAG logic (if present and fully implemented).

## Dependencies

The application requires the following Python libraries:

- `Flask`: For web framework.
- `werkzeug`: For utility functions for web applications.
- `PyPDF2`: For PDF processing.
- `docx2txt`: For DOCX processing.
- `google-generativeai`: For interacting with Gemini models.
- `sentence-transformers`: For creating text embeddings.
- `faiss-cpu`: For efficient similarity search of embeddings.
- `python-dotenv`: For loading environment variables.

You can install these dependencies using pip:

```bash
pip install Flask werkzeug PyPDF2 docx2txt google-generativeai sentence-transformers faiss-cpu python-dotenv
```
*(Note: Please ensure all dependencies are correctly listed in `pyproject.toml` or `requirements.txt` for accurate dependency management.)*


## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r pyproject.toml  # Or use the pip install command above if pyproject.toml is incomplete
    ```

2.  **Set Environment Variables:**
    - Ensure you have set up necessary environment variables, especially `SESSION_SECRET` for Flask sessions and any API keys required for Gemini integration (if applicable).

3.  **Run the Application:**
    ```bash
    python main.py
    ```
    or
    ```bash
    python app.py
    ```

4.  **Access the Application:**
    Open your web browser and go to `http://localhost:5000/`.

## Usage

1.  **Upload a Document**:
    - On the homepage, use the file upload form to upload a document (PDF, DOCX, or TXT).
    - Upon successful upload, the application will process the document and create embeddings.

2.  **Query the Document**:
    - Enter your query in the text input field.
    - Select the RAG mode ("self" or "agent").
    - Submit the query.
    - The application will return a response generated by Gemini, along with sources from the document and RAG metrics depending on the mode.

## RAG Modes

- **Self-RAG**: Focuses on self-reflection to filter and refine the retrieved context, aiming for more relevant and accurate responses. It evaluates the sufficiency of the context and can indicate missing information.
- **Agentic RAG**: Employs a more agent-like approach to RAG, potentially involving iterative retrieval, context analysis, and more complex reasoning to generate responses. (Note: Agentic RAG functionality may be under development or less fully featured than Self-RAG).

---
