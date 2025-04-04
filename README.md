# Gemini RAG Assistant

A powerful document question-answering application featuring advanced Retrieval-Augmented Generation (RAG) capabilities with Google's Gemini 2.0 Flash model.

![Gemini RAG Assistant](https://img.shields.io/badge/Gemini-RAG%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Flask](https://img.shields.io/badge/Flask-3.1.0+-orange)

## Features

### Self-RAG Capabilities
- **Relevance Evaluation**: Automatically evaluates how relevant each context chunk is to the query
- **Context Filtering**: Removes less relevant information to improve response quality
- **Sufficiency Analysis**: Determines if the retrieved context is sufficient to answer the query
- **Adaptive Retrieval**: Retrieves additional context when needed

### Agentic RAG Capabilities
- **Query Reformulation**: Transforms user queries for more effective retrieval
- **Iterative Analysis**: Multiple rounds of context analysis and improvement
- **Follow-up Query Generation**: Generates specific queries to fill information gaps
- **Context Synthesis**: Creates optimized context by combining and reorganizing information

### Document Processing
- Supports PDF, DOCX, and TXT documents
- Automatically chunks documents for improved retrieval
- Semantic search using Google's embedding model

## Architecture

- **Flask Web Application**: Lightweight web interface with responsive design
- **Modular Components**: Separate modules for document processing, embedding, retrieval, and generation
- **In-Memory Storage**: Session-based storage for document embeddings
- **Gemini 2.0 Flash**: Leverages Google's latest LLM for intelligent RAG operations

## Project Structure

- `app.py`: The main Flask application file. Defines routes for:
    - `/`:  Homepage, renders the `index.html` template.
    - `/upload`: Handles document uploads, processes documents, creates embeddings, and saves them.
    - `/query`: Handles user queries, retrieves relevant context using either Self-RAG or Agentic RAG, generates responses using Gemini, and returns responses along with source information and RAG metrics.
- `main.py`: Entry point to run the Flask application.
- `pyproject.toml`: Project configuration file, including dependencies.
- `/utils`: Core RAG functionality
  - `agentic_rag.py`: Autonomous RAG agent implementation
  - `document_processor.py`: Document parsing and chunking
  - `embedding.py`: Document and query embedding functions
  - `gemini_integration.py`: Integration with Gemini models
  - `retrieval.py`: Semantic search functionality
- `/static`: Frontend assets
  - `/css`: Stylesheets
  - `/js`: JavaScript files
- `/templates`: HTML templates

## Getting Started

### Prerequisites

- Python 3.11+
- A valid Google API key for Gemini API access

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GeminiRagAssistant.git
   cd GeminiRagAssistant
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key_here"
   export SESSION_SECRET="a_secure_random_string"
   ```
   On Windows:
   ```
   set GOOGLE_API_KEY=your_google_api_key_here
   set SESSION_SECRET=a_secure_random_string
   ```

### Dependencies

The application requires the following Python libraries:

- Flask: Web framework
- Google Generative AI: Gemini API access
- PyPDF2: PDF processing
- docx2txt: DOCX processing
- NumPy: Numerical operations
- werkzeug: For utility functions for web applications

You can install these dependencies using pip:

```bash
pip install Flask google-generativeai PyPDF2 docx2txt numpy werkzeug
```

### Running the Application

Run the application with:
```bash
python main.py
```

The application will be available at http://localhost:5000

## Usage Guide

1. **Upload a Document**:
   - Click on the "Upload Documents" section
   - Select a document (PDF, DOCX, or TXT format)
   - Wait for processing (document will be chunked and embedded)

2. **Ask Questions**:
   - Type your question in the query box
   - Select your preferred RAG mode:
     - **Self-RAG**: Faster with real-time relevance filtering
     - **Agentic RAG**: More thorough with iterative improvements
   - Click "Ask" and wait for the response

3. **View the Response**:
   - The answer will be displayed in the response section
   - You can see which sources were used and their relevance
   - For Agentic RAG, you'll see additional metrics like context quality and follow-up queries

## How It Works

### Self-RAG Process
1. User uploads document and asks a question
2. System retrieves initial context chunks based on semantic similarity
3. Each chunk is evaluated for relevance to the query
4. Low-relevance chunks are filtered out
5. System analyzes if the filtered context is sufficient
6. If needed, additional context is retrieved
7. Final response is generated using the optimized context

### Agentic RAG Process
1. User uploads document and asks a question
2. System reformulates the query to improve retrieval
3. Initial context chunks are retrieved
4. System analyzes context quality and identifies gaps
5. Context chunks are prioritized by relevance
6. System generates follow-up queries to fill gaps
7. Additional context is retrieved using follow-up queries
8. Context is synthesized into optimized form
9. Final response is generated with detailed process metrics

## License

[MIT License](LICENSE)

## Acknowledgements

- Built with Google's Gemini 2.0 Flash model
- Inspired by research on Self-RAG and Agentic RAG approaches
