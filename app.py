import os
import logging
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

# Import utility functions
from utils.document_processor import process_document
from utils.embedding import create_embeddings, save_embeddings, load_embeddings
# Explicitly import retrieve_context at module level to avoid scope issues
from utils.retrieval import retrieve_context
from utils.gemini_integration import generate_response

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Initialize session data if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file with secure filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process document
            document_text = process_document(file_path)
            
            # Create embeddings
            document_chunks, document_embeddings = create_embeddings(document_text)
            
            # Save embeddings with user's session ID
            user_id = session['user_id']
            save_embeddings(user_id, document_chunks, document_embeddings)
            
            return jsonify({
                'success': True,
                'message': f'File {filename} processed successfully',
                'document_size': len(document_text),
                'chunks': len(document_chunks)
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        user_id = session['user_id']
        
        # Get RAG mode (defaults to "self" for Self-RAG if not specified)
        rag_mode = data.get('rag_mode', 'self')
        
        # Load user's embeddings
        document_chunks, document_embeddings = load_embeddings(user_id)
        
        if document_chunks is None or len(document_chunks) == 0:
            return jsonify({'error': 'No documents found. Please upload a document first.'}), 400
        
        # Retrieve initial relevant context
        initial_context_chunks = retrieve_context(query, document_chunks, document_embeddings)
        
        # Process based on the RAG mode
        if rag_mode == 'agent' or rag_mode == 'agentic':
            # Use Agentic RAG
            from utils.agentic_rag import RAGAgent
            # No need to re-import retrieve_context since it's already imported at the top
            
            # Initialize the RAG Agent
            agent = RAGAgent()
            
            # Execute the full agentic RAG pipeline with the imported retrieve_context function
            # Using the retrieve_context that was imported at the top of the file
            response, agent_metadata = agent.execute_agentic_rag(
                query, 
                document_chunks, 
                document_embeddings, 
                retrieve_context
            )
            
            # Prepare sources with additional metadata from the agent
            sources = []
            used_chunks = agent_metadata.get("final_context_chunks", 0)
            follow_up_queries = agent_metadata.get("follow_up_queries", [])
            
            # Get the most recent context analysis if available
            context_analysis = None
            for reflection in reversed(agent.reflection_history):
                if reflection.get("type") == "context_analysis":
                    context_analysis = reflection.get("analysis")
                    break
            
            # Create source entries with relevance information
            for i, chunk in enumerate(initial_context_chunks):
                # Determine if this chunk was considered relevant in the final analysis
                is_relevant = True
                if context_analysis and "irrelevant_chunks" in context_analysis:
                    is_relevant = (i+1) not in context_analysis.get("irrelevant_chunks", [])
                    
                sources.append({
                    'text': chunk[:150] + '...' if len(chunk) > 150 else chunk,
                    'relevant': is_relevant,
                    'index': i + 1
                })
            
            # Calculate metrics for the UI
            quality_score = None
            if context_analysis and "overall_quality" in context_analysis:
                quality_score = context_analysis["overall_quality"]
                
            # Format the agent metrics for the UI
            agent_rag_metrics = {
                'initial_chunks': len(initial_context_chunks),
                'used_chunks': used_chunks,
                'filtered_out': len(initial_context_chunks) - used_chunks,
                'iterations': agent_metadata.get("iterations", 0),
                'context_quality': quality_score,
                'follow_up_queries': follow_up_queries,
                'missing_info': context_analysis.get("missing_information", []) if context_analysis else []
            }
            
            return jsonify({
                'response': response,
                'sources': sources,
                'agent_rag_metrics': agent_rag_metrics,
                'rag_mode': 'agent'
            })
            
        else:
            # Use Self-RAG (default mode)
            from google.generativeai import GenerativeModel
            from utils.gemini_integration import self_rag_filter_context, self_rag_analysis
            
            # Initialize the Gemini model for self-evaluation
            model = GenerativeModel(model_name="models/gemini-2.0-flash")
            
            # Apply Self-RAG filtering to evaluate relevance
            filtered_chunks = self_rag_filter_context(model, query, initial_context_chunks)
            filtered_context = "\n\n".join(filtered_chunks)
            
            # Analyze if the filtered context is sufficient
            is_sufficient, missing_info = self_rag_analysis(model, query, filtered_context)
            
            # Add more context if needed
            if not is_sufficient and len(filtered_chunks) < len(initial_context_chunks):
                additional_chunks = [c for c in initial_context_chunks if c not in filtered_chunks][:3]
                filtered_chunks.extend(additional_chunks)
                
            # Generate response using Gemini with Self-RAG
            response = generate_response(query, initial_context_chunks)
            
            # Calculate number of chunks filtered out
            filtered_out = len(initial_context_chunks) - len(filtered_chunks)
            
            # Prepare sources with relevance indication
            sources = []
            for i, chunk in enumerate(initial_context_chunks):
                is_filtered = chunk in filtered_chunks
                sources.append({
                    'text': chunk[:150] + '...' if len(chunk) > 150 else chunk,
                    'relevant': is_filtered,
                    'index': i + 1
                })
            
            # Include self-RAG metrics
            self_rag_metrics = {
                'initial_chunks': len(initial_context_chunks),
                'filtered_chunks': len(filtered_chunks),
                'filtered_out': filtered_out,
                'is_sufficient': is_sufficient,
                'missing_info': missing_info if not is_sufficient else ""
            }
            
            return jsonify({
                'response': response,
                'sources': sources,
                'self_rag_metrics': self_rag_metrics,
                'rag_mode': 'self'
            })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
