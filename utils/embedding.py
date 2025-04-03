import os
import pickle
import logging
import numpy as np
import google.generativeai as genai
from .document_processor import chunk_text

logger = logging.getLogger(__name__)

# Configure Google AI API
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Storage for embeddings (in-memory for simplicity)
# In a production environment, use a database instead
EMBEDDINGS_STORAGE = {}

def create_embeddings(document_text):
    """
    Create embeddings for a document by:
    1. Chunking the document text
    2. Generating embeddings for each chunk
    
    Args:
        document_text (str): The document text to embed
        
    Returns:
        tuple: (list of document chunks, numpy array of embeddings)
    """
    try:
        # Chunk the document
        document_chunks = chunk_text(document_text)
        
        # Generate embeddings for each chunk
        document_embeddings = []
        
        for chunk in document_chunks:
            # Use the embed_content function from the genai module directly
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            document_embeddings.append(result['embedding'])
        
        # Convert to numpy array for faster processing
        document_embeddings = np.array(document_embeddings)
        
        return document_chunks, document_embeddings
    
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def save_embeddings(user_id, document_chunks, document_embeddings):
    """
    Save embeddings for a user
    
    Args:
        user_id (str): User ID to associate with the embeddings
        document_chunks (list): List of document chunks
        document_embeddings (numpy.ndarray): Array of embeddings
    """
    try:
        EMBEDDINGS_STORAGE[user_id] = {
            'chunks': document_chunks,
            'embeddings': document_embeddings
        }
        logger.info(f"Saved embeddings for user: {user_id}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        raise

def load_embeddings(user_id):
    """
    Load embeddings for a user
    
    Args:
        user_id (str): User ID to retrieve embeddings for
        
    Returns:
        tuple: (list of document chunks, numpy array of embeddings)
                or (None, None) if not found
    """
    try:
        if user_id in EMBEDDINGS_STORAGE:
            data = EMBEDDINGS_STORAGE[user_id]
            return data['chunks'], data['embeddings']
        else:
            logger.warning(f"No embeddings found for user: {user_id}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise
