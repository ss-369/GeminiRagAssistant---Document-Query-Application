import logging
import numpy as np
import google.generativeai as genai

logger = logging.getLogger(__name__)

def cosine_similarity(query_embedding, document_embeddings):
    """
    Calculate cosine similarity between a query embedding and document embeddings
    
    Args:
        query_embedding (list): Query embedding vector
        document_embeddings (numpy.ndarray): Document embeddings matrix
        
    Returns:
        numpy.ndarray: Array of similarity scores
    """
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Normalize the document embeddings
    norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    normalized_embeddings = document_embeddings / norms
    
    # Calculate dot product (cosine similarity for normalized vectors)
    similarities = np.dot(normalized_embeddings, query_embedding)
    
    return similarities

def retrieve_context(query, document_chunks, document_embeddings, top_k=5):
    """
    Retrieve the most relevant document chunks for a query
    
    Args:
        query (str): Query text
        document_chunks (list): List of document chunks
        document_embeddings (numpy.ndarray): Array of document embeddings
        top_k (int): Number of top chunks to retrieve
        
    Returns:
        list: List of the most relevant document chunks
    """
    try:
        if not document_chunks or document_chunks is None or len(document_chunks) == 0:
            logger.warning("No document chunks available for retrieval")
            return []
        
        # Generate embedding for the query using genai.embed_content
        query_result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_result['embedding']
        
        # Calculate similarities between query and all document chunks
        similarities = cosine_similarity(query_embedding, document_embeddings)
        
        # Get indices of top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return the top_k most similar chunks
        top_chunks = [document_chunks[i] for i in top_indices]
        
        return top_chunks
    
    except Exception as e:
        logger.error(f"Error in context retrieval: {str(e)}")
        raise
