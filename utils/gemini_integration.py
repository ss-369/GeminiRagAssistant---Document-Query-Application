import os
import logging
import json
import google.generativeai as genai
from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)

# Configure Google AI API
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

def self_rag_evaluate_relevance(model, query, context_chunk):
    """
    Self-evaluate the relevance of a context chunk to the query
    
    Args:
        model: The Gemini model instance
        query (str): User's query
        context_chunk (str): Single context chunk to evaluate
        
    Returns:
        float: Relevance score between 0 and 1
    """
    try:
        # Create prompt for relevance evaluation
        evaluation_prompt = f"""
Task: Evaluate the relevance of the provided context to the user question.
Question: {query}
Context: {context_chunk}

Rate the relevance on a scale of 0 to 10, where:
- 0 means completely irrelevant
- 10 means highly relevant and directly answers the question

Output only the numerical score between 0 and 10.
"""
        
        # Generate evaluation response
        response = model.generate_content(evaluation_prompt)
        
        # Extract the score
        score_text = response.text.strip()
        try:
            # Try to extract the integer or float value
            score = float(score_text)
            # Normalize to 0-1 range
            normalized_score = min(max(score / 10, 0), 1)
            return normalized_score
        except ValueError:
            logger.warning(f"Could not parse relevance score from: {score_text}")
            return 0.5
            
    except Exception as e:
        logger.error(f"Error evaluating relevance: {str(e)}")
        return 0.5  # Default to middle relevance on error

def self_rag_filter_context(model, query, context_chunks, threshold=0.6):
    """
    Filter context chunks based on self-evaluated relevance
    
    Args:
        model: The Gemini model instance
        query (str): User's query
        context_chunks (list): List of context chunks
        threshold (float): Minimum relevance score threshold
        
    Returns:
        list: Filtered list of relevant context chunks
    """
    if not context_chunks:
        logger.warning("No context chunks provided to filter")
        return []
        
    filtered_chunks = []
    all_chunks_with_scores = []  # Store all chunks with their scores
    
    for i, chunk in enumerate(context_chunks):
        relevance = self_rag_evaluate_relevance(model, query, chunk)
        all_chunks_with_scores.append((chunk, relevance))
        
        if relevance >= threshold:
            filtered_chunks.append(chunk)
    
    # If no chunks pass the threshold, include the highest scoring one
    if not filtered_chunks and all_chunks_with_scores:
        # Sort by score in descending order and take the first one
        all_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        best_chunk = all_chunks_with_scores[0][0]
        filtered_chunks.append(best_chunk)
        logger.info(f"No chunks passed the threshold, using best available with score {all_chunks_with_scores[0][1]}")
    
    return filtered_chunks

def generate_self_query(model, query, missing_info=None):
    """
    Generate additional queries to retrieve missing information
    
    Args:
        model: The Gemini model instance
        query (str): Original user query
        missing_info (str): Description of missing information
        
    Returns:
        str: Generated follow-up query
    """
    try:
        missing_context = missing_info if missing_info else "the original question needs more specific information"
        
        # Create prompt for query generation
        query_prompt = f"""
Based on this user question: "{query}"

The current information is insufficient because {missing_context}.

Generate a specific follow-up query that would help retrieve the most relevant information to answer the user's question.
Make the query specific, focused and directly related to answering the original question.

Output only the follow-up query text, nothing else:
"""
        
        # Generate query
        response = model.generate_content(query_prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating self-query: {str(e)}")
        return query  # Fall back to original query on error

def self_rag_analysis(model, query, filtered_context):
    """
    Analyze if the filtered context is sufficient to answer the query
    
    Args:
        model: The Gemini model instance
        query (str): User's query
        filtered_context (str): The filtered context text
        
    Returns:
        tuple: (is_sufficient, missing_info)
    """
    # Check if filtered context is empty or too short
    if not filtered_context or len(filtered_context.strip()) < 20:
        return False, "No relevant context was found to answer the question"
        
    try:
        # Create prompt for analysis
        analysis_prompt = f"""
Task: Analyze if the provided context is sufficient to answer the user's question.

User Question: {query}

Available Context:
{filtered_context}

Output a JSON object with the following fields:
- "is_sufficient": boolean (true if context is sufficient, false if not)
- "missing_information": string describing what information is missing (if any)

JSON:
"""
        
        # Generate analysis
        response = model.generate_content(analysis_prompt)
        
        try:
            # Parse the JSON response
            analysis = json.loads(response.text)
            is_sufficient = analysis.get("is_sufficient", False)
            missing_info = analysis.get("missing_information", "")
            return is_sufficient, missing_info
            
        except json.JSONDecodeError:
            logger.warning(f"Could not parse analysis JSON: {response.text}")
            return False, "Could not determine what information is missing"
            
    except Exception as e:
        logger.error(f"Error in self-RAG analysis: {str(e)}")
        return False, "Error during content analysis"

def generate_response(query, context_chunks):
    """
    Generate a response using Gemini 2.0 Flash model with Self-RAG capabilities
    
    Args:
        query (str): User's query
        context_chunks (list): List of relevant context chunks
        
    Returns:
        str: Generated response with self-verification and enhancement
    """
    try:
        # Check if context is empty
        if not context_chunks:
            logger.warning("Empty context chunks provided to generate_response")
            return "I don't have any document information to answer your question. Please upload a document first."
            
        # Initialize Gemini model
        model = GenerativeModel(model_name="models/gemini-2.0-flash")
        
        # Self-RAG Step 1: Filter context by relevance
        filtered_chunks = self_rag_filter_context(model, query, context_chunks)
        
        # If filtering resulted in no chunks (unlikely due to our fallback in filter function)
        if not filtered_chunks:
            logger.warning("No chunks remained after filtering, using original chunks")
            filtered_chunks = context_chunks[:2]  # Use first two chunks as fallback
            
        filtered_context = "\n\n".join(filtered_chunks)
        
        # Self-RAG Step 2: Analyze if context is sufficient
        is_sufficient, missing_info = self_rag_analysis(model, query, filtered_context)
        
        # Self-RAG Step 3: Generate additional queries if needed
        if not is_sufficient and len(filtered_chunks) < len(context_chunks):
            # Try to get more context from original set
            additional_chunks = [c for c in context_chunks if c not in filtered_chunks][:3]
            filtered_chunks.extend(additional_chunks)
            filtered_context = "\n\n".join(filtered_chunks)
            logger.info(f"Added {len(additional_chunks)} additional chunks due to insufficient context")
        
        # Create final prompt with potentially enhanced context
        prompt = f"""
You are a helpful AI assistant that answers questions based on provided context using self-retrieval augmented generation.
Please provide accurate, concise, and helpful answers based ONLY on the information in the context provided.
If the answer is not in the context, say "I don't have enough information to answer that question."

CONTEXT:
{filtered_context}

USER QUESTION:
{query}

ANSWER:
"""
        
        # Generate final response
        response = model.generate_content(prompt)
        
        # Extract and return text
        if hasattr(response, 'text'):
            return response.text
        else:
            # Fallback for different response structure
            return response.parts[0].text if response.parts else "I couldn't generate a response. Please try again."
    
    except Exception as e:
        logger.error(f"Error generating response with Self-RAG: {str(e)}")
        return f"I encountered an error while processing your request: {str(e)}"
