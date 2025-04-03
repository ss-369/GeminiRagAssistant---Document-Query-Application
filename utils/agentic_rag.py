import os
import logging
import json
from typing import List, Dict, Tuple, Any, Optional
import google.generativeai as genai
from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)

# Configure Google AI API
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

class RAGAgent:
    """
    An agentic RAG system that can autonomously improve retrieval and generation
    through iterative reflection and query reformulation.
    """
    
    def __init__(self, model_name="models/gemini-2.0-flash"):
        """Initialize the RAG Agent with a specific model."""
        self.model = GenerativeModel(model_name=model_name)
        self.reflection_history = []
        self.action_history = []
        self.max_iterations = 3
    
    def formulate_search_query(self, original_query: str) -> str:
        """
        Transform the original query into a more effective search query
        for document retrieval.
        
        Args:
            original_query (str): The user's original query
            
        Returns:
            str: Reformulated search query
        """
        prompt = f"""
        Your task is to reformulate this user query into a more effective search query.
        Make the query more specific and add key terms that would help in retrieving relevant documents.
        Extract important entities and concepts that should be searched for.
        
        Original query: "{original_query}"
        
        Output only the reformulated search query, nothing else:
        """
        
        try:
            response = self.model.generate_content(prompt)
            reformulated_query = response.text.strip()
            
            # Track this action in history
            self.action_history.append({
                "action": "query_reformulation",
                "original": original_query,
                "reformulated": reformulated_query
            })
            
            logger.info(f"Reformulated query: '{original_query}' -> '{reformulated_query}'")
            return reformulated_query
        except Exception as e:
            logger.error(f"Error reformulating query: {str(e)}")
            return original_query
    
    def analyze_retrieved_context(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze retrieved context chunks to determine quality and relevance.
        
        Args:
            query (str): The user's query
            context_chunks (List[str]): List of retrieved context chunks
            
        Returns:
            Dict: Analysis results including quality scores and missing information
        """
        context_text = "\n\n---\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""
        Analyze the quality and relevance of these retrieved context chunks for answering the user's query.
        
        User Query: {query}
        
        Retrieved Context:
        {context_text}
        
        Provide a JSON with the following:
        1. overall_quality: rating from 0-10 on how well the context addresses the query
        2. missing_information: list of key information that's missing but needed
        3. irrelevant_chunks: list of chunk numbers that are not relevant to the query
        4. most_relevant_chunks: list of chunk numbers most relevant to the query
        5. knowledge_gaps: concepts mentioned but not properly explained
        
        JSON:
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract the JSON part of the response
            json_text = response.text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(json_text)
            
            # Save this reflection
            self.reflection_history.append({
                "type": "context_analysis",
                "query": query,
                "analysis": analysis
            })
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing context: {str(e)}")
            return {
                "overall_quality": 5,
                "missing_information": ["Unable to analyze context"],
                "irrelevant_chunks": [],
                "most_relevant_chunks": list(range(len(context_chunks))),
                "knowledge_gaps": []
            }
    
    def generate_follow_up_queries(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up queries to fill knowledge gaps based on analysis.
        
        Args:
            query (str): The original query
            analysis (Dict): Context analysis results
            
        Returns:
            List[str]: List of follow-up queries
        """
        # Construct prompt from missing information and knowledge gaps
        missing_info = analysis.get("missing_information", [])
        knowledge_gaps = analysis.get("knowledge_gaps", [])
        
        missing_text = "\n".join([f"- {item}" for item in missing_info])
        gaps_text = "\n".join([f"- {item}" for item in knowledge_gaps])
        
        prompt = f"""
        Based on the original query and the identified gaps in information, generate 1-3 specific follow-up queries
        that would help retrieve additional relevant information.
        
        Original query: "{query}"
        
        Missing information:
        {missing_text}
        
        Knowledge gaps:
        {gaps_text}
        
        Generate specific, targeted follow-up queries that would help fill these gaps.
        Output only the list of follow-up queries, one per line:
        """
        
        try:
            response = self.model.generate_content(prompt)
            follow_up_queries = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
            
            # Track this action
            self.action_history.append({
                "action": "follow_up_queries",
                "original_query": query,
                "follow_up_queries": follow_up_queries
            })
            
            return follow_up_queries[:3]  # Limit to 3 follow-up queries
        except Exception as e:
            logger.error(f"Error generating follow-up queries: {str(e)}")
            return []
    
    def prioritize_chunks(self, chunks: List[str], analysis: Dict[str, Any]) -> List[str]:
        """
        Reorder chunks based on relevance analysis.
        
        Args:
            chunks (List[str]): Original context chunks
            analysis (Dict): Context analysis with relevance scores
            
        Returns:
            List[str]: Reordered chunks with most relevant first
        """
        try:
            # Get the most relevant chunk indices (1-indexed in the analysis)
            most_relevant = [i-1 for i in analysis.get("most_relevant_chunks", [])]
            
            # Get the irrelevant chunk indices (1-indexed in the analysis)
            irrelevant = [i-1 for i in analysis.get("irrelevant_chunks", [])]
            
            # Create a new ordering: most relevant first, then others, exclude irrelevant
            prioritized_indices = []
            
            # First add the most relevant chunks
            for idx in most_relevant:
                if 0 <= idx < len(chunks) and idx not in prioritized_indices:
                    prioritized_indices.append(idx)
            
            # Then add any remaining chunks that aren't irrelevant
            for idx in range(len(chunks)):
                if idx not in prioritized_indices and idx not in irrelevant:
                    prioritized_indices.append(idx)
            
            # Reorder the chunks
            prioritized_chunks = [chunks[idx] for idx in prioritized_indices]
            
            # Track this action
            self.action_history.append({
                "action": "chunk_prioritization",
                "original_count": len(chunks),
                "prioritized_count": len(prioritized_chunks),
                "removed_count": len(chunks) - len(prioritized_chunks)
            })
            
            return prioritized_chunks
        except Exception as e:
            logger.error(f"Error prioritizing chunks: {str(e)}")
            return chunks
    
    def synthesize_context(self, query: str, chunks: List[str]) -> str:
        """
        Create a synthesized version of the context optimized for the query.
        
        Args:
            query (str): The user query
            chunks (List[str]): Relevant context chunks
            
        Returns:
            str: Synthesized context
        """
        if not chunks:
            return ""
            
        chunks_text = "\n\n---\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        
        prompt = f"""
        Your task is to synthesize these information chunks into a coherent, comprehensive context
        that directly addresses the user's query.
        
        User Query: {query}
        
        Information Chunks:
        {chunks_text}
        
        Create a synthesized context that:
        1. Combines related information from different chunks
        2. Removes redundant information
        3. Organizes the information logically
        4. Prioritizes information most relevant to the query
        5. Maintains all factual details without adding new information
        
        Synthesized Context:
        """
        
        try:
            response = self.model.generate_content(prompt)
            synthesized = response.text.strip()
            
            # Track this action
            self.action_history.append({
                "action": "context_synthesis",
                "input_chunks": len(chunks),
                "synthesized_length": len(synthesized)
            })
            
            return synthesized
        except Exception as e:
            logger.error(f"Error synthesizing context: {str(e)}")
            return "\n\n".join(chunks)  # Fall back to original chunks
    
    def execute_agentic_rag(self, original_query: str, document_chunks: List[str], 
                           document_embeddings: Any, retrieval_func: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Run the full agentic RAG pipeline with iterative improvement.
        
        Args:
            original_query (str): The user's original query
            document_chunks (List[str]): All available document chunks
            document_embeddings (Any): Embeddings for the document chunks
            retrieval_func (callable): Function to retrieve context chunks
            
        Returns:
            Tuple[str, Dict]: Generated response and process metadata
        """
        # Reset history for this execution
        self.reflection_history = []
        self.action_history = []
        
        # Step 1: Query reformulation
        search_query = self.formulate_search_query(original_query)
        
        # Step 2: Initial retrieval
        retrieved_chunks = retrieval_func(search_query, document_chunks, document_embeddings)
        
        # Iterative improvement loop
        current_query = search_query
        current_chunks = retrieved_chunks
        
        for iteration in range(self.max_iterations):
            # Step 3: Analyze retrieved context
            analysis = self.analyze_retrieved_context(original_query, current_chunks)
            
            # Record the quality score
            quality_score = analysis.get("overall_quality", 0)
            logger.info(f"Iteration {iteration+1} quality score: {quality_score}/10")
            
            # Step 4: Check if quality is good enough to stop
            if quality_score >= 8:
                logger.info(f"Context quality sufficient ({quality_score}/10), stopping iterations")
                break
                
            # Step 5: Prioritize and filter chunks
            current_chunks = self.prioritize_chunks(current_chunks, analysis)
            
            # Step 6: Generate follow-up queries if needed
            follow_up_queries = self.generate_follow_up_queries(original_query, analysis)
            
            # Step 7: Retrieve additional context with follow-up queries
            additional_chunks = []
            for query in follow_up_queries:
                new_chunks = retrieval_func(query, document_chunks, document_embeddings)
                additional_chunks.extend([c for c in new_chunks if c not in current_chunks])
            
            # Add new unique chunks
            if additional_chunks:
                logger.info(f"Added {len(additional_chunks)} additional chunks from follow-up queries")
                current_chunks.extend(additional_chunks[:3])  # Limit to top 3 new chunks
        
        # Step 8: Final context synthesis
        synthesized_context = self.synthesize_context(original_query, current_chunks)
        
        # Step 9: Generate final response
        final_prompt = f"""
        You are a helpful AI assistant that answers questions based on the provided context.
        Answer the user's question accurately and completely based on the context provided.
        If the answer is not in the context, say "I don't have enough information to answer that question."
        
        Context:
        {synthesized_context}
        
        User Question: {original_query}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(final_prompt)
            answer = response.text.strip()
            
            # Get the most recent analysis if available, or create a default one
            latest_analysis = {"overall_quality": 5}
            for entry in reversed(self.reflection_history):
                if entry.get("type") == "context_analysis" and "analysis" in entry:
                    latest_analysis = entry["analysis"]
                    break
            
            # Final metadata to return
            metadata = {
                "original_query": original_query,
                "search_query": search_query,
                "iterations": len(self.reflection_history),
                "context_quality": latest_analysis.get("overall_quality", 5),
                "action_history": self.action_history,
                "final_context_chunks": len(current_chunks),
                "follow_up_queries": [],  # Initialize with empty list
                "synthesized_context_used": True
            }
            
            # Add follow-up queries from action history if they exist
            for action in reversed(self.action_history):
                if action.get("action") == "follow_up_queries" and "follow_up_queries" in action:
                    metadata["follow_up_queries"] = action["follow_up_queries"]
                    break
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Error in final response generation: {str(e)}")
            return f"I encountered an error processing your request: {str(e)}", {"error": str(e)}