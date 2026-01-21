import warnings
import time
from colorama import Fore, Style
from typing import List, Optional, Dict, Any
from datetime import datetime
from rag_query.hybrid_search import RAG
from llm import LLMProvider
from prompts.rag_query_prompt import get_rag_prompt

warnings.filterwarnings("ignore")


class RAGQuerySystem:
    def __init__(
        self,
        collection_name: str = "documents",
        llm_provider: str = "groq",
        llm_model: str = "llama-3.3-70b-versatile",
        use_reranker: bool = False
    ):
        self.rag = RAG(
            collection_name=collection_name,
            use_reranker=use_reranker
        )
        
        self.llm = LLMProvider(
            provider=llm_provider,
            model=llm_model
        )
    
    def _format_context(self, chunks):
        """Format retrieved chunks into context string with citations."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            file_name = chunk.metadata.get('file_name', 'Unknown')
            page = chunk.metadata.get('page', '')
            
            citation = f"[Source {i}: {file_name}"
            if page:
                citation += f", Page {page}"
            citation += "]"
            
            context_parts.append(f"{citation}\n{chunk.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def query(
        self,
        question: str,
        k: int = 5,
        rerank: bool = False,
        date_range: Optional[Dict[str, str]] = None,
        recency_boost: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system and return answer with metadata.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            rerank: Whether to use reranking
            date_range: Optional dict with "start" and/or "end" dates for temporal filtering (ISO format: YYYY-MM-DD)
            recency_boost: Whether to boost more recent documents
            
        Returns:
            Dictionary with answer, citations, and timing info
        """
        # Step 1: Retrieve chunks
        retrieval_start = time.time()
        try:
            chunks = self.rag.query(
                question, 
                k=k, 
                rerank=rerank,
                date_range=date_range,
                recency_boost=recency_boost
            )
        except Exception as e:
            error_msg = f"Retrieval failed: {type(e).__name__}: {str(e)}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            total_time = time.time() - retrieval_start
            return {
                "answer": f"Error retrieving information: {error_msg}",
                "citations": [],
                "timing": {"total": total_time}
            }
        
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "citations": [],
                "timing": {"total": retrieval_time}
            }
        
        # Step 2: Format context and generate prompt
        context = self._format_context(chunks)
        prompt = get_rag_prompt(context, question)
        
        # Step 3: Generate answer
        llm_start = time.time()
        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            error_msg = f"LLM invocation failed: {type(e).__name__}: {str(e)}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            total_time = time.time() - retrieval_start
            return {
                "answer": f"Error generating answer: {error_msg}",
                "citations": [],
                "timing": {"total": total_time}
            }
        llm_time = time.time() - llm_start
        total_time = time.time() - retrieval_start
        
        # Get citations
        citations = []
        for chunk in chunks:
            citation = {"source": chunk.metadata.get('file_name', 'Unknown')}
            if chunk.metadata.get('page'):
                citation["page"] = chunk.metadata['page']
            citations.append(citation)
        
        return {
            "answer": answer,
            "citations": citations,
            "chunks_count": len(chunks),
            "timing": {"total": total_time}
        }

