import warnings
import time
from typing import List, Optional, Dict, Any
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
        
        self.timing_stats = {}
    
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
        rerank: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system and return answer with metadata.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            rerank: Whether to use reranking
            
        Returns:
            Dictionary with answer, citations, and timing info
        """
        # Step 1: Retrieve chunks
        retrieval_start = time.time()
        chunks = self.rag.query(question, k=k, rerank=rerank)
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "citations": [],
                "timing": {"retrieval": retrieval_time, "total": retrieval_time}
            }
        
        # Step 2: Format context
        formatting_start = time.time()
        context = self._format_context(chunks)
        formatting_time = time.time() - formatting_start
        
        # Step 3: Generate prompt
        prompt_start = time.time()
        prompt = get_rag_prompt(context, question)
        prompt_time = time.time() - prompt_start
        
        # Step 4: Generate answer
        llm_start = time.time()
        answer = self.llm.invoke(prompt)
        llm_time = time.time() - llm_start
        
        # Get citations
        citations = []
        for i, chunk in enumerate(chunks, 1):
            file_name = chunk.metadata.get('file_name', 'Unknown')
            page = chunk.metadata.get('page', '')
            citation = {"source": file_name}
            if page:
                citation["page"] = page
            citations.append(citation)
        
        total_time = retrieval_time + formatting_time + prompt_time + llm_time
        
        return {
            "answer": answer,
            "citations": citations,
            "chunks_count": len(chunks),
            "timing": {
                "retrieval": retrieval_time,
                "formatting": formatting_time,
                "prompt": prompt_time,
                "llm": llm_time,
                "total": total_time
            }
        }

