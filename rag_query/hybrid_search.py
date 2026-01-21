import warnings
import re
from typing import List, Optional, Dict, Any
from pathlib import Path
from rank_bm25 import BM25Okapi

from data_ingestion.embeddings import EmbeddingProvider
from data_ingestion.vector_store import VectorStore

warnings.filterwarnings("ignore")

class RAG:
    def __init__(
        self,
        collection_name: str,
        embedding_provider: str = "voyage",
        embedding_model: str = "voyage-large-2",
        persist_directory: Optional[str] = None,
        use_cloud: bool = True,  # Default to cloud
        use_reranker: bool = False
    ):
        if persist_directory is None:
            project_root = Path(__file__).parent.parent
            persist_directory = str(project_root / "chroma_db")
        self.collection_name = collection_name
        self.use_reranker = use_reranker
        
        self.embedding_provider = EmbeddingProvider(
            provider=embedding_provider,
            model=embedding_model
        )
        
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_function=self.embedding_provider.embeddings,
            persist_directory=persist_directory,
            use_cloud=use_cloud
        )
        
        if use_reranker:
            from .reranker import Reranker
            self.reranker = Reranker()
        
        self._bm25_index = None
        self._documents = None
        self._bm25_data_cache = None  # Cache the data to avoid multiple view_data() calls
    
    def _build_bm25_index(self):
        """Build BM25 index from documents in vector store. Caches data to avoid redundant calls."""
        if self._bm25_index is None or self._bm25_data_cache is None:
            data = self.vector_store.view_data()
            self._bm25_data_cache = data  # Cache the data
            documents = data.get("documents", [])
            self._documents = documents
            
            tokenized_docs = [re.findall(r'\w+', doc.lower()) for doc in documents]
            self._bm25_index = BM25Okapi(tokenized_docs)
    
    def query(
        self, 
        question: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None, 
        rerank: bool = False, 
        rerank_top_k: int = 20,
        date_range: Optional[Dict[str, str]] = None,
        recency_boost: bool = True,
        hybrid_alpha: float = 0.5
    ) -> List:
        from datetime import datetime
        from langchain_core.documents import Document
        
        # Parse date range
        date_start = date_end = None
        if date_range:
            try:
                date_start = datetime.fromisoformat(date_range["start"]) if date_range.get("start") else None
            except (ValueError, TypeError):
                pass
            try:
                date_end = datetime.fromisoformat(date_range["end"]) if date_range.get("end") else None
            except (ValueError, TypeError):
                pass
        
        # Need expanded retrieval for date filtering, recency boosting, or reranking
        needs_filtering = (date_start or date_end) or recency_boost
        initial_k = max(rerank_top_k if (rerank or self.use_reranker) else k, k * 3 if needs_filtering else k * 2)
        
        # Step 1: Hybrid Search (BM25 + Semantic)
        # Build BM25 index
        self._build_bm25_index()
        
        # Semantic search (vector similarity)
        vector_results = self.vector_store.similarity_search_with_score(question, k=initial_k, filter=filter)
        vector_scores = {doc.page_content: (doc, score) for doc, score in vector_results}
        
        # BM25 search (keyword matching)
        tokenized_query = re.findall(r'\w+', question.lower())
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        
        # Use cached data instead of calling view_data() again
        data = self._bm25_data_cache
        all_docs = data.get("documents", [])
        all_ids = data.get("ids", [])
        all_metadatas = data.get("metadatas", [])
        
        # Combine BM25 and semantic scores
        combined_results = []
        for idx, (doc_text, doc_id, metadata) in enumerate(zip(all_docs, all_ids, all_metadatas or [{}] * len(all_docs))):
            # Get vector score
            vector_doc, vector_score = vector_scores.get(doc_text, (None, float('inf')))
            if vector_score == float('inf'):
                vector_score = 0.0
            
            # Get BM25 score
            bm25_score = bm25_scores[idx]
            
            # Normalize scores
            normalized_vector = 1.0 / (1.0 + vector_score) if vector_score > 0 else 0.0
            normalized_bm25 = (bm25_score - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-8) if max(bm25_scores) > min(bm25_scores) else 0.0
            
            # Hybrid score: alpha * semantic + (1-alpha) * BM25
            hybrid_score = hybrid_alpha * normalized_vector + (1 - hybrid_alpha) * normalized_bm25
            
            # Use vector doc if available, otherwise create new document
            doc = vector_doc if vector_doc else Document(page_content=doc_text, metadata=metadata)
            combined_results.append((doc, hybrid_score))
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Step 2: Temporal Filtering (if date range provided)
        if date_start or date_end:
            filtered = []
            docs_without_dates = []
            for doc, score in combined_results:
                try:
                    doc_date_str = doc.metadata.get("content_date", "")
                    if not doc_date_str:
                        # Document has no date - include it but with lower priority
                        docs_without_dates.append((doc, score * 0.8))
                        continue
                    doc_date = datetime.fromisoformat(doc_date_str)
                    if (date_start and doc_date < date_start) or (date_end and doc_date > date_end):
                        continue
                    filtered.append((doc, score))
                except (ValueError, TypeError):
                    # Invalid date format - include it but with lower priority
                    docs_without_dates.append((doc, score * 0.8))
            
            # If date filtering resulted in no results, include documents without dates
            if not filtered and docs_without_dates:
                filtered = docs_without_dates
            # Otherwise, append documents without dates at the end (lower priority)
            elif filtered and docs_without_dates:
                filtered.extend(docs_without_dates)
            
            combined_results = filtered
                
        if recency_boost:
            boosted_results = []
            for doc, hybrid_score in combined_results:
                recency = 0.0
                if doc.metadata.get("content_date"):
                    try:
                        days_ago = (datetime.now() - datetime.fromisoformat(doc.metadata["content_date"])).days
                        recency = 1.0 / (1.0 + days_ago / 365.0)
                    except (ValueError, TypeError):
                        pass
                
                # Combine hybrid score (70%) and recency (30%)
                final_score = 0.7 * hybrid_score + 0.3 * recency
                boosted_results.append((doc, final_score))
            
            combined_results = boosted_results
            combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k before reranking
        results = [doc for doc, _ in combined_results[:rerank_top_k if (rerank or self.use_reranker) else k]]
        
        # Step 4: Reranking (if enabled)
        if rerank and self.use_reranker:
            results = self.reranker.rerank(question, results, top_k=k)
        elif len(results) > k:
            results = results[:k]
        
        return results
    
    def query_with_scores(self, question: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List:
        return self.vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter=filter
        )

