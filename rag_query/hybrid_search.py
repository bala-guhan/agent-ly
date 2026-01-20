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
    
    def _build_bm25_index(self):
        """Build BM25 index from documents in vector store."""
        if self._bm25_index is None:
            data = self.vector_store.view_data()
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
        recency_boost: bool = True
    ) -> List:
        from datetime import datetime
        
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
        
        # Need expanded retrieval for date filtering or recency boosting
        needs_filtering = (date_start or date_end) or recency_boost
        initial_k = max(rerank_top_k if (rerank or self.use_reranker) else k, k * 3 if needs_filtering else k)
        
        # Get results with scores if filtering needed
        if needs_filtering:
            results_with_scores = self.vector_store.similarity_search_with_score(question, k=initial_k, filter=filter)
            if not results_with_scores:
                return []
            
            # Filter by date range
            filtered = []
            for doc, score in results_with_scores:
                if date_start or date_end:
                    try:
                        doc_date = datetime.fromisoformat(doc.metadata.get("content_date", ""))
                        if (date_start and doc_date < date_start) or (date_end and doc_date > date_end):
                            continue
                    except (ValueError, TypeError):
                        if date_start or date_end:
                            continue
                
                # Calculate recency score
                recency = 0.0
                if recency_boost and doc.metadata.get("content_date"):
                    try:
                        days_ago = (datetime.now() - datetime.fromisoformat(doc.metadata["content_date"])).days
                        recency = 1.0 / (1.0 + days_ago / 365.0)
                    except (ValueError, TypeError):
                        pass
                
                # Combine similarity (70%) and recency (30%)
                norm_sim = 1.0 / (1.0 + score) if score > 0 else 0.0
                filtered.append((doc, 0.7 * norm_sim + 0.3 * recency if recency_boost else score))
            
            results = [doc for doc, _ in sorted(filtered, key=lambda x: x[1], reverse=recency_boost)[:k]]
        else:
            results = self.vector_store.similarity_search(question, k=initial_k, filter=filter)
            if not results:
                return []
        
        if rerank and self.use_reranker:
            results = self.reranker.rerank(question, results, top_k=k)
        
        return results
    
    def query_with_scores(self, question: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List:
        return self.vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter=filter
        )
    
    def hybrid_query(self, question: str, k: int = 4, alpha: float = 0.5) -> List:
        from langchain_core.documents import Document
        
        self._build_bm25_index()
        
        vector_results = self.vector_store.similarity_search_with_score(question, k=k * 2)
        vector_scores = {doc.page_content: (doc, score) for doc, score in vector_results}
        
        tokenized_query = re.findall(r'\w+', question.lower())
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        
        data = self.vector_store.view_data()
        all_docs = data.get("documents", [])
        all_ids = data.get("ids", [])
        all_metadatas = data.get("metadatas", [])
        
        combined_scores = []
        for idx, (doc_text, doc_id, metadata) in enumerate(zip(all_docs, all_ids, all_metadatas or [{}] * len(all_docs))):
            vector_score = vector_scores.get(doc_text, (None, float('inf')))[1]
            bm25_score = bm25_scores[idx]
            
            if vector_score == float('inf'):
                vector_score = 0.0
            
            normalized_vector = 1 / (1 + vector_score) if vector_score > 0 else 0
            normalized_bm25 = (bm25_score - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-8)
            
            combined_score = alpha * normalized_vector + (1 - alpha) * normalized_bm25
            combined_scores.append((Document(page_content=doc_text, metadata=metadata), combined_score))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in combined_scores[:k]]

