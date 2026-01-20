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
        use_cloud: bool = False,
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
    
    def query(self, question: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, rerank: bool = False, rerank_top_k: int = 20) -> List:
        initial_k = rerank_top_k if (rerank or self.use_reranker) else k
        results = self.vector_store.similarity_search(
            query=question,
            k=initial_k,
            filter=filter
        )
        
        if not results:
            return []
        
        # Only rerank if explicitly requested via rerank=True parameter
        # and reranker is initialized
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

