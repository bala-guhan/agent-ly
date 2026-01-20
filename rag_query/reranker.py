import warnings
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.documents import Document
import cohere

warnings.filterwarnings("ignore")
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class Reranker:
    def __init__(self, model: str = "rerank-english-v3.0"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        self.client = cohere.Client(api_key)
        self.model = model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        if not documents:
            return []
        
        doc_texts = [doc.page_content for doc in documents]
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=top_k if top_k else len(documents)
        )
        
        # Cohere rerank returns a RerankResponse object with a results attribute
        # Handle different possible response structures
        if hasattr(response, 'results'):
            results_list = response.results
        elif isinstance(response, list):
            results_list = response
        else:
            # Try to iterate directly
            results_list = response
        
        reranked_docs = []
        for result in results_list:
            # Access the index property - handle different access patterns
            idx = None
            
            # Try to get index as an attribute (not a method)
            if hasattr(result, 'index'):
                index_attr = getattr(result, 'index')
                # Check if it's not a callable (method) - if it's callable, it's the list.index method
                if not callable(index_attr):
                    idx = index_attr
            
            # If that didn't work, try as dict
            if idx is None and isinstance(result, dict):
                idx = result.get('index')
            
            # If still None, try accessing via __dict__
            if idx is None and hasattr(result, '__dict__'):
                idx = result.__dict__.get('index')
            
            # Ensure idx is an integer before using as index
            if isinstance(idx, int) and 0 <= idx < len(documents):
                reranked_docs.append(documents[idx])
        
        return reranked_docs

