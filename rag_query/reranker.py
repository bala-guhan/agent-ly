import warnings
import os
import sys
from colorama import Fore, Style
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
        
        try:
            doc_texts = [doc.page_content for doc in documents]
            
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k if top_k else len(documents)
            )
        except Exception as e:
            error_msg = f"Reranking failed: {type(e).__name__}: {str(e)}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            # Return original documents if reranking fails
            return documents[:top_k] if top_k else documents
        
        # Cohere rerank returns a RerankResponse with results list
        results_list = response.results if hasattr(response, 'results') else response
        
        # Extract indices and build reranked document list
        reranked_docs = []
        for result in results_list:
            # Get index - try attribute first, then dict access
            idx = getattr(result, 'index', None) if not callable(getattr(result, 'index', None)) else None
            if idx is None and isinstance(result, dict):
                idx = result.get('index')
            
            # Add document if valid index
            if isinstance(idx, int) and 0 <= idx < len(documents):
                reranked_docs.append(documents[idx])
        
        return reranked_docs

