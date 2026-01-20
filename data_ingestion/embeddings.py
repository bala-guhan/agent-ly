import warnings
import os
from dotenv import load_dotenv
from typing import List, Union

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

class EmbeddingProvider:
    def __init__(self, provider: str = "voyage", model: str = "voyage-large-2"):
        self.provider = provider.lower()
        self.model = model
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        if self.provider == "voyage":
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY not found in environment variables")
            return VoyageAIEmbeddings(
                voyage_api_key=api_key,
                model=self.model
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: 'voyage'"
            )
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        return self.embeddings.embed_documents(texts)
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise TypeError("Input must be a string or a list of strings")
    
    def get_provider_info(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self._get_embedding_dimension()
        }
    
    def _get_embedding_dimension(self) -> int:
        dimension_map = {
            "voyage-large-2": 1536,
            "voyage-code-2": 1536,
            "voyage-2": 1024,
            "voyage-lite-02": 1024
        }
        return dimension_map.get(self.model, "unknown")


if __name__ == "__main__":
    try:
        embedding_provider = EmbeddingProvider(
            provider="voyage",
            model="voyage-large-2"
        )
        print("Voyage AI Embeddings initialized successfully")
        print(f"Provider info: {embedding_provider.get_provider_info()}\n")
        
        query = "What is machine learning?"
        query_embedding = embedding_provider.embed_query(query)
        print(f"Query: '{query}'")
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}\n")
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
        doc_embeddings = embedding_provider.embed_documents(documents)
        print(f"Number of documents: {len(documents)}")
        print(f"Number of embeddings: {len(doc_embeddings)}")
        print(f"Each embedding dimension: {len(doc_embeddings[0])}\n")
        
        # Test unified embed method
        single_result = embedding_provider.embed("Hello world")
        multiple_result = embedding_provider.embed(documents)
        print(f"Single embed result type: {type(single_result)}")
        print(f"Multiple embed result type: {type(multiple_result)}")
        
    except Exception as e:
        print(f"Error: {e}")