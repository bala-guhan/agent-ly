import warnings
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_function,
        persist_directory: Optional[str] = None,
        use_cloud: bool = False,
        api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.use_cloud = use_cloud
        
        if use_cloud:
            api_key = api_key or os.getenv("CHROMA_API_KEY")
            if not api_key:
                raise ValueError(
                    "CHROMA_API_KEY not found. Required when use_cloud=True"
                )
            self.client_settings = {
                "chroma_api_impl": "rest",
                "chroma_server_host": "api.trychroma.com",
                "chroma_server_http_port": 443,
                "chroma_server_ssl_enabled": True,
            }
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                client_settings=self.client_settings,
                client_kwargs={"api_key": api_key}
            )
        else:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory
            )
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None):
        if ids:
            return self.vectorstore.add_documents(documents=documents, ids=ids)
        return self.vectorstore.add_documents(documents=documents)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        if ids and metadatas:
            return self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        elif ids:
            return self.vectorstore.add_texts(texts=texts, ids=ids)
        elif metadatas:
            return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        else:
            return self.vectorstore.add_texts(texts=texts)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_collection(self):
        return self.vectorstore._collection
    
    def view_data(self, limit: Optional[int] = None, include_embeddings: bool = False):
        collection = self.get_collection()
        include_list = ["documents", "metadatas"]
        if include_embeddings:
            include_list.append("embeddings")
        
        results = collection.get(include=include_list)
        
        if limit:
            for key in results:
                if isinstance(results[key], list):
                    results[key] = results[key][:limit]
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        collection = self.get_collection()
        count = collection.count()
        data = collection.get()
        
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "has_ids": len(data.get("ids", [])) > 0,
            "has_metadata": len(data.get("metadatas", [])) > 0,
            "metadata_keys": set() if not data.get("metadatas") else set(
                key for meta in data.get("metadatas", []) if meta 
                for key in meta.keys()
            )
        }
    
    def print_data(self, limit: Optional[int] = 10):
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"{'='*60}\n")
        
        if stats['total_documents'] == 0:
            print("No documents in collection.")
            return
        
        data = self.view_data(limit=limit)
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        for i, (doc_id, doc_text, metadata) in enumerate(zip(ids, documents, metadatas or [None] * len(ids)), 1):
            print(f"Document {i}:")
            print(f"  ID: {doc_id}")
            print(f"  Text: {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
            if metadata:
                print(f"  Metadata: {metadata}")
            print()
    
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None):
        collection = self.get_collection()
        if ids:
            collection.delete(ids=ids)
        elif filter:
            collection.delete(where=filter)
        else:
            raise ValueError("Must provide either ids or filter")
    
    def delete_collection(self):
        collection = self.get_collection()
        self.vectorstore.delete_collection()


if __name__ == "__main__":
    from embeddings import EmbeddingProvider
    
    embedding_provider = EmbeddingProvider(
        provider="voyage",
        model="voyage-large-2"
    )
    
    vector_store = VectorStore(
        collection_name="test_collection",
        embedding_function=embedding_provider.embeddings,
        persist_directory="./chroma_db"
    )
    
    test_documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Apple is a great fruit to have in summer.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Pineapple is the brother of ordinary apples"
    ]
    
    ids = vector_store.add_texts(
        texts=test_documents,
        metadatas=[{"source": f"doc_{i+1}", "topic": "AI"} for i in range(len(test_documents))]
    )
    print(f"Added {len(ids)} documents\n")
    
    
    # Search for similar documents
    query = "What do you think about machine learning?"
    results = vector_store.similarity_search(query, k=2)
    print(f"Search Query: '{query}'")
    print(f"Top {len(results)} Results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content[:100]}...")
        if doc.metadata:
            print(f"     Metadata: {doc.metadata}")
    print()
    