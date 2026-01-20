import warnings
import sys
from typing import List, Optional, Union
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.documents import DocumentProcessor
from data_ingestion.embeddings import EmbeddingProvider
from data_ingestion.vector_store import VectorStore

warnings.filterwarnings("ignore")

class RAGSystem:
    def __init__(
        self,
        collection_name: str = "rag_collection",
        embedding_provider: str = "voyage",
        embedding_model: str = "voyage-large-2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: Optional[str] = None,
        use_cloud: bool = False
    ):
        if persist_directory is None:
            from pathlib import Path
            project_root = Path(__file__).parent
            persist_directory = str(project_root / "chroma_db")
        self.collection_name = collection_name
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
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
    
    def add_documents(
        self,
        source: Union[str, List[str], Path, List[Path]],
        metadata: Optional[dict] = None,
        auto_chunk: bool = True
    ) -> List[str]:
        documents = self.document_processor.load_and_chunk(
            source=source,
            metadata=metadata,
            auto_chunk=auto_chunk
        )
        
        return self.vector_store.add_documents(documents)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        auto_chunk: bool = True
    ) -> List[str]:
        documents = self.document_processor.load_texts(
            texts=texts,
            metadata=None,
            auto_chunk=auto_chunk
        )
        
        if metadatas:
            for doc, meta in zip(documents, metadatas):
                doc.metadata.update(meta)
        
        return self.vector_store.add_documents(documents)
    
    def get_stats(self):
        return self.vector_store.get_stats()
    
    def get_persist_directory(self) -> Optional[str]:
        return self.vector_store.persist_directory


if __name__ == "__main__":
    rag = RAGSystem(
        collection_name="documents",
        persist_directory="./chroma_db"
    )
    
    print("="*60)
    print("RAG Indexing System Initialized")
    print("="*60)
    
    print(f"Collection: {rag.collection_name}")
    print(f"Persist Directory: {rag.get_persist_directory()}")
    print(f"Embedding Model: {rag.embedding_provider.model}\n")
    
    print("Adding documents to vector store...")
    
    pdf_file = "balaguhanesh-cbc-report.pdf"
    try:
        ids = rag.add_documents(
            pdf_file,
            metadata={"document_type": "medical_report", "category": "CBC"},
            auto_chunk=True
        )
        print(f"Added {len(ids)} chunks from {pdf_file}\n")
    except FileNotFoundError:
        print(f"File {pdf_file} not found. Skipping...\n")
    
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "ChromaDB is a vector database for storing embeddings."
    ]
    
    ids = rag.add_texts(test_texts, metadatas=[
        {"topic": "AI"},
        {"topic": "programming"},
        {"topic": "database"}
    ])
    
    print(f"Added {len(ids)} text documents\n")
    
    stats = rag.get_stats()
    print("Collection Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Metadata Keys: {stats['metadata_keys']}\n")
    
    print("Indexing complete! Use ChromaDB Explorer to view the data.")
