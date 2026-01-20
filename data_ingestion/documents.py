import warnings
import os
from typing import List, Optional, Union
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader
)

warnings.filterwarnings("ignore")


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def _get_file_loader(self, file_path: str):
        """Get appropriate loader based on file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            return PyPDFLoader(file_path)
        elif file_ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding="utf-8")
        else:
            return UnstructuredFileLoader(file_path)
    
    def load_and_chunk(
        self,
        source: Union[str, List[str], Path, List[Path]],
        metadata: Optional[dict] = None,
        auto_chunk: bool = True
    ) -> List[Document]:
        base_metadata = metadata or {}
        documents = []
        
        sources = [source] if not isinstance(source, list) else source
        sources = [str(s) for s in sources]
        
        for src in sources:
            src_path = Path(src)
            
            if src_path.exists() and src_path.is_file():
                loader = self._get_file_loader(src)
                docs = loader.load()
                
                file_metadata = {
                    **base_metadata,
                    "source": str(src_path),
                    "file_name": src_path.name
                }
                
                for doc in docs:
                    doc.metadata.update(file_metadata)
                documents.extend(docs)
            else:
                doc = Document(
                    page_content=src,
                    metadata={**base_metadata, "source": "text_input"}
                )
                documents.append(doc)
        
        if auto_chunk and documents:
            return self.text_splitter.split_documents(documents)
        
        return documents
    
    def load_text(self, text: str, metadata: Optional[dict] = None, auto_chunk: bool = True) -> List[Document]:
        """Load and chunk a single text string."""
        return self.load_and_chunk(text, metadata=metadata, auto_chunk=auto_chunk)
    
    def load_texts(self, texts: List[str], metadata: Optional[dict] = None, auto_chunk: bool = True) -> List[Document]:
        """Load and chunk multiple text strings."""
        return self.load_and_chunk(texts, metadata=metadata, auto_chunk=auto_chunk)
    
    def load_file(self, file_path: Union[str, Path], metadata: Optional[dict] = None, auto_chunk: bool = True) -> List[Document]:
        """Load and chunk a single file."""
        return self.load_and_chunk(file_path, metadata=metadata, auto_chunk=auto_chunk)
    
    def load_files(self, file_paths: List[Union[str, Path]], metadata: Optional[dict] = None, auto_chunk: bool = True) -> List[Document]:
        """Load and chunk multiple files."""
        return self.load_and_chunk(file_paths, metadata=metadata, auto_chunk=auto_chunk)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Manually chunk already loaded documents."""
        return self.text_splitter.split_documents(documents)

