import warnings
from typing import Optional
from langchain_core.tools import tool
from rag_query.rag_query_system import RAGQuerySystem
from llm import LLMProvider
from lib.supabase_client import supabase

warnings.filterwarnings("ignore")


# Initialize systems (these will be set by create_tools)
_rag_system = None
_general_llm = None


def create_tools(llm_provider: str, llm_model: str):
    """Initialize tools with LLM provider and model."""
    global _rag_system, _general_llm
    
    _rag_system = RAGQuerySystem(
        collection_name="documents",
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    _general_llm = LLMProvider(provider=llm_provider, model=llm_model)
    

    return [rag_search, supabase_query]


@tool
def rag_search(
    query: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    recency_boost: bool = True
) -> str:
    """Search enterprise documents using RAG for 2care questions.
    
    Args:
        query: Search query about company, products, services, documentation, etc.
        date_start: Start date for filtering (ISO: YYYY-MM-DD). Only include when query mentions specific time period.
        date_end: End date for filtering (ISO: YYYY-MM-DD). Only include when query mentions specific time period.
        recency_boost: Boost recent documents (default: True).
    
    Returns:
        str: Answer with citations from documents.
    """
    if _rag_system is None:
        return "RAG system not initialized"
    
    # Prepare date range if provided
    date_range = None
    if date_start or date_end:
        date_range = {}
        if date_start and date_start.strip():
            date_range["start"] = date_start.strip()
        if date_end and date_end.strip():
            date_range["end"] = date_end.strip()
    
    result = _rag_system.query(query, k=5, date_range=date_range, recency_boost=recency_boost)
    answer = result["answer"]
    citations = "\n".join([f"[{i+1}] {c['source']}" for i, c in enumerate(result["citations"])])
    return f"{answer}\n\nSources:\n{citations}"


@tool
def supabase_query(query: str) -> str:
    """Query Supabase database for user data, orders, or other structured information.
    
    Args:
        query: The database query or description of what data to retrieve.
    """
    try:
        # Example: query users table
        response = supabase.table("users").select("*").limit(5).execute()
        if response.data:
            return f"Found {len(response.data)} users: {response.data}"
        return "No data found"
    except Exception as e:
        return f"Error querying database: {str(e)}"


@tool
def general_query(query: str) -> str:
    """Answer general questions not related to the enterprise.
    
    Args:
        query: General question, small talk, or question unrelated to the enterprise.
    """
    if _general_llm is None:
        return "General LLM not initialized"
    
    return _general_llm.invoke(query)

