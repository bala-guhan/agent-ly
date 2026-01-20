import warnings
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
def rag_search(query: str) -> str:
    """Search enterprise documents using RAG. Use this for questions about 2care company, products, services, or documentation.
    
    Args:
        query: The search query about company, products, services, documentation, technical details, implementation guides, security, sales, etc. Must be a string.
    
    Returns:
        str: The answer with citations from enterprise documents.
    """
    if _rag_system is None:
        return "RAG system not initialized"
    
    result = _rag_system.query(query, k=5)
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

