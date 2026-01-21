import warnings
import os
from colorama import Fore, Style
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_core.tools import tool
from rag_query.rag_query_system import RAGQuerySystem
from llm import LLMProvider
from lib.supabase_client import supabase
from prompts.web_search_reform_prompt import get_web_search_reform_prompt
from prompts.web_search_polish_prompt import get_web_search_polish_prompt

warnings.filterwarnings("ignore")

_rag_system = None
_general_llm = None
_web_search_llm = None
_web_search_structured_llm = None


class QueryReform(BaseModel):
    """Three reformulated search queries."""
    query1: str = Field(description="First search query approaching the topic from one angle")
    query2: str = Field(description="Second search query approaching the topic from a different angle")
    query3: str = Field(description="Third search query approaching the topic from yet another angle")


class DateExtraction(BaseModel):
    """Extracted date range from query."""
    date_start: Optional[str] = Field(
        default=None,
        description="Start date in ISO format (YYYY-MM-DD). Extract if query mentions year, quarter, month, or date range."
    )
    date_end: Optional[str] = Field(
        default=None,
        description="End date in ISO format (YYYY-MM-DD). Extract if query mentions year, quarter, month, or date range."
    )


class ToolCallDetail(BaseModel):
    """Details for a single tool call with reasoning and confidence."""
    tool_name: Literal["rag_search", "web_search", "supabase_query"] = Field(
        description="Name of the tool to call"
    )
    reasoning: str = Field(
        description="Why this specific tool is needed for this query"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence (0.0-1.0) that this tool will provide useful results"
    )


class ToolDecision(BaseModel):
    """Decision model for which tools to call with reasoning and confidence."""
    reasoning: str = Field(
        description="Overall reasoning for this decision - why direct_answer or tool_calls were chosen"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence score (0.0-1.0) in this decision. Higher = more certain."
    )
    direct_answer: bool = Field(
        default=False,
        description="If True, answer directly without tools (high confidence that no tools needed)"
    )
    tool_calls: List[ToolCallDetail] = Field(
        default_factory=list,
        description="List of tools to call with reasoning and confidence for each. Empty if direct_answer=True."
    )


def create_tools(llm_provider: str, llm_model: str):
    global _rag_system, _general_llm, _web_search_llm, _web_search_structured_llm, _rag_date_extraction_llm
    
    _rag_system = RAGQuerySystem(
        collection_name="documents",
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    _general_llm = LLMProvider(provider=llm_provider, model=llm_model)
    
    _web_search_llm = LLMProvider(provider="groq", model="llama-3.3-70b-versatile")
    _web_search_structured_llm = _web_search_llm.llm.with_structured_output(QueryReform)
    
    # LLM for date extraction from queries (use Groq for speed)
    date_extraction_llm_provider = LLMProvider(provider="groq", model="llama-3.3-70b-versatile")
    _rag_date_extraction_llm = date_extraction_llm_provider.llm.with_structured_output(DateExtraction)
    

    return [rag_search, supabase_query, web_search]


@tool
def rag_search(
    query: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    recency_boost: bool = True
) -> str:
    """Search enterprise documents for information about 2care company, products, services, or documentation.
    
    Use this tool for ANY question mentioning "2care" or asking about the enterprise/company.
    
    Args:
        query: Search query about 2care company, products, services, documentation, etc.
        date_start: Start date in ISO format (YYYY-MM-DD). Extract from query if it mentions:
            - Year: "2024" → "2024-01-01", "2025" → "2025-01-01"
            - Quarter: "Q4 2024" → "2024-10-01", "Q1 2025" → "2025-01-01"
            - Month: "January 2024" → "2024-01-01", "March 2025" → "2025-03-01"
            - Range: "between X and Y" → extract start date
        date_end: End date in ISO format (YYYY-MM-DD). Extract from query if it mentions:
            - Year: "2024" → "2024-12-31", "2025" → "2025-12-31"
            - Quarter: "Q4 2024" → "2024-12-31", "Q1 2025" → "2025-03-31"
            - Month: "January 2024" → "2024-01-31", "March 2025" → "2025-03-31"
            - Range: "between X and Y" → extract end date
        recency_boost: Boost recent documents (default: True).
    
    Returns:
        str: Answer with citations from documents.
    """
    print(f"{Fore.MAGENTA}[rag_search]{Style.RESET_ALL} query: {query[:100]}..., date_start: {date_start}, date_end: {date_end}")
    if _rag_system is None:
        print(f"{Fore.RED}Error: RAG system not initialized{Style.RESET_ALL}")
        return "RAG system not initialized"
    
    try:
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
        citations = "\n".join([f"{Fore.BLUE}[{i+1}]{Style.RESET_ALL} {c['source']}" for i, c in enumerate(result["citations"])])
        return f"{answer}\n\n{Fore.BLUE}Sources:{Style.RESET_ALL}\n{citations}"
    except Exception as e:
        error_msg = f"Error in RAG search: {type(e).__name__}: {str(e)}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        return f"Error: {error_msg}"


@tool
def supabase_query(query: str) -> str:
    """Query Supabase database for user data, orders, or other structured information.
    
    Args:
        query: The database query or description of what data to retrieve.
    """
    print(f"{Fore.MAGENTA}[db_search]{Style.RESET_ALL} query: {query[:100]}...")
    try:
        # Example: query users table
        response = supabase.table("users").select("*").limit(5).execute()
        if response.data:
            return f"Found {len(response.data)} users: {response.data}"
        return "No data found"
    except Exception as e:
        error_msg = f"Error querying database: {type(e).__name__}: {str(e)}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        return f"Error: {error_msg}"


@tool
def general_query(query: str) -> str:
    """Answer general questions not related to the enterprise.
    
    Args:
        query: General question, small talk, or question unrelated to the enterprise.
    """
    if _general_llm is None:
        return "General LLM not initialized"
    
    return _general_llm.invoke(query)


@tool
def web_search(query: str, conversation_context: Optional[str] = None) -> str:
    """Search the web for current information, real-time data, news, or information not in enterprise documents.
    
    Use when user explicitly requests web search or needs current/real-time information from the internet.
    
    Args:
        query: Search query about current events, news, or information requiring web search.
        conversation_context: Optional conversation history for context (e.g., to understand "the company" refers to 2care).
    
    Returns:
        str: Organized web search results with sources and citations.
    """
    print(f"{Fore.MAGENTA}[web_search]{Style.RESET_ALL} query: {query[:100]}...")
    if _web_search_llm is None or _web_search_structured_llm is None:
        print(f"{Fore.RED}Error: Web search system not initialized{Style.RESET_ALL}")
        return "Web search system not initialized"
    
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print(f"{Fore.RED}Error: TAVILY_API_KEY not found in environment variables{Style.RESET_ALL}")
            return "TAVILY_API_KEY not found in environment variables"
        
        tavily_client = TavilyClient(api_key=api_key)
        
        # Step 1: Reform query into 3 different queries using structured output with context
        reform_prompt = get_web_search_reform_prompt(query, conversation_context)
        reform_result = _web_search_structured_llm.invoke(reform_prompt)
        queries = [reform_result.query1, reform_result.query2, reform_result.query3]
        
        # Step 2: Search with Tavily for each query and collect results
        all_results = []
        for q in queries:
            try:
                response = tavily_client.search(query=q, max_results=5, search_depth="basic")
                if response and "results" in response:
                    all_results.extend(response["results"])
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Tavily search failed for query '{q}': {e}{Style.RESET_ALL}")
                continue
        
        if not all_results:
            print(f"{Fore.YELLOW}Warning: No web search results found{Style.RESET_ALL}")
            return "No web search results found."
        
        # Step 3: Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Step 4: Format results for polishing
        results_text = "\n\n---\n\n".join([
            f"[{i}] {r.get('title', 'No title')}\nURL: {r.get('url', '')}\nContent: {r.get('content', r.get('raw_content', ''))[:500]}..."
            for i, r in enumerate(unique_results[:10], 1)
        ])
        
        # Step 5: Polish and organize results using Groq
        polish_prompt = get_web_search_polish_prompt(query, results_text)
        polished_output = _web_search_llm.invoke(polish_prompt)
        return polished_output
        
    except ImportError:
        error_msg = "Tavily library not installed. Install with: pip install tavily-python"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        return error_msg
    except Exception as e:
        error_msg = f"Error during web search: {type(e).__name__}: {str(e)}"
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        return f"Error: {error_msg}"

