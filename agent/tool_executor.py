import warnings
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict

warnings.filterwarnings("ignore")


def execute_tools_parallel(state, tools_dict):
    """Execute all selected tools in parallel."""
    tool_calls = state.get("tool_calls", [])
    query = state.get("query", "")
    conversation_context = state.get("conversation_context", "")
    
    if not tool_calls:
        return {"tool_results": {}}
    
    # Extract dates for rag_search if needed
    date_start = date_end = None
    if "rag_search" in tool_calls:
        from agent.tools import _rag_date_extraction_llm
        from prompts.date_extraction_prompt import get_date_extraction_prompt
        
        if _rag_date_extraction_llm:
            try:
                date_prompt = get_date_extraction_prompt(query, conversation_context)
                date_extraction = _rag_date_extraction_llm.invoke(date_prompt)
                date_start = date_extraction.date_start
                date_end = date_extraction.date_end
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Date extraction failed: {e}. Proceeding without date filtering.{Style.RESET_ALL}")
    
    # Prepare tool arguments dynamically
    def get_tool_args(tool_name):
        base = {"query": query}
        if tool_name == "rag_search":
            base.update({"date_start": date_start, "date_end": date_end})
        elif tool_name == "web_search":
            base["conversation_context"] = conversation_context
        return base
    
    # Execute tools in parallel
    tool_results = {}
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {
            tool_name: executor.submit(tools_dict[tool_name].invoke, get_tool_args(tool_name))
            for tool_name in tool_calls
            if tool_name in tools_dict
        }
        
        # Collect results
        for tool_name, future in futures.items():
            try:
                result = future.result(timeout=60)  # 60 second timeout per tool
                tool_results[tool_name] = result
            except FutureTimeoutError:
                error_msg = f"Tool {tool_name} timed out after 60 seconds"
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
                tool_results[tool_name] = f"Error: {error_msg}"
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
                tool_results[tool_name] = f"Error: {error_msg}"
    
    return {"tool_results": tool_results}

