import warnings
from prompts.synthesis_prompt import get_synthesis_prompt

warnings.filterwarnings("ignore")


def synthesis_node(state, llm):
    """Combine all tool results into final answer."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", {})
    
    # If no tool results, return error
    if not tool_results:
        return {"final_answer": "No results from tools. Please try rephrasing your question."}
    
    # Format tool results for synthesis
    results_text = "\n\n".join([
        f"=== {tool_name} ===\n{result}"
        for tool_name, result in tool_results.items()
    ])
    
    # Synthesize using LLM
    synthesis_prompt = get_synthesis_prompt(query, results_text)
    final_answer = llm.llm.invoke(synthesis_prompt)
    
    # Extract content if it's a message object
    if hasattr(final_answer, 'content'):
        final_answer = final_answer.content
    
    return {"final_answer": final_answer}

