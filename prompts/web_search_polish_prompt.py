def get_web_search_polish_prompt(query: str, results: str) -> str:
    return f"""Organize and summarize the following web search results for the query: "{query}"

Search Results:
{results}

Provide a clear, well-organized summary that:
1. Synthesizes the key information from all sources
2. Groups related findings together
3. Includes source citations with URLs
4. Is concise but comprehensive
5. Maintains accuracy and relevance
6. DO NOT add disclaimers, notes, or explanations about source consistency at the end
7. Provide only the summary and sources - no meta-commentary

Summary:"""

