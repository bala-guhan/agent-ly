def get_synthesis_prompt(query: str, tool_results: str) -> str:
    return f"""Combine and synthesize the following tool results to answer the user's query.

User Query: "{query}"

Tool Results:
{tool_results}

Instructions:
1. Synthesize all information into a coherent, comprehensive answer
2. Remove redundancy and contradictions
3. Organize information logically
4. Include relevant citations/sources
5. Be concise but complete
6. If information conflicts, prioritize the most relevant or recent source
7. DO NOT add disclaimers, notes, or explanations about source consistency at the end
8. Provide only the answer and sources - no meta-commentary

Final Answer:"""

