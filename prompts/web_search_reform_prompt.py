from typing import Optional

def get_web_search_reform_prompt(query: str, conversation_context: Optional[str] = None) -> str:
    context_section = ""
    if conversation_context:
        context_section = f"\nConversation context:\n{conversation_context}\n\nUse this context to understand references like 'the company', 'they', etc."
    
    return f"""Transform the search query into exactly 3 different queries for comprehensive web search.
Each query should approach from a different angle.

Query: {query}{context_section}

Requirements:
- Generate exactly 3 queries
- Each explores different dimensions
- Be specific and searchable
- Avoid redundancy

Generate the 3 queries:"""

