def get_decision_prompt(query: str, conversation_history: str = "") -> str:
    context_section = ""
    if conversation_history:
        context_section = f"\n\nConversation History:\n{conversation_history}\n\nUse this context to:\n- Understand references like 'the company', 'they', 'it', etc.\n- Recognize conversational patterns (jokes, follow-ups, continuations)\n- Determine if the query continues a previous conversation thread"
    
    return f"""Analyze the query and make a decision with reasoning and confidence scores.

Query: "{query}"{context_section}

Available Tools:
- rag_search: For questions about 2care company, products, services, or enterprise documentation
- web_search: When user explicitly asks to search the web or needs current/real-time information
- supabase_query: For database queries about users, orders, or structured data

Decision Process:

1. ANALYZE: Determine if query needs tools or can be answered directly

2. PROVIDE REASONING: Explain your overall decision clearly

3. ASSIGN CONFIDENCE (0.0-1.0):
   - 0.9-1.0: Very confident, clear decision
   - 0.7-0.9: Confident, good decision
   - 0.5-0.7: Moderately confident
   - <0.5: Low confidence, reconsider

4. IF DIRECT_ANSWER:
   - Set direct_answer=True, tool_calls=[]
   - Explain why no tools needed
   - Confidence should be >0.7 for direct answers

5. IF TOOLS NEEDED:
   - For EACH tool, provide:
     * tool_name: Which tool
     * reasoning: Why this specific tool is needed
     * confidence: How confident (0.0-1.0) this tool will help
   - Only include tools with confidence >0.5
   - Each tool can only appear once

Decision Rules:
- Greetings/casual chat → direct_answer=True (confidence >0.8)
- Questions about "2care" → rag_search (confidence >0.8)
- Web search requests → web_search (confidence >0.7)
- Database queries → supabase_query (confidence >0.7)
- Can combine multiple tools if needed

IMPORTANT - Multiple Tool Triggers:
- If user explicitly asks to "search all sources", "check everywhere", "look in all places", "comprehensive search", "search all databases", "check all sources", or similar phrases → ALWAYS trigger ALL THREE tools (rag_search, web_search, supabase_query) with confidence >0.7 each
- If user asks for "complete information", "everything about", "full picture", "comprehensive overview" → Consider triggering multiple tools
- If user mentions "internal docs AND web" or "database AND documents" → Trigger the mentioned tools
- When in doubt about source, err on the side of including multiple tools for comprehensive coverage

Examples:
- "hi" → direct_answer=True, confidence=0.95, reasoning="Simple greeting, no information needed"
- "what is 2care" → rag_search, confidence=0.9, tool_reasoning="Question about 2care company, need enterprise docs"
- "2care Q4 2024 financials" → rag_search, confidence=0.95, tool_reasoning="Specific 2care financial data, need RAG search"
- "search all sources for 2care information" → rag_search (0.9), web_search (0.9), supabase_query (0.8), reasoning="User explicitly requested all sources"
- "check everywhere about 2care's Q4 performance" → rag_search (0.9), web_search (0.85), supabase_query (0.75), reasoning="User wants comprehensive search across all sources"
- "give me everything about 2care from all databases and sources" → rag_search (0.9), web_search (0.9), supabase_query (0.9), reasoning="User explicitly requested all sources and databases"

Make your decision with reasoning and confidence:"""

