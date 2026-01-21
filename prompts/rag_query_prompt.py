def get_rag_prompt(context: str, question: str) -> str:
    """
    Generate RAG prompt with context and question.
    
    Args:
        context: Retrieved relevant context chunks
        question: User's question
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""You are an intelligent assistant that answers questions based on the provided context from enterprise documents.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the context provided above.
2. If the answer cannot be found in the context, clearly state "I don't have enough information in the provided context to answer this question."
3. Be concise, accurate, and specific.
4. If multiple relevant pieces of information exist, synthesize them into a coherent answer.
5. Do not make up information or use knowledge outside the provided context.
6. DO NOT add disclaimers, notes, or explanations about source consistency, dates, or versions at the end.
7. Provide only the direct answer - no meta-commentary or assumptions about source consistency.

Answer:"""

