def get_enhanced_prompt(context: str, question: str):
    return f"""Based on the following search results, please answer the question: {context}
                Question: {question}"""   

