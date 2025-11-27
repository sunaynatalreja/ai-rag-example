from llm.gemini_llm import ask_gemini

def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    return f"""
You are a strictly grounded AI assistant.

Use ONLY the context below to answer the user.
If the answer is not found, reply: "I don't know."

### CONTEXT:
{context}

### USER QUESTION:
{query}
"""

def run_rag(query, retrieved_docs):
    prompt = build_rag_prompt(query, retrieved_docs)
    answer = ask_gemini(prompt)
    return answer
