import sys
import ollama
from milvus_search import search

def generate_rag_answer(question):
    print(f"Searching Milvus for: {question}...")
    results = search(question, top_k=3)
    
    # Extract context from Milvus results
    context = "\n\n".join([hit.entity.get("text") for hit in results[0]])
    
    # Create the prompt for Ollama
    prompt = f"""
    You are an AI assistant helping with a resume query. 
    Use the Context provided below to answer the Question accurately. 
    If the answer is not in the context, say "I don't have that information in the resume."

    Context:
    {context}

    Question: {question}
    """

    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error using Ollama: {str(e)}"

if __name__ == "__main__":
    user_q = input("Ask a question about your resume: ")
    print("\n--- AI ANSWER (Ollama) ---")
    print(generate_rag_answer(user_q))