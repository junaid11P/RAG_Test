import ollama
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

collection = Collection("resume_chunks")
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding, 
        anns_field="embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["text"]
    )
    return results

def search_and_answer(query):
    # 1. Retrieve relevant chunks from Milvus
    results = search(query, top_k=3)
    
    # 2. Extract and join the context
    context = "\n".join([hit.entity.get("text") for hit in results[0]])
    
    # 3. Generate the answer with Ollama
    prompt = f"Use this context to answer: {context}\n\nQuestion: {query}"
    response = ollama.generate(model='llama3.2', prompt=prompt)
    
    return response['response']

if __name__ == "__main__":
    q = input("Ask a question about the resume: ")
    print("\n--- AI Answer ---")
    print(search_and_answer(q))