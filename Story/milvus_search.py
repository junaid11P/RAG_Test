from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

collection = Collection("story_chunks")
collection.load()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")

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

if __name__ == "__main__":
    q = input("Enter query to search in Milvus: ")
    results = search(q)
    print("\n--- Search Results ---")
    for hit in results[0]:
        print(f"Score: {hit.score:.4f}")
        print(f"Content: {hit.entity.get('text')}")
        print("-" * 20)