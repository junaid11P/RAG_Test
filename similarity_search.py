import numpy as np
import read
from chunk import manual_chunk_text
from sentence_transformers import SentenceTransformer

# ---------------------------------------------
# 1️⃣ Load text and recreate chunks
# ---------------------------------------------

chunks = manual_chunk_text(read.text, chunk_size=500, chunk_overlap=50)
print(f"Total chunks loaded: {len(chunks)}")

# ---------------------------------------------
# 2️⃣ Load saved embeddings
# ---------------------------------------------

embeddings = np.load("embeddings.npy")
print(f"Total embeddings loaded: {len(embeddings)}")

# Safety check
assert len(chunks) == len(embeddings), "Chunks and embeddings count mismatch!"

# ---------------------------------------------
# 3️⃣ Load embedding model
# ---------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------
# 4️⃣ Manual cosine similarity function
# ---------------------------------------------

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

# ---------------------------------------------
# 5️⃣ Similarity search
# ---------------------------------------------

def search(query, top_k=3):
    query_embedding = model.encode(query)

    scores = []

    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, chunks[i]))

    # Sort by similarity score (highest first)
    scores.sort(key=lambda x: x[0], reverse=True)

    return scores[:top_k]

# ---------------------------------------------
# 6️⃣ Run search
# ---------------------------------------------

if __name__ == "__main__":
    query = input("\nAsk a question: ")

    results = search(query, top_k=3)

    print("\n--- Retrieved Chunks (Manual Similarity Search) ---\n")

    for i, (score, chunk) in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Similarity Score: {score:.4f}")
        print(chunk)
        print("-" * 70)
