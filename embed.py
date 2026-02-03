import numpy as np
import read
from chunk import manual_chunk_text
from sentence_transformers import SentenceTransformer

# Create chunks
chunks = manual_chunk_text(read.text, chunk_size=500, chunk_overlap=50)

print(f"Total chunks: {len(chunks)}")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

print("Embedding dimension:", embeddings.shape)

# Save embeddings
np.save("embeddings.npy", embeddings)
print("Embeddings saved successfully")
