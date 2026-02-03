import numpy as np
import read
from chunk import manual_chunk_text
from pymilvus import connections, Collection

# Connect
connections.connect("default", host="localhost", port="19530")

# Load collection
collection = Collection("resume_chunks")

# Load chunks and embeddings
chunks = manual_chunk_text(read.text, chunk_size=500, chunk_overlap=50)
embeddings = np.load("embeddings.npy")

assert len(chunks) == len(embeddings)

# Insert data
data = [
    embeddings.tolist(),
    chunks
]

collection.insert(data)
collection.flush()

print("Embeddings inserted into Milvus successfully")
