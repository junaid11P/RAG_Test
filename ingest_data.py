import numpy as np
import read
from chunk import manual_chunk_text
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. Setup Connections
connections.connect(alias="default", host="localhost", port="19530")

# 2. Define Schema and Create Collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
]
schema = CollectionSchema(fields=fields, description="Resume RAG embeddings")
collection = Collection(name="resume_chunks", schema=schema)

# 3. Process Data (Chunking & Embedding)
chunks = manual_chunk_text(read.text, chunk_size=500, chunk_overlap=50)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 4. Insert Data
data = [
    embeddings.tolist(),
    chunks
]
collection.insert(data)
collection.flush()
print(f"Successfully inserted {len(chunks)} chunks.")

# 5. Create Index (Crucial for search)
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("Index created and system ready for search.")