import read
from chunk import manual_chunk_text
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. Setup Connections
connections.connect(alias="default", host="localhost", port="19530")

# 2. Define Schema and Create Collection
from pymilvus import utility
if utility.has_collection("story_chunks"):
    utility.drop_collection("story_chunks")
    print("Dropped old collection for a clean ingest.")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields=fields, description="Story RAG embeddings")
collection = Collection(name="story_chunks", schema=schema)

# 3. Process Data (Chunking & Embedding)
chunks = manual_chunk_text(read.text, chunk_size=512, chunk_overlap=50)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
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