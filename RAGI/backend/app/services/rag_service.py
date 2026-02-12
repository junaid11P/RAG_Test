from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
from app.db.mongodb import db

class RAGService:
    def __init__(self):
        # FastEmbed is much lighter than HuggingFaceEmbeddings (no PyTorch)
        # Using the same model name to maintain vector compatibility if possible
        self.embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.collection_name = "vectors"

    async def create_rag(self, text: str, user_id: str, doc_id: str):
        """Chunks text and stores in MongoDB Atlas Vector Search."""
        chunks = self.text_splitter.split_text(text)
        
        vector_data = []
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk)
            vector_data.append({
                "user_id": user_id,
                "doc_id": doc_id,
                "text": chunk,
                "embedding": embedding
            })
        
        if vector_data:
            await db.db[self.collection_name].insert_many(vector_data)
        
        return f"mongodb_vector_{doc_id}"

    async def query_rag(self, doc_id: str, query: str, user_id: str):
        """Performs similarity search using MongoDB Atlas Vector Search."""
        query_embedding = self.embeddings.embed_query(query)
        
        # MongoDB Atlas Vector Search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 3,
                    "filter": {
                        "$and": [
                            {"doc_id": {"$eq": doc_id}},
                            {"user_id": {"$eq": user_id}}
                        ]
                    }
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        cursor = db.db[self.collection_name].aggregate(pipeline)
        results = await cursor.to_list(length=3)
        return [res["text"] for res in results]
