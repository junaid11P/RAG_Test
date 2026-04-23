import math
import re
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.db.mongodb import db


class RAGService:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        self.collection_name = "document_embeddings"

    async def create_rag(
        self,
        text: str,
        user_id: str,
        doc_id: str,
        email: str = None,
        expires_at=None
    ) -> str:
        """
        Chunk text and store vectors in MongoDB Atlas Vector Search.
        """
        logging.info(f"Chunking text for {doc_id}...")
        chunks = self.text_splitter.split_text(text)
        logging.info(f"Created {len(chunks)} chunks. Generating embeddings...")

        now = datetime.utcnow()
        vector_data = []

        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logging.info(f"Processing chunk {i}/{len(chunks)}...")

            dates = re.findall(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", chunk)

            section = "General"
            if re.search(r"(?i)^(chapter|section|\d+\.)|\n[A-Z][a-zA-Z\s]+:", chunk):
                section = "Header/Structured"
            elif "?" in chunk:
                section = "Q&A"

            entities = list(set(re.findall(r"(?<!^)(?<!\.\s)[A-Z][a-z]+", chunk)))
            measurements = list(set(re.findall(r"\d+\.?\d*\s?(mg|kg|dL|%)", chunk)))

            extracted_metadata = {
                "T_i": dates,
                "S_i": section,
                "E_i": entities[:5] + measurements[:5]
            }

            embedding = self.embeddings.embed_query(chunk)

            vector_data.append({
                "user_id": user_id,
                "email": email,
                "doc_id": doc_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": extracted_metadata,
                "created_at": now,
                "created_date": now.strftime("%Y-%m-%d"),
                "created_time": now.strftime("%H:%M:%S"),
                "expires_at": expires_at
            })

        if vector_data:
            logging.info(f"Uploading {len(vector_data)} vectors to Atlas...")
            await db.db[self.collection_name].insert_many(vector_data)
            logging.info("Vector storage complete.")

        return f"mongodb_vector_{doc_id}"

    def _is_temporal_priority_query(self, query: str) -> bool:
        """
        Stronger recency preference for questions asking for latest guidance,
        best practice, recommendation, current approach, etc.
        """
        q = query.lower()
        temporal_terms = [
            "latest",
            "current",
            "recent",
            "newest",
            "best practice",
            "best practices",
            "recommended",
            "recommendation",
            "modern",
            "now",
            "today",
            "currently"
        ]
        return any(term in q for term in temporal_terms)

    def _parse_best_chunk_date(self, text: str, metadata: Dict[str, Any], created_at):
        date_matches = []
        date_matches.extend(re.findall(r"(\d{4}-\d{2}-\d{2})", text))
        date_matches.extend(re.findall(r"(\d{2}/\d{2}/\d{4})", text))

        if metadata.get("T_i"):
            date_matches.extend(metadata["T_i"])

        chunk_date = created_at
        if date_matches:
            parsed_dates = []
            for ds in date_matches:
                try:
                    if "-" in ds:
                        parsed_dates.append(datetime.strptime(ds, "%Y-%m-%d"))
                    elif "/" in ds:
                        parsed_dates.append(datetime.strptime(ds, "%d/%m/%Y"))
                except Exception:
                    pass

            if parsed_dates:
                chunk_date = max(parsed_dates)

        return chunk_date

    async def query_rag(
        self,
        doc_id: str,
        query: str,
        user_id: str,
        return_scores: bool = False,
        time_sensitive: bool = False,
        top_k: int = 8
    ):
        """
        Vector search with:
        - semantic similarity
        - exact-term boost
        - pseudo rerank
        - discourse/entity weighting
        - temporal weighting
        - extra recency priority for 'latest/best practice/current' style questions
        """
        query_embedding = self.embeddings.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 120,
                    "limit": max(12, top_k),
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
                    "created_at": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        cursor = db.db[self.collection_name].aggregate(pipeline)
        results = await cursor.to_list(length=max(12, top_k))

        if not results:
            return []

        now = datetime.utcnow()
        query_terms = set(re.findall(r"(?i)\b[a-z0-9_\-]+\b", query.lower()))
        temporal_priority = self._is_temporal_priority_query(query)

        # Weights
        ALPHA = 0.35   # semantic similarity
        BETA = 0.15    # rerank
        GAMMA = 0.15   # entity match
        DELTA = 0.10   # discourse score
        ETA = 0.10     # authority score
        MU = 0.15      # contradiction penalty

        scored_results: List[Dict[str, Any]] = []

        for res in results:
            text = res.get("text", "")
            text_lower = text.lower()
            semantic_score = float(res.get("score", 0.0))
            created_at = res.get("created_at")
            metadata = res.get("metadata", {})

            # Exact term boost
            exact_match_boost = 0.0
            for term in query_terms:
                if len(term) >= 4 and term in text_lower:
                    exact_match_boost += 0.03
            exact_match_boost = min(exact_match_boost, 0.18)
            semantic_score = min(1.0, semantic_score + exact_match_boost)

            # Pseudo rerank
            rerank_score = min(1.0, (semantic_score * 0.85) + exact_match_boost)

            # Entity overlap
            chunk_entities = set(e.lower() for e in metadata.get("E_i", []))
            overlap = query_terms.intersection(chunk_entities)
            entity_score = min(1.0, len(overlap) * 0.25) if chunk_entities else 0.0

            # Discourse score
            section = metadata.get("S_i", "General")
            if section == "Header/Structured":
                discourse_score = 1.0
            elif section == "Q&A":
                discourse_score = 0.80
            else:
                discourse_score = 0.55

            # Placeholder authority/contradiction
            authority_score = 1.0
            contradiction_risk = 0.0

            # Temporal term
            chunk_date = self._parse_best_chunk_date(text, metadata, created_at)

            q_t = 0.80 if time_sensitive else 0.20
            lambda_c = 0.20

            temporal_term = 1.0
            if chunk_date:
                try:
                    delta_t = (
                        float(now.year - chunk_date.year)
                        + float(now.month - chunk_date.month) / 12.0
                    )
                    if delta_t < 0:
                        delta_t = 0

                    temporal_term = q_t * math.exp(-lambda_c * delta_t) + (1 - q_t)

                    # NEW: stronger recency preference for "best practice/current/latest" queries
                    if temporal_priority:
                        temporal_term *= 1.20

                    # Keep within stable range
                    temporal_term = min(1.25, temporal_term)

                except Exception:
                    temporal_term = 1.0

            base_score = (
                (ALPHA * semantic_score)
                + (BETA * rerank_score)
                + (GAMMA * entity_score)
                + (DELTA * discourse_score)
                + (ETA * authority_score)
            )

            final_score = base_score * temporal_term * (1 - (MU * contradiction_risk))

            scored_results.append({
                "text": text,
                "semantic_score": semantic_score,
                "temporal_weight": temporal_term,
                "confidence_score": final_score,
                "timestamp": (
                    chunk_date.isoformat()
                    if hasattr(chunk_date, "isoformat")
                    else str(chunk_date)
                )
            })

        # IMPORTANT:
        # For temporal-priority queries, sort first by score, then by timestamp recency.
        if temporal_priority:
            scored_results.sort(
                key=lambda x: (x["confidence_score"], x["timestamp"]),
                reverse=True
            )
        else:
            scored_results.sort(key=lambda x: x["confidence_score"], reverse=True)

        top_results = scored_results[:top_k]

        if return_scores:
            return [
                {
                    "text": r["text"],
                    "confidence_score": r["confidence_score"],
                    "semantic_score": r["semantic_score"],
                    "temporal_weight": r["temporal_weight"],
                    "timestamp": r["timestamp"]
                }
                for r in top_results
            ]

        return [r["text"] for r in top_results]