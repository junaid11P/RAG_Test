from app.db.mongodb import db

class UsageService:
    @staticmethod
    async def track_upload(user_id: str, file_size: int):
        """Updates user's upload stats."""
        await db.db["users"].update_one(
            {"id": user_id},
            {
                "$inc": {
                    "usage.files_uploaded": 1,
                    "usage.total_bytes": file_size
                }
            }
        )

    @staticmethod
    async def track_delete(user_id: str, file_size: int):
        """Decrements user's upload stats on deletion."""
        await db.db["users"].update_one(
            {"id": user_id},
            {
                "$inc": {
                    "usage.files_uploaded": -1,
                    "usage.total_bytes": -file_size
                }
            }
        )

    @staticmethod
    async def track_api_call(user_id: str, email: str = None):
        """Logs a query event for the user."""
        from datetime import datetime
        await db.db["query_logs"].insert_one({
            "user_id": user_id,
            "email": email,
            "timestamp": datetime.utcnow()
        })

    @staticmethod
    async def get_usage(user_id: str):
        """Retrieves user's usage and calculates cost dynamically."""
        user = await db.db["users"].find_one({"id": user_id})
        if not user:
            return None
        
        # Calculate actual files and bytes from the documents collection
        doc_stats = await db.db["documents"].aggregate([
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "bytes": {"$sum": "$file_size"}
                }
            }
        ]).to_list(length=1)

        actual_count = doc_stats[0]["count"] if doc_stats else 0
        actual_bytes = doc_stats[0]["bytes"] if doc_stats else 0
        
        # Calculate actual queries from the queries collection
        query_count = await db.db["query_logs"].count_documents({"user_id": user_id})
        
        # Simple Pricing Strategy (MVP)
        # $0.1 per MB
        # $0.01 per query
        file_cost = (actual_bytes / (1024 * 1024)) * 0.1
        api_cost = query_count * 0.01
        
        return {
            "files_uploaded": actual_count,
            "api_calls": query_count,
            "total_bytes": actual_bytes,
            "estimated_bill": round(file_cost + api_cost, 2)
        }

    @staticmethod
    async def get_guest_query_count(guest_id: str) -> int:
        """Retrieves and increments the query count for a guest."""
        from datetime import datetime
        guest = await db.db["guest_sessions"].find_one_and_update(
            {"id": guest_id},
            {
                "$inc": {"queries": 1},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True,
            return_document=True
        )
        return guest.get("queries", 0)
