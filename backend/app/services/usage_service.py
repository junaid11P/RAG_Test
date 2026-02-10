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
    async def track_api_call(user_id: str):
        """Updates user's API call stats."""
        await db.db["users"].update_one(
            {"id": user_id},
            {"$inc": {"usage.api_calls": 1}}
        )

    @staticmethod
    async def get_usage(user_id: str):
        """Retrieves user's usage and calculates cost."""
        user = await db.db["users"].find_one({"id": user_id})
        if not user:
            return None
        
        usage = user.get("usage", {})
        
        # Simple Pricing Strategy (MVP)
        # $0.1 per MB
        # $0.01 per API call
        file_cost = (usage.get("total_bytes", 0) / (1024 * 1024)) * 0.1
        api_cost = usage.get("api_calls", 0) * 0.01
        
        return {
            "files_uploaded": usage.get("files_uploaded", 0),
            "api_calls": usage.get("api_calls", 0),
            "total_bytes": usage.get("total_bytes", 0),
            "estimated_bill": round(file_cost + api_cost, 2)
        }

    @staticmethod
    async def get_guest_query_count(guest_id: str) -> int:
        """Retrieves and increments the query count for a guest."""
        guest = await db.db["guests"].find_one_and_update(
            {"id": guest_id},
            {"$inc": {"queries": 1}},
            upsert=True,
            return_document=True
        )
        return guest.get("queries", 0)
