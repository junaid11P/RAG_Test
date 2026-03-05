from app.db.mongodb import db
from datetime import datetime
from typing import List, Dict

class ChatService:
    @staticmethod
    async def save_message(user_id: str, doc_id: str, role: str, content: str, email: str = None, expires_at=None):
        """Saves a single chat message to the database."""
        now = datetime.utcnow()
        await db.db["conversation_history"].insert_one({
            "user_id": user_id,
            "email": email,
            "doc_id": doc_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M:%S"),
            "expires_at": expires_at
        })

    @staticmethod
    async def get_history(user_id: str, doc_id: str) -> List[Dict]:
        """Retrieves chat history for a specific document and user."""
        cursor = db.db["conversation_history"].find(
            {"user_id": user_id, "doc_id": doc_id}
        ).sort("timestamp", 1)
        
        messages = await cursor.to_list(length=100)
        # Format for frontend
        return [{"role": m["role"], "content": m["content"]} for m in messages]
